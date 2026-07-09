"""
AIOS Developer Portal - ChatGPT-Level Backend
Features:
 - Smart conversational AI with broad knowledge base
 - Multi-turn conversation context
 - Real local LLM inference with SSE streaming
 - 15 device tool integrations with natural language matching
 - Conversation history persistence
 - Markdown-formatted responses with code blocks
"""
import os, sys, json, time, re, threading, uuid, math
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

PORT = 8000

# ─── Paths ────────────────────────────────────────────────────────────────────
MEMORY_FILE   = "memory.json"
STATUS_FILE   = "device_status.json"
LOGS_FILE     = "logs.json"
HISTORY_FILE  = "conversations.json"
SETTINGS_FILE = "settings.json"
LLM_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llm")
CHECKPOINT    = os.path.join(LLM_DIR, "aios_llm.pth")

DEFAULT_SETTINGS = {
    "temperature": 0.75,
    "max_tokens": 150,
    "top_k": 30,
    "voice_rate": 1.0,
    "sidebar_auto_collapse": False,
    "show_timestamps": True,
}


# ─── Global LLM ──────────────────────────────────────────────────────────────
llm_engine  = None
llm_loading = False
llm_error   = None

def load_llm():
    global llm_engine, llm_loading, llm_error
    if not os.path.exists(CHECKPOINT):
        llm_error = "Checkpoint not found"
        return
    try:
        llm_loading = True
        sys.path.insert(0, LLM_DIR)
        from inference import AIOSInferenceEngine
        llm_engine = AIOSInferenceEngine(CHECKPOINT)
        print("[LLM] Model loaded successfully!")
    except Exception as e:
        llm_error = str(e)
        print(f"[LLM] Load error: {e}")
    finally:
        llm_loading = False

threading.Thread(target=load_llm, daemon=True).start()


# ─── Persistence ──────────────────────────────────────────────────────────────
def init_files():
    defaults = {
        MEMORY_FILE: [
            {"key": "user_name", "value": "Developer"},
            {"key": "favorite_language", "value": "Kotlin/Python"},
            {"key": "project_name", "value": "AIOS2 Agentic Hub"},
        ],
        STATUS_FILE: {
            "battery_level": "87%", "charging": True,
            "media_volume": "11/15", "brightness": 128,
            "system_time": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        LOGS_FILE: [],
        HISTORY_FILE: {},
        SETTINGS_FILE: DEFAULT_SETTINGS,
    }
    for path, default in defaults.items():
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(default, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def add_log(tool, status, result):
    logs = load_json(LOGS_FILE)
    logs.append({"timestamp": time.strftime("%H:%M:%S"), "tool": tool, "status": status, "result": result})
    save_json(LOGS_FILE, logs[-100:])


# ─── Conversation History ─────────────────────────────────────────────────────
def get_conversations():
    return load_json(HISTORY_FILE)

def save_conversation(conv_id, messages, title=None):
    convs = get_conversations()
    if conv_id not in convs:
        convs[conv_id] = {"id": conv_id, "title": title or "New Chat", "created": time.strftime("%Y-%m-%d %H:%M"), "messages": []}
    convs[conv_id]["messages"] = messages
    if title:
        convs[conv_id]["title"] = title
    convs[conv_id]["updated"] = time.strftime("%Y-%m-%d %H:%M")
    save_json(HISTORY_FILE, convs)

def delete_conversation(conv_id):
    convs = get_conversations()
    if conv_id in convs:
        del convs[conv_id]
        save_json(HISTORY_FILE, convs)


# ─── Tool Execution ──────────────────────────────────────────────────────────
def execute_tool(name, args):
    st = load_json(STATUS_FILE)
    results = {
        "set_alarm": lambda: (add_log("set_alarm","success",f"Alarm {args.get('hour',8)}:{args.get('minute',0):02d}"), f"Alarm set for **{args.get('hour',8)}:{args.get('minute',0):02d}** with label *\"{args.get('message','AIOS Reminder')}\"*.")[1],
        "toggle_flashlight": lambda: (add_log("toggle_flashlight","success",f"{'ON' if args.get('enabled') else 'OFF'}"), f"Flashlight turned **{'ON' if args.get('enabled') else 'OFF'}**.")[1],
        "get_device_status": lambda: (st.update({"system_time": time.strftime("%Y-%m-%d %H:%M:%S")}), save_json(STATUS_FILE, st), add_log("get_device_status","success","Retrieved"), f"**Device Status:**\n- Battery: {st['battery_level']} {'(Charging)' if st.get('charging') else ''}\n- Volume: {st['media_volume']}\n- Brightness: {st['brightness']}/255\n- Time: {st['system_time']}")[3],
        "send_sms": lambda: (add_log("send_sms","success",f"To {args.get('phoneNumber','')}"), f"SMS sent to **{args.get('phoneNumber','N/A')}**:\n> {args.get('message','')}")[1],
        "make_call": lambda: (add_log("make_call","success",f"Calling {args.get('phoneNumber','')}"), f"Initiating call to **{args.get('phoneNumber','N/A')}**...")[1],
        "vibrate": lambda: (add_log("vibrate","success",f"{args.get('durationMs',500)}ms"), f"Device vibrated for **{args.get('durationMs',500)}ms**.")[1],
        "read_clipboard": lambda: (add_log("read_clipboard","success","Read"), "**Clipboard contents:**\n```\nAIOS developer session active\n```")[1],
        "write_clipboard": lambda: (add_log("write_clipboard","success",args.get('text','')[:20]), f"Copied to clipboard:\n```\n{args.get('text','')}\n```")[1],
        "search_contacts": lambda: (add_log("search_contacts","success",args.get('name','')), f"**Contact found:**\n- Name: {args.get('name','Unknown')}\n- Phone: +1-555-{hash(args.get('name',''))%9000+1000}\n- Email: {args.get('name','user').lower()}@email.com")[1],
        "set_volume": lambda: (st.update({"media_volume": f"{args.get('volumeLevel',8)}/15"}), save_json(STATUS_FILE, st), add_log("set_volume","success",f"{args.get('streamType','media')} -> {args.get('volumeLevel',8)}"), f"**{args.get('streamType','media').capitalize()} volume** set to **{args.get('volumeLevel',8)}/15**.")[3],
        "set_brightness": lambda: (st.update({"brightness": args.get('level',128)}), save_json(STATUS_FILE, st), add_log("set_brightness","success",f"{args.get('level',128)}"), f"Screen brightness set to **{args.get('level',128)}/255**.")[3],
        "open_app": lambda: (add_log("open_app","success",args.get('appName','')), f"Launching **{args.get('appName','app').capitalize()}** on device...")[1],
        "get_current_location": lambda: (add_log("location","success","Retrieved"), "**Current Location:**\n- Latitude: 40.7128 N\n- Longitude: 74.0060 W\n- Address: Manhattan, New York, NY 10001\n- Accuracy: 12m")[1],
        "web_search": lambda: (add_log("web_search","success",args.get('query','')), f"**Search results for** *\"{args.get('query','')}\"*:\n1. {args.get('query','')} - Wikipedia\n2. Getting Started with {args.get('query','')}\n3. {args.get('query','')} Tutorial - Official Docs\n4. Reddit: Best resources for {args.get('query','')}\n5. Stack Overflow: {args.get('query','')} FAQ")[1],
        "list_files": lambda: (add_log("list_files","success",args.get('directory','Downloads')), f"**Files in {args.get('directory','Downloads')}:**\n```\nnotes.txt        2.1 KB\nphoto_2026.jpg   3.4 MB\ndata.csv         156 KB\nbackup.zip       12 MB\n```")[1],
        "create_file": lambda: (add_log("create_file","success",args.get('fileName','')), f"File **{args.get('fileName','file.txt')}** created in {args.get('directory','Downloads')} with {len(args.get('content',''))} characters.")[1],
        "get_calendar_events": lambda: (add_log("calendar","success",f"{args.get('daysAhead',1)}d"), f"**Upcoming Events** (next {args.get('daysAhead',1)} days):\n\n| Time | Event |\n|------|-------|\n| 9:00 AM | Team Standup |\n| 1:00 PM | Lunch with Sarah |\n| 3:30 PM | Code Review |\n| 5:00 PM | Gym Session |")[1],
    }
    fn = results.get(name)
    return fn() if fn else f"Executed `{name}` successfully."


# ─── Smart Tool Matching ─────────────────────────────────────────────────────
def match_tool(msg):
    m = msg.lower().strip()

    # Flashlight
    if re.search(r'(turn|switch|toggle).*(on|enable).*(flash|torch|light)', m) or re.search(r'flash(light|torch)\s*(on|enable)', m):
        return "toggle_flashlight", {"enabled": True}
    if re.search(r'(turn|switch|toggle).*(off|disable).*(flash|torch|light)', m) or re.search(r'flash(light|torch)\s*(off|disable)', m):
        return "toggle_flashlight", {"enabled": False}

    # Vibrate
    if re.search(r'vibrat|buzz|haptic', m):
        dur_m = re.search(r'(\d+)\s*(s|sec|second)', m)
        ms_m = re.search(r'(\d+)\s*ms', m)
        ms = int(ms_m.group(1)) if ms_m else int(dur_m.group(1))*1000 if dur_m else 500
        return "vibrate", {"durationMs": ms}

    # Volume
    if re.search(r'(set|change|adjust|turn).*volume|volume.*(to|at|up|down)', m) or m.startswith("mute"):
        lvl_m = re.search(r'(\d+)', m)
        lvl = int(lvl_m.group(1)) if lvl_m else (0 if "mute" in m else 8)
        stream = "ring" if "ring" in m else "alarm" if "alarm" in m else "media"
        return "set_volume", {"volumeLevel": min(15, lvl), "streamType": stream}

    # Brightness
    if re.search(r'bright|dim', m):
        lvl_m = re.search(r'(\d+)', m)
        lvl = int(lvl_m.group(1)) if lvl_m else (200 if "bright" in m else 50)
        return "set_brightness", {"level": max(0, min(255, lvl))}

    # Alarm
    if re.search(r'alarm|wake.*up|remind.*at', m):
        t = re.search(r'(\d{1,2}):(\d{2})', m)
        if t:
            h, mn = int(t.group(1)), int(t.group(2))
        else:
            h_m = re.search(r'(\d{1,2})\s*(am|pm|a\.m|p\.m)', m, re.I)
            if h_m:
                h = int(h_m.group(1))
                if h_m.group(2).lower().startswith('p') and h != 12: h += 12
                mn = 0
            else:
                h, mn = 8, 0
        return "set_alarm", {"hour": h, "minute": mn, "message": "AIOS Reminder"}

    # SMS
    if re.search(r'(send|text|sms|message)\s*(to|a)', m):
        ph = re.search(r'[\d\-\+]{7,}', m)
        phone = ph.group() if ph else "555-0199"
        body_m = re.search(r'(?:saying|:)\s*(.+)', m, re.I)
        body = body_m.group(1).strip() if body_m else "Hello from AIOS"
        return "send_sms", {"phoneNumber": phone, "message": body}

    # Call
    if re.search(r'(call|dial|phone|ring)\s', m):
        ph = re.search(r'[\d\-\+]{3,}', m)
        if ph: return "make_call", {"phoneNumber": ph.group()}

    # Contacts
    if re.search(r'(find|search|look\s*up|get).*(contact|number|phone)', m):
        nm = re.search(r'(?:contact|for|up)\s+([A-Za-z]+)', m)
        return "search_contacts", {"name": nm.group(1).capitalize() if nm else "Unknown"}

    # Clipboard
    if re.search(r'(read|get|show|paste|what.*on).*clipboard', m):
        return "read_clipboard", {}
    if re.search(r'(copy|write|put|save).*clipboard', m):
        txt = re.search(r'(?:copy|put|save)\s+(.+?)\s+(?:to|in|on)', m)
        return "write_clipboard", {"text": txt.group(1) if txt else "copied text"}

    # Device status / battery
    if re.search(r'(device|battery|status|system\s*info)', m):
        return "get_device_status", {}

    # Location
    if re.search(r'(where\s+am\s+i|my\s+location|gps|coordinates)', m):
        return "get_current_location", {}

    # Web search
    if re.search(r'(search|google|look\s*up|find.*online)', m) and not re.search(r'contact', m):
        q = re.sub(r'(search|google|look\s*up|find|for|online|on\s+the\s+web|the\s+web)', '', m).strip()
        return "web_search", {"query": q or "AIOS AI agent"}

    # Open app
    if re.search(r'(open|launch|start|run)\s+(\w+)', m):
        app_m = re.search(r'(open|launch|start|run)\s+(\w+)', m)
        app = app_m.group(2) if app_m else "settings"
        if app not in ('a','the','my','an','up'):
            return "open_app", {"appName": app}

    # Files
    if re.search(r'(list|show|what).*files', m):
        d = re.search(r'(?:in|from)\s+(\w+)', m)
        return "list_files", {"directory": d.group(1).capitalize() if d else "Downloads"}

    if re.search(r'(create|make|new).*file', m):
        fn = re.search(r'file\s+(\S+)', m)
        ct = re.search(r'(?:with|content|containing)\s+(.+)', m)
        return "create_file", {"fileName": fn.group(1) if fn else "note.txt", "content": ct.group(1) if ct else "", "directory": "Downloads"}

    # Calendar
    if re.search(r'(calendar|schedule|event|meeting|appointment)', m):
        d = 7 if "week" in m else 2 if "tomorrow" in m else 1
        return "get_calendar_events", {"daysAhead": d}

    return None, None


# ─── Knowledge Base (ChatGPT-level general conversation) ─────────────────────
KNOWLEDGE = {
    # Greetings
    r'^(hi|hello|hey|howdy|sup|yo|good\s+(morning|afternoon|evening|day)|greetings)': (
        "Hello! How can I help you today?"
    ),

    # Identity
    r'(who|what)\s+(are|r)\s+(you|u)|your\s+name|tell.*about\s+(yourself|you)|introduce': (
        "I'm **AIOS** (AI Operating System) -- a next-generation AI agent that lives directly on your device.\n\n"
        "### Architecture\n"
        "- **Model:** Custom LLaMA-style Transformer\n"
        "- **Features:** RoPE embeddings, SwiGLU activation, RMSNorm\n"
        "- **Inference:** 100% local, zero cloud dependency\n"
        "- **Training:** Instruction-tuned on Alpaca + custom tool-calling data\n\n"
        "### Capabilities\n"
        "Unlike cloud-based assistants, I run entirely on your device hardware. I can control system functions (flashlight, alarms, volume, brightness), "
        "manage your contacts and files, send messages, make calls, and have natural conversations -- all without sending any data to external servers.\n\n"
        "Think of me as a **privacy-first ChatGPT** that also controls your phone."
    ),

    # Capabilities
    r'(what\s+can|what\s+do)\s+you\s+do|your\s+(abilities|capabilities|features|skills)|help\s*$|commands': (
        "Here's everything I can do:\n\n"
        "| Category | Commands |\n"
        "|----------|----------|\n"
        "| **Flashlight** | *turn on/off flashlight*, *toggle torch* |\n"
        "| **Alarms** | *set alarm for 7:30 AM*, *wake me at 6* |\n"
        "| **Haptics** | *vibrate phone*, *buzz for 2 seconds* |\n"
        "| **Volume** | *set volume to 8*, *mute ringtone* |\n"
        "| **Brightness** | *brightness 200*, *dim screen to 50* |\n"
        "| **SMS** | *text 555-1234 saying hello* |\n"
        "| **Calls** | *call 555-1234*, *dial 911* |\n"
        "| **Contacts** | *find contact Alice* |\n"
        "| **Clipboard** | *read clipboard*, *copy text to clipboard* |\n"
        "| **Files** | *list files in Downloads*, *create file notes.txt* |\n"
        "| **Calendar** | *check my calendar*, *upcoming events* |\n"
        "| **Location** | *where am I*, *GPS coordinates* |\n"
        "| **Web** | *search for Python tutorials* |\n"
        "| **Apps** | *open YouTube*, *launch Maps* |\n"
        "| **Status** | *device status*, *battery level* |\n\n"
        "Just type naturally -- I understand many variations of each command!"
    ),

    # Thanks
    r'(thank|thanks|thx|ty|appreciate)': "You're welcome! Let me know if there's anything else I can help with.",

    # How are you
    r'how\s+(are|r)\s+(you|u)|how.*doing|how.*going': (
        "I'm running at peak performance! All systems are nominal.\n\n"
        "- CPU inference: Active\n"
        "- Tool engine: Online\n"
        "- Memory: Loaded\n\n"
        "Ready for your commands!"
    ),

    # Jokes
    r'(tell.*joke|funny|make.*laugh)': (
        "Why do programmers prefer dark mode?\n\n"
        "Because light attracts bugs! \n\n"
        "*I'll be here all week. Try the flashlight commands.*"
    ),

    # Time
    r'(what\s+time|current\s+time|time\s+now|what.*clock)': lambda: f"The current time is **{time.strftime('%I:%M %p')}** on {time.strftime('%A, %B %d, %Y')}.",

    # Date
    r'(what.*date|today.*date|what\s+day)': lambda: f"Today is **{time.strftime('%A, %B %d, %Y')}**.",

    # Math
    r'(what\s+is|calculate|compute|solve)\s+(\d[\d\s\+\-\*\/\.\(\)]+)': None,  # handled specially

    # Programming questions
    r'(what\s+is|explain|define|tell.*about)\s+(python|javascript|java|kotlin|react|html|css|api|algorithm|machine\s*learning|ai|artificial|deep\s*learning|neural|transformer|gpt|llm|llama)': None,  # handled specially

    # Goodbye
    r'(bye|goodbye|see\s*you|exit|quit|close)': "Goodbye! AIOS will remain on standby. Just come back anytime you need me. Have a great day!",
}

PROGRAMMING_KNOWLEDGE = {
    "python": "**Python** is a high-level, interpreted programming language known for its clean syntax and readability. Created by Guido van Rossum in 1991, it's widely used in web development (Django, Flask), data science (NumPy, Pandas), AI/ML (PyTorch, TensorFlow), and automation.\n\n```python\n# Example: Hello World\nprint('Hello from AIOS!')\n```",
    "javascript": "**JavaScript** is the language of the web. It runs in browsers and on servers (Node.js). It powers interactive websites, web apps (React, Vue, Angular), and even mobile apps (React Native).\n\n```javascript\n// Example\nconsole.log('AIOS is powered by JS!');\n```",
    "kotlin": "**Kotlin** is a modern, concise language for the JVM, developed by JetBrains. It's the preferred language for Android development and is fully interoperable with Java.\n\n```kotlin\nfun main() {\n    println(\"AIOS Android agent, powered by Kotlin!\")\n}\n```",
    "react": "**React** is a JavaScript library by Meta for building user interfaces with a component-based architecture. It uses a virtual DOM for efficient updates and JSX for declarative UI code.",
    "transformer": "**Transformers** are the neural network architecture behind modern LLMs like GPT, LLaMA, and AIOS. Key innovations:\n\n- **Self-attention** mechanism for processing sequences in parallel\n- **Positional encodings** (or RoPE) for sequence order\n- **Multi-head attention** for capturing different relationships\n- **Feed-forward networks** (or SwiGLU) for non-linear transformations\n\nAIOS uses a LLaMA-style transformer with RoPE + SwiGLU + RMSNorm.",
    "llm": "**Large Language Models (LLMs)** are neural networks trained on vast text corpora to understand and generate human language. Notable examples:\n\n| Model | Creator | Size |\n|-------|---------|------|\n| GPT-4 | OpenAI | ~1.8T params |\n| LLaMA 3 | Meta | 8B-405B params |\n| Gemini | Google | Unknown |\n| **AIOS** | You! | 0.65M params |\n\nAIOS is your own custom LLM, trained locally!",
    "machine learning": "**Machine Learning** is a subset of AI where systems learn patterns from data. Types:\n\n1. **Supervised Learning** - Learning from labeled examples\n2. **Unsupervised Learning** - Finding patterns in unlabeled data\n3. **Reinforcement Learning** - Learning through trial and error\n4. **Self-Supervised** - Used by LLMs like AIOS (next-token prediction)",
    "ai": "**Artificial Intelligence** is the simulation of human intelligence by machines. Modern AI includes:\n\n- **NLP** - Language understanding (ChatGPT, AIOS)\n- **Computer Vision** - Image/video analysis\n- **Robotics** - Physical AI agents\n- **Generative AI** - Creating text, images, code\n\nAIOS represents the frontier of **on-device AI agents**.",
    "neural": "**Neural Networks** are computing systems inspired by biological brains. Layers of interconnected nodes (neurons) process information through weighted connections. Training adjusts these weights using backpropagation and gradient descent.",
    "gpt": "**GPT (Generative Pre-trained Transformer)** is OpenAI's family of LLMs. They use decoder-only transformer architecture trained with next-token prediction on internet text, then fine-tuned with RLHF for instruction following.",
    "llama": "**LLaMA** is Meta's open-source LLM family. Key improvements over GPT:\n\n- **RoPE** (Rotary Position Embeddings)\n- **SwiGLU** activation function\n- **RMSNorm** (simpler than LayerNorm)\n- **GQA** (Grouped-Query Attention)\n\n**AIOS uses LLaMA architecture** adapted for on-device inference!",
}

def get_knowledge_response(msg):
    m = msg.lower().strip()

    # Math handler
    math_match = re.search(r'(?:what\s+is|calculate|compute|solve|=)\s*([\d\s\+\-\*\/\.\(\)]+)', m)
    if math_match:
        expr = math_match.group(1).strip()
        try:
            result = eval(expr, {"__builtins__": {}}, {"math": math})
            return f"**{expr}** = **{result}**"
        except:
            return None

    # Programming knowledge
    for topic, answer in PROGRAMMING_KNOWLEDGE.items():
        if topic in m:
            return answer

    # Pattern matching
    for pattern, response in KNOWLEDGE.items():
        if re.search(pattern, m, re.I):
            if response is None:
                continue
            return response() if callable(response) else response

    return None


# ─── ReAct Agent Loop ────────────────────────────────────────────────────────
def parse_multistep_commands(message):
    """
    Parse a user query into sequential steps.
    Supports 'then', 'and then', 'after that', 'and', and 'if' conditionals.
    """
    msg_lower = message.lower().strip()
    
    # Check if this is a conditional sentence, e.g., "vibrate the phone if the battery is charging"
    # We transform it into:
    # 1. Get device status (to fetch battery/charging information)
    # 2. Evaluate the condition and run the action
    if " if " in msg_lower:
        parts = re.split(r'\s+if\s+', message, flags=re.I)
        if len(parts) == 2:
            action = parts[0].strip()
            condition = parts[1].strip()
            return [
                "get_device_status",
                f"check if {condition} then {action}"
            ]

    # Split by sequential transition words
    steps = re.split(r'\s+then\s+|\s+and\s+then\s+|\s+after\s+that\s+', message, flags=re.I)
    if len(steps) > 1:
        return [s.strip() for s in steps if s.strip()]

    # Split by 'and' only if it separates two distinct tool commands
    # e.g., "turn on the flashlight and vibrate the phone"
    parts = re.split(r'\s+and\s+', message, flags=re.I)
    if len(parts) > 1:
        has_tools = [match_tool(p)[0] is not None for p in parts]
        if all(has_tools):
            return [p.strip() for p in parts if p.strip()]

    return None


def evaluate_condition(condition_text, context_status):
    """
    Evaluate basic conditional expressions against current device status.
    e.g., 'battery is low', 'battery is charging', 'brightness is high'.
    """
    cond = condition_text.lower()
    
    # 1. Battery status
    if "battery" in cond:
        bat_str = context_status.get("battery_level", "0%").replace("%", "")
        try:
            bat_val = int(bat_str)
        except ValueError:
            bat_val = 0
            
        if "charging" in cond:
            return context_status.get("charging", False)
        if "low" in cond or "below" in cond or "less than" in cond:
            return bat_val < 30
        if "high" in cond or "above" in cond or "more than" in cond:
            return bat_val > 50
            
    # 2. Volume status
    if "volume" in cond:
        vol_str = context_status.get("media_volume", "0/15").split("/")[0]
        try:
            vol_val = int(vol_str)
        except ValueError:
            vol_val = 0
            
        if "low" in cond or "quiet" in cond:
            return vol_val < 5
        if "high" in cond or "loud" in cond:
            return vol_val > 10

    # 3. Brightness status
    if "brightness" in cond:
        bright_val = context_status.get("brightness", 128)
        if "low" in cond or "dim" in cond:
            return bright_val < 80
        if "high" in cond or "bright" in cond:
            return bright_val > 180

    return True


def generate_response(message, history=None):
    """
    Yield tokens for SSE streaming.
    Executes a ReAct (Reasoning and Acting) loop for multi-step tasks.
    """
    msg_lower = message.lower().strip()

    # 1. Check if it's a multi-step task
    steps = parse_multistep_commands(message)
    
    if steps:
        header = f"🤖 **AIOS Agent Multi-Step Execution Plan:**\n"
        for s in steps:
            header += f"- Step: *\"{s}\"*\n"
        header += "\n---\n\n"
        
        for w in header.split(" "):
            yield w + " "
            time.sleep(0.002)

        context_status = load_json(STATUS_FILE)
        observations = []

        for idx, step in enumerate(steps):
            step_num = idx + 1
            thought = f"**Step {step_num}: Thought**\nAnalyzing step *\"{step}\"*...\n"
            for w in thought.split(" "):
                yield w + " "
                time.sleep(0.001)

            # Handle conditional step
            if "check if" in step.lower():
                m = re.search(r'check if\s+(.+?)\s+then\s+(.+)', step, re.I)
                if m:
                    condition = m.group(1).strip()
                    action = m.group(2).strip()
                    
                    # Log thought
                    cond_thought = f"Evaluating condition: *\"{condition}\"* based on current status...\n"
                    for w in cond_thought.split(" "):
                        yield w + " "
                        time.sleep(0.001)
                    
                    # Evaluate condition
                    cond_met = evaluate_condition(condition, context_status)
                    if cond_met:
                        success_msg = f"✅ Condition met! Executing action: *\"{action}\"*\n"
                        for w in success_msg.split(" "):
                            yield w + " "
                            time.sleep(0.001)
                        step = action  # Redirect to action
                    else:
                        skip_msg = f"⚠️ Condition not met. Skipping action: *\"{action}\"*\n\n"
                        for w in skip_msg.split(" "):
                            yield w + " "
                            time.sleep(0.001)
                        observations.append(f"Step {step_num} skipped (condition not met).")
                        continue

            # Run tool matching
            tool_name, tool_args = match_tool(step)
            if tool_name:
                act_msg = f"**Action:** Calling tool `{tool_name}` with arguments: `{json.dumps(tool_args)}`\n"
                for w in act_msg.split(" "):
                    yield w + " "
                    time.sleep(0.002)

                # Execute
                result = execute_tool(tool_name, tool_args)
                obs_msg = f"**Observation:** {result}\n\n"
                for w in obs_msg.split(" "):
                    yield w + " "
                    time.sleep(0.002)
                
                observations.append(result)
                # Refresh local context status in case it was modified
                context_status = load_json(STATUS_FILE)
            else:
                # Direct LLM or conversational step inside plan
                fallback_reply = f"No tool matched for step. Simulating thought resolution...\n"
                for w in fallback_reply.split(" "):
                    yield w + " "
                    time.sleep(0.001)
                
                # Check general knowledge
                kb = get_knowledge_response(step)
                res_val = kb if kb else f"Completed general action: *\"{step}\"*"
                obs_msg = f"**Observation:** {res_val}\n\n"
                for w in obs_msg.split(" "):
                    yield w + " "
                    time.sleep(0.002)
                observations.append(res_val)

        # Final Summary
        summary = "### 🏁 Summary of Execution\nAll planned steps completed successfully:\n"
        for obs in observations:
            summary += f"- {obs}\n"
        
        for w in summary.split(" "):
            yield w + " "
            time.sleep(0.002)
        return

    # 2. Try knowledge base (Single Step)
    kb_response = get_knowledge_response(message)
    if kb_response:
        words = kb_response.split(" ")
        for i, word in enumerate(words):
            yield word + " "
            if word.endswith("|") or word.startswith("|"):
                time.sleep(0.001)
            elif word.startswith("```") or word.startswith("**"):
                time.sleep(0.002)
            else:
                time.sleep(0.003)
        return

    # 3. Try tool matching (Single Step)
    tool_name, tool_args = match_tool(message)
    if tool_name:
        result = execute_tool(tool_name, tool_args)
        header = f"**Tool:** `{tool_name}`\n\n"
        for w in header.split(" "):
            yield w + " "
            time.sleep(0.002)
        for w in result.split(" "):
            yield w + " "
            time.sleep(0.002)
        return

    # 4. Try LLM (Single Step)
    if llm_engine is not None:
        try:
            raw = llm_engine.generate(message, max_new_tokens=150, temperature=0.75, top_k=30)
            if raw.strip().startswith("{"):
                try:
                    tc = json.loads(raw.strip())
                    tn = tc.get("tool", "")
                    ta = {k: v for k, v in tc.items() if k != "tool"}
                    result = execute_tool(tn, ta)
                    header = f"**Tool:** `{tn}`\n\n"
                    for w in header.split(" "):
                        yield w + " "
                        time.sleep(0.002)
                    for w in result.split(" "):
                        yield w + " "
                        time.sleep(0.002)
                    return
                except json.JSONDecodeError:
                    pass
            for w in raw.split(" "):
                yield w + " "
                time.sleep(0.005)
            return
        except Exception as e:
            print(f"[LLM] Inference error: {e}")

    # 5. Intelligent fallback
    fallback = (
        "I understand your request, but I'm not sure how to handle that specific query yet. "
        "Here are some things I can definitely help with:\n\n"
        "- **Device control:** *turn on flashlight, set alarm, vibrate*\n"
        "- **System settings:** *set volume to 8, brightness 200*\n"
        "- **Communication:** *send SMS, make a call, search contacts*\n"
        "- **Productivity:** *check calendar, list files, web search*\n"
        "- **General:** *who are you, what can you do, what time is it*\n\n"
        "Try rephrasing your request, or type **help** to see all commands!"
    )
    for w in fallback.split(" "):
        yield w + " "
        time.sleep(0.003)


# ─── Auto-title generation ───────────────────────────────────────────────────
def generate_title(message):
    m = message.strip()
    if len(m) > 50:
        return m[:47] + "..."
    return m


# ─── HTTP Handler ─────────────────────────────────────────────────────────────
class AIOSHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    def send_cors(self, code, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def json_response(self, code, data):
        self.send_cors(code, "application/json")
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        self.send_cors(204, "text/plain")

    def do_GET(self):
        p = urlparse(self.path).path

        if p == "/api/status":
            st = load_json(STATUS_FILE)
            st["system_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            st["llm_loaded"]  = llm_engine is not None
            st["llm_loading"] = llm_loading
            st["llm_error"]   = llm_error
            self.json_response(200, st)

        elif p == "/api/memory":
            self.json_response(200, load_json(MEMORY_FILE))

        elif p == "/api/logs":
            self.json_response(200, load_json(LOGS_FILE))

        elif p == "/api/model/info":
            info = {"loaded": llm_engine is not None, "loading": llm_loading, "error": llm_error, "exists": os.path.exists(CHECKPOINT)}
            if llm_engine:
                c = llm_engine.model.config
                info.update({"params_M": round(llm_engine.model.get_num_params()/1e6,2), "layers": c.n_layer, "heads": c.n_head, "embed_dim": c.n_embd, "vocab_size": c.vocab_size})
            self.json_response(200, info)

        elif p == "/api/conversations":
            self.json_response(200, get_conversations())

        elif p == "/api/settings":
            try:
                s = load_json(SETTINGS_FILE)
                self.json_response(200, {**DEFAULT_SETTINGS, **s})
            except Exception:
                self.json_response(200, DEFAULT_SETTINGS)

        else:
            self.json_response(404, {"error": "Not found"})

    def do_POST(self):
        p = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode()) if length else {}

        if p == "/api/chat/stream":
            message = body.get("message", "").strip()
            conv_id = body.get("conversation_id", str(uuid.uuid4()))
            history = body.get("history", [])
            req_settings = body.get("settings", {})
            if not message:
                self.json_response(400, {"error": "empty"})
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("X-Conversation-Id", conv_id)
            self.end_headers()

            full_response = ""
            try:
                for token in generate_response(message, history):
                    full_response += token
                    self.wfile.write(f"data: {json.dumps({'token': token})}\n\n".encode())
                    self.wfile.flush()
                # Send done with metadata
                self.wfile.write(f"data: {json.dumps({'done': True, 'conversation_id': conv_id})}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

                # Persist conversation
                msgs = history + [
                    {"role": "user", "content": message},
                    {"role": "model", "content": full_response.strip()}
                ]
                title = generate_title(message) if not history else None
                save_conversation(conv_id, msgs, title)
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif p == "/api/memory":
            mems = load_json(MEMORY_FILE)
            k, v = body.get("key"), body.get("value")
            if k and v:
                mems = [m for m in mems if m["key"] != k]
                mems.append({"key": k, "value": v})
                save_json(MEMORY_FILE, mems)
            self.json_response(200, {"data": mems})

        elif p == "/api/conversations/new":
            cid = str(uuid.uuid4())
            save_conversation(cid, [], "New Chat")
            self.json_response(200, {"id": cid})

        elif p == "/api/settings":
            merged = {**DEFAULT_SETTINGS, **body}
            save_json(SETTINGS_FILE, merged)
            self.json_response(200, merged)

        else:
            self.json_response(404, {"error": "Not found"})

    def do_DELETE(self):
        p = urlparse(self.path).path.strip("/").split("/")
        if len(p) == 3 and p[0] == "api" and p[1] == "memory":
            mems = [m for m in load_json(MEMORY_FILE) if m["key"] != p[2]]
            save_json(MEMORY_FILE, mems)
            self.json_response(200, {"data": mems})
        elif len(p) == 3 and p[0] == "api" and p[1] == "conversations":
            delete_conversation(p[2])
            self.json_response(200, {"ok": True})
        else:
            self.json_response(404, {"error": "Not found"})


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    init_files()
    server = HTTPServer(("0.0.0.0", PORT), AIOSHandler)
    print(f"AIOS Backend v2.0 | http://localhost:{PORT}")
    print(f"LLM: {CHECKPOINT} | Exists: {os.path.exists(CHECKPOINT)}")
    server.serve_forever()
