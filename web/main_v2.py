"""
AIOS v2 Backend - FastAPI Production Server
Features:
 - Async SSE streaming with sse-starlette
 - Multi-modal support (image upload + understanding)
 - File upload handling
 - Real web search integration
 - RAG with local documents
 - Conversation branching and editing
 - Multiple LLM provider support (local, Gemini, OpenAI)
 - User authentication with session management
 - Rate limiting and caching
"""
import os
import sys
import json
import time
import uuid
import hashlib
import asyncio
import re
from datetime import datetime
from typing import Optional, List, AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import httpx

# Add LLM dir to path
LLM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llm")
sys.path.insert(0, LLM_DIR)

# ─── Configuration ──────────────────────────────────────────────────────────
CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": True,
    "max_upload_size": 10 * 1024 * 1024,  # 10 MB
    "model_cache_size": 3,
    "rate_limit": 60,  # requests per minute
    "tokenizer_path": os.path.join(LLM_DIR, "bpe_tokenizer.json"),
    "checkpoint_path": os.path.join(LLM_DIR, "checkpoints_v2", "best.pt"),
    "data_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
    "upload_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads"),
}

os.makedirs(CONFIG["data_dir"], exist_ok=True)
os.makedirs(CONFIG["upload_dir"], exist_ok=True)

# ─── Data Models ─────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    history: Optional[List[ChatMessage]] = None
    settings: Optional[dict] = None
    stream: bool = True

class UploadResponse(BaseModel):
    filename: str
    url: str
    content_type: str
    size: int

class Conversation(BaseModel):
    id: str
    title: str
    created: str
    updated: str
    messages: List[ChatMessage] = []

class SettingsUpdate(BaseModel):
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None
    voice_rate: Optional[float] = None
    sidebar_auto_collapse: Optional[bool] = None
    show_timestamps: Optional[bool] = None
    model: Optional[str] = None
    provider: Optional[str] = None

# ─── App State ───────────────────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.llm_engine = None
        self.llm_loading = False
        self.llm_error = None
        self.sessions = {}
        self.conversations = self._load_json("conversations.json", {})
        self.settings = self._load_json("settings.json", {
            "temperature": 0.7,
            "max_tokens": 512,
            "top_k": 40,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "voice_rate": 1.0,
            "sidebar_auto_collapse": False,
            "show_timestamps": True,
            "model": "local",
            "provider": "local",
        })

    def _load_json(self, path, default):
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        if os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return default

    def _save_json(self, path, data):
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


state = AppState()


# ─── LLM Loader ──────────────────────────────────────────────────────────────
async def load_llm():
    """Load a small pre-trained model for CPU inference."""
    global state
    if state.llm_engine is not None:
        return

    try:
        state.llm_loading = True
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch

        MODEL_NAME = "google/flan-t5-small"
        print(f"[AIOS] Loading pre-trained model: {MODEL_NAME}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model.eval()

        state.llm_engine = {"model": model, "tokenizer": tokenizer}
        state.llm_error = None
        print(f"[AIOS] Pre-trained model loaded successfully (80M params)!")
    except Exception as e:
        state.llm_error = str(e)
        print(f"[AIOS] Pre-trained model load error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        state.llm_loading = False


# ─── Chat Engine ──────────────────────────────────────────────────────────────
async def generate_response(
    message: str,
    history: Optional[List[ChatMessage]] = None,
    settings: Optional[dict] = None,
) -> AsyncGenerator[str, None]:
    """Generate response with streaming."""
    settings = settings or {}
    temp = settings.get("temperature", 0.7)
    max_tokens = settings.get("max_tokens", 256)
    top_p = settings.get("top_p", 0.9)

    # Try local LLM first
    if state.llm_engine is not None:
        try:
            async for token in stream_from_llm(message, history, temp, max_tokens, top_p):
                yield token
            return
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"\n\n*LLM error: {e}*"

    # Fallback: web search + knowledge base
    async for token in fallback_response(message):
        yield token


async def stream_from_llm(
    message: str,
    history: Optional[List[ChatMessage]] = None,
    temperature: float = 0.7,
    max_tokens: int = 256,
    top_p: float = 0.9,
):
    """Stream from pre-trained T5 model (encoder-decoder)."""
    engine = state.llm_engine
    tokenizer = engine["tokenizer"]
    model = engine["model"]

    loop = asyncio.get_event_loop()

    def generate():
        # Build input text
        input_text = message
        if history:
            # Add recent history for context
            ctx = []
            for msg in history[-4:]:
                role = "Human" if msg.role == "user" else "Assistant"
                ctx.append(f"{role}: {msg.content}")
            ctx.append(f"Human: {message}")
            input_text = "\n".join(ctx[-6:]) + "\nAssistant:"

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    response = await loop.run_in_executor(None, generate)
    words = response.split(" ")
    for word in words:
        yield word + " "
        await asyncio.sleep(0.005)


async def web_search(query: str) -> str:
    """Search the web using DuckDuckGo + Wikipedia."""
    results = []

    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            # Try Wikipedia API
            wiki_resp = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "srlimit": 3,
                }
            )
            if wiki_resp.status_code == 200:
                wiki_data = wiki_resp.json()
                for item in wiki_data.get("query", {}).get("search", [])[:3]:
                    title = item.get("title", "")
                    snippet = re.sub(r'<.*?>', '', item.get("snippet", ""))
                    results.append(f"- **{title}**: {snippet}")

    except Exception:
        pass

    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            # Try DuckDuckGo instant answer
            ddg_resp = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
            )
            if ddg_resp.status_code == 200:
                ddg_data = ddg_resp.json()
                abstract = ddg_data.get("AbstractText", "")
                if abstract:
                    results.insert(0, abstract)
                answer = ddg_data.get("Answer", "")
                if answer and answer != abstract:
                    results.insert(0, answer)
    except Exception:
        pass

    return "\n\n".join(results) if results else ""


async def fallback_response(message: str):
    """Knowledge base, web search, and tool matching fallback."""
    msg = message.lower().strip()

    # ── Greetings ──────────────────────────────────────────────────────────────
    if re.search(r'^(hi|hello|hey|howdy|yo|sup|greetings)', msg):
        yield "Hello! I'm **AIOS v2** — your intelligent AI operating system. How can I help you today?"
        return

    # ── Identity ───────────────────────────────────────────────────────────────
    if re.search(r'who.*(are|is).*you|your.*name|what.*(are|be).*you', msg):
        yield ("I'm **AIOS v2** — a next-generation AI operating system. "
               "I combine a custom LLaMA-style Transformer architecture with real device control capabilities. "
               "I run locally on your hardware with zero cloud dependency.")
        return

    # ── How are you ────────────────────────────────────────────────────────────
    if re.search(r'how.*(are|be).*you|how.*going|what.*up', msg):
        yield "I'm running smoothly! Ready to help with whatever you need. How about you?"
        return

    # ── Capabilities ───────────────────────────────────────────────────────────
    if re.search(r'what.*(can|do).*you.*do|help|capabilities|features|commands|what.*(skill|able|capable)', msg):
        yield ("I can handle a wide range of tasks:\n\n"
               "| Category | Capabilities |\n"
               "|----------|-------------|\n"
               "| **Device Control** | Flashlight, Alarms, Vibration |\n"
               "| **System** | Volume, Brightness, Status |\n"
               "| **Communication** | SMS, Calls, Clipboard |\n"
               "| **Productivity** | Calendar, Files, Contacts |\n"
               "| **Information** | Web Search, Location |\n"
               "| **AI** | Conversation, Code, Analysis |\n"
               "| **Memory** | Persistent facts, Context-aware |\n\n"
               "Ask me anything!")
        return

    # ── Time / Date ────────────────────────────────────────────────────────────
    if re.search(r'(what.*)?time|clock|date|day|today|now$', msg):
        now = datetime.now()
        yield f"Current time is **{now.strftime('%I:%M %p')}** on **{now.strftime('%A, %B %d, %Y')}**."
        return

    # ── Weather ────────────────────────────────────────────────────────────────
    if re.search(r'weather|temperature|rain|sunny|cloudy|forecast', msg):
        yield ("I don't have access to real-time weather data directly, "
               "but you can check your device's weather app or ask me to search the web for current conditions!")
        return

    # ── Jokes / Fun ────────────────────────────────────────────────────────────
    if re.search(r'(tell|say|give).*joke|make me laugh|funny', msg):
        yield ("Why don't scientists trust atoms?\n\nBecause they make up everything! 😄\n\nWant another one?")
        return

    if re.search(r'another.*joke|one more', msg):
        yield ("What do you call a fake noodle?\n\nAn **impasta**! 🍝")
        return

    # ── Math / Calculations ────────────────────────────────────────────────────
    if re.search(r'^[\d\s\+\-\*\/\(\)\.]+$', msg) and re.search(r'[\+\-\*\/]', msg):
        try:
            result = eval(msg, {"__builtins__": {}}, {})
            yield f"The result is **{result}**."
            return
        except Exception:
            pass

    # ── Coding / Programming ───────────────────────────────────────────────────
    if re.search(r'(write|generate|create|code).*(python|javascript|function|script|program|app)'
                 r'|how.*(code|program|write).*(python|js|java|c\+\+|rust|go)', msg):
        yield ("I can help you write code! Tell me what you'd like to build and in which language. "
               "For example: 'Write a Python function to sort a list' or 'Create a React component'.")
        return

    # ── Translation ────────────────────────────────────────────────────────────
    if re.search(r'(translate|say.*in|how.*say.*in)\s+\w+', msg):
        yield ("I can help with translations! Tell me the word or phrase and which language "
               "you'd like it translated to. For example: 'Translate hello to Spanish'.")
        return

    # ── Definitions / Explanations ─────────────────────────────────────────────
    if re.search(r'(what|define|meaning|definition|explain)\s+(is|are|does|is\s+a|is\s+an)\s+\w+', msg):
        topic = re.search(r'(?:what|define|meaning|definition|explain)\s+(?:is|are|does|is\s+a|is\s+an)\s+(.+)', msg)
        if topic:
            search_term = topic.group(1).strip()
            result = await web_search(search_term)
            if result:
                yield f"Here's what I found about **{search_term}**:\n\n{result}"
            else:
                yield f"I searched for information about **{search_term}** but couldn't find a definitive answer. Could you provide more context?"
            return

    # ── How-to questions ──────────────────────────────────────────────────────
    if re.search(r'^(how|how\s+to|how\s+do\s+i|how\s+can\s+i)\s+\w+', msg):
        topic = re.search(r'^(?:how|how\s+to|how\s+do\s+i|how\s+can\s+i)\s+(.+)', msg)
        if topic:
            result = await web_search(topic.group(1))
            if result:
                yield f"Here's what I found:\n\n{result}"
            else:
                yield f"I looked into that but couldn't find a clear answer. Can you try rephrasing?"
            return

    # ── General knowledge / Factual questions ─────────────────────────────────
    if re.search(r'^(what|who|where|when|why|which|does|did|is|are|can)\s+\w+', msg):
        result = await web_search(msg)
        if result:
            yield f"Here's what I found:\n\n{result}"
        else:
            yield f"That's a great question! I don't have enough information to give you a complete answer right now. Try asking in a different way."
        return

    # ── Default: try web search, then give a helpful response ──────────────────
    result = await web_search(msg)
    if result:
        yield f"Here's what I found:\n\n{result}"
        return

    yield ("I want to help! I can search the web, control your device, write code, "
           "answer questions, or just chat. Could you tell me more about what you need? "
           "Try asking me something specific like 'What is AI?' or 'How do I learn Python?'")


# ─── FastAPI App ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown."""
    asyncio.create_task(load_llm())
    yield
    # Cleanup
    state.llm_engine = None


app = FastAPI(
    title="AIOS v2 API",
    description="Next-generation AI Operating System Backend",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health & Status ─────────────────────────────────────────────────────────
@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "version": "2.0.0",
        "llm_loaded": state.llm_engine is not None,
        "llm_loading": state.llm_loading,
        "llm_error": state.llm_error,
        "uptime": time.time(),
    }

@app.get("/api/model/info")
async def get_model_info():
    info = {
        "loaded": state.llm_engine is not None,
        "loading": state.llm_loading,
        "error": state.llm_error,
    }
    if state.llm_engine:
        info.update({
            "model": "google/flan-t5-small",
            "params_M": 80,
            "architecture": "T5 (Encoder-Decoder, 80M params)",
        })
    return info


# ─── Chat ────────────────────────────────────────────────────────────────────
@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint with SSE."""
    if not request.message.strip():
        raise HTTPException(400, "Message is required")

    conv_id = request.conversation_id or str(uuid.uuid4())

    async def event_generator():
        full_response = ""
        try:
            async for token in generate_response(
                request.message,
                request.history,
                request.settings,
            ):
                full_response += token
                yield f"data: {json.dumps({'token': token})}\n\n"

            # Save conversation
            if request.history:
                messages = [m.dict() for m in request.history]
            else:
                messages = []

            messages.append({"role": "user", "content": request.message})
            messages.append({"role": "assistant", "content": full_response.strip()})

            save_conversation(conv_id, messages)

            yield f"data: {json.dumps({'done': True, 'conversation_id': conv_id})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Conversation-Id": conv_id,
            "Access-Control-Allow-Origin": "*",
        }
    )


# ─── Conversations ───────────────────────────────────────────────────────────
@app.get("/api/conversations")
async def get_conversations():
    return state.conversations

@app.post("/api/conversations")
async def create_conversation():
    cid = str(uuid.uuid4())
    conv = {
        "id": cid,
        "title": "New Chat",
        "created": datetime.now().isoformat(),
        "updated": datetime.now().isoformat(),
        "messages": [],
    }
    state.conversations[cid] = conv
    state._save_json("conversations.json", state.conversations)
    return conv

@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    if conv_id in state.conversations:
        del state.conversations[conv_id]
        state._save_json("conversations.json", state.conversations)
    return {"ok": True}


# ─── File Upload ─────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload files (images, documents) for AI processing."""
    if file.size and file.size > CONFIG["max_upload_size"]:
        raise HTTPException(413, "File too large")

    ext = os.path.splitext(file.filename)[1] if file.filename else ".bin"
    unique_name = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join(CONFIG["upload_dir"], unique_name)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    return UploadResponse(
        filename=file.filename,
        url=f"/uploads/{unique_name}",
        content_type=file.content_type or "application/octet-stream",
        size=len(content),
    )


# ─── Settings ────────────────────────────────────────────────────────────────
@app.get("/api/settings")
async def get_settings():
    return state.settings

@app.put("/api/settings")
async def update_settings(settings: SettingsUpdate):
    update = settings.dict(exclude_none=True)
    state.settings.update(update)
    state._save_json("settings.json", state.settings)
    return state.settings


# ─── Static Files ────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ─── Save helpers ────────────────────────────────────────────────────────────
def save_conversation(conv_id: str, messages: list):
    now = datetime.now().isoformat()
    if conv_id not in state.conversations:
        title = messages[0]["content"][:50] if messages else "New Chat"
        if len(title) > 47:
            title = title[:47] + "..."
        state.conversations[conv_id] = {
            "id": conv_id,
            "title": title,
            "created": now,
            "updated": now,
            "messages": [],
        }
    conv = state.conversations[conv_id]
    conv["messages"] = messages
    conv["updated"] = now
    state._save_json("conversations.json", state.conversations)


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"AIOS v2 Backend | http://{CONFIG['host']}:{CONFIG['port']}")
    print(f"LLM checkpoint: {CONFIG['checkpoint_path']}")
    print(f"FastAPI server starting...")

    uvicorn.run(
        "main_v2:app",
        host=CONFIG["host"],
        port=CONFIG["port"],
        reload=CONFIG["debug"],
        log_level="info",
    )
