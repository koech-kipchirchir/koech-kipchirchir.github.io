import React, { useState, useEffect, useRef, useCallback } from 'react';
import hljs from 'highlight.js';
import 'highlight.js/styles/github-dark.min.css';
import './App.css';

const API = 'http://localhost:8000';

/* ─── Markdown Renderer with Syntax Highlighting ────────────────────────────── */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(text));
  return div.innerHTML;
}

function renderMarkdown(text) {
  if (!text) return '';
  let html = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  html = html.replace(/^### (.*?)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.*?)$/gm,  '<h2>$1</h2>');
  html = html.replace(/^# (.*?)$/gm,   '<h1>$1</h1>');

  html = html.replace(/```(\w*)\n([\s\S]*?)\n```/g, (_, lang, code) => {
    const safeCode = code.replace(/`/g,'\\`').replace(/\$/g,'\\$');
    let highlighted = code;
    try {
      if (lang && hljs.getLanguage(lang)) {
        highlighted = hljs.highlight(code, { language: lang }).value;
      } else {
        highlighted = hljs.highlightAuto(code).value;
      }
    } catch {}
    return `<div class="code-block"><div class="code-header"><span>${lang || 'code'}</span>` +
      `<button class="btn-copy-code" data-code="${safeCode}">Copy</button></div>` +
      `<pre><code class="hljs language-${lang || ''}">${highlighted}</code></pre></div>`;
  });

  html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>');

  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/__(.*?)__/g,      '<strong>$1</strong>');
  html = html.replace(/\*(.*?)\*/g,      '<em>$1</em>');
  html = html.replace(/_(.*?)_/g,        '<em>$1</em>');
  html = html.replace(/^\s*-\s+(.*?)$/gm,'<li>$1</li>');
  html = html.replace(/(<li>.*?<\/li>)+/g, '<ul>$&</ul>');
  html = html.replace(/<\/ul>\s*<ul>/g,  '');
  html = html.replace(/\n/g, '<br/>');

  // Tables
  html = html.replace(/\|(.+)\|/g, (match, rowContent) => {
    const cols = rowContent.split('|').map(c => c.trim());
    return `<tr>${cols.map(c => `<td>${c}</td>`).join('')}</tr>`;
  });
  html = html.replace(/(<tr>[\s\S]*?<\/tr>)+/g, m =>
    `<div class="table-container"><table><tbody>${m}</tbody></table></div>`
  );

  return html;
}

// Copy handler
document.addEventListener('click', (e) => {
  const btn = e.target.closest('.btn-copy-code');
  if (btn) {
    const code = btn.dataset.code;
    navigator.clipboard.writeText(code || '');
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
  }
});

/* ─── Thinking Dots ─────────────────────────────────────────────────────────── */
function ThinkingDots() {
  return (
    <div className="thinking-row animate-fade-in">
      <div className="msg-avatar model-avatar">🤖</div>
      <div className="thinking-bubble">
        <span className="dot" /><span className="dot" /><span className="dot" />
      </div>
    </div>
  );
}

/* ─── Message Bubble ────────────────────────────────────────────────────────── */
function MessageBubble({ msg, isStreaming, onRegenerate }) {
  const isUser = msg.role === 'user';
  const content = msg.content || '';
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const timeLabel = msg.ts
    ? new Date(msg.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : null;

  return (
    <div className={`msg-row ${isUser ? 'user-row' : 'model-row'} animate-fade-in`}>
      <div className={`msg-avatar ${isUser ? 'user-avatar' : 'model-avatar'}`}>
        {isUser ? '👤' : '🤖'}
      </div>
      <div className={`msg-bubble ${isUser ? 'user-bubble' : 'model-bubble'}`}>
        <div
          className="msg-content"
          dangerouslySetInnerHTML={{ __html: renderMarkdown(content) }}
        />
        {isStreaming && !isUser && <span className="cursor-blink">▋</span>}
        <div className="msg-footer">
          {timeLabel && <span className="msg-time">{timeLabel}</span>}
          <div className="msg-actions">
            <button className="bubble-action-btn" onClick={handleCopy} title="Copy">
              {copied ? '✅' : '📋'}
            </button>
            {!isUser && !isStreaming && onRegenerate && (
              <button className="bubble-action-btn" onClick={onRegenerate} title="Regenerate">🔄</button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ─── Hero Welcome ──────────────────────────────────────────────────────────── */
function HeroWelcome({ onPrompt }) {
  const STARTERS = [
    { icon: '💬', label: 'Who are you?', desc: 'Learn about AIOS v2' },
    { icon: '💡', label: 'What can you do?', desc: 'See all capabilities' },
    { icon: '🔦', label: 'Turn on flashlight', desc: 'Device control' },
    { icon: '⏰', label: 'Set alarm 7:30 AM', desc: 'System tools' },
    { icon: '📊', label: 'Device status', desc: 'Check battery' },
    { icon: '📳', label: 'Vibrate phone', desc: 'Haptic feedback' },
    { icon: '💻', label: 'Write Python code', desc: 'Code generation' },
    { icon: '🌐', label: 'Search the web', desc: 'Web search' },
    { icon: '🧠', label: 'Explain AI concepts', desc: 'Deep knowledge' },
  ];
  return (
    <div className="hero-welcome animate-fade-in">
      <div className="hero-logo">🧠</div>
      <h2 className="hero-title">How can I help you today?</h2>
      <p className="hero-sub">Choose a suggestion or type your own message</p>
      <div className="hero-grid">
        {STARTERS.map(s => (
          <button key={s.label} className="hero-card" onClick={() => onPrompt(s.label)}>
            <span className="hero-card-icon">{s.icon}</span>
            <div className="hero-card-text">
              <span className="hero-card-label">{s.label}</span>
              <span className="hero-card-desc">{s.desc}</span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

/* ─── Model Badge ───────────────────────────────────────────────────────────── */
function ModelStatusBadge({ info }) {
  if (!info) return null;
  if (info.loading) return <span className="badge badge-loading">⏳ Loading…</span>;
  if (info.loaded)  return <span className="badge badge-online">✅ {info.params_M}M LLM</span>;
  return <span className="badge badge-offline">⚡ Tool Mode</span>;
}

/* ─── Settings Panel ────────────────────────────────────────────────────────── */
function SettingsPanel({ settings, onSave }) {
  const [local, setLocal] = useState(settings);
  const set = (k, v) => setLocal(p => ({ ...p, [k]: v }));

  return (
    <div className="panel animate-fade-in">
      <h2 className="panel-title">⚙️ Settings</h2>
      <p className="panel-sub">Configure AIOS behaviour and model inference</p>

      <div className="settings-list">
        <div className="setting-row">
          <div>
            <div className="setting-label">Response Temperature</div>
            <div className="setting-desc">Higher = more creative, lower = more focused</div>
          </div>
          <div className="setting-control">
            <input type="range" min="0.1" max="1.5" step="0.05"
              value={local.temperature}
              onChange={e => set('temperature', parseFloat(e.target.value))}
            />
            <span className="setting-val">{local.temperature.toFixed(2)}</span>
          </div>
        </div>

        <div className="setting-row">
          <div>
            <div className="setting-label">Max Tokens</div>
            <div className="setting-desc">Maximum number of tokens to generate per reply</div>
          </div>
          <div className="setting-control">
            <input type="range" min="50" max="500" step="10"
              value={local.max_tokens}
              onChange={e => set('max_tokens', parseInt(e.target.value))}
            />
            <span className="setting-val">{local.max_tokens}</span>
          </div>
        </div>

        <div className="setting-row">
          <div>
            <div className="setting-label">Top-K Sampling</div>
            <div className="setting-desc">Limits word selection to the top K candidates</div>
          </div>
          <div className="setting-control">
            <input type="range" min="5" max="100" step="5"
              value={local.top_k}
              onChange={e => set('top_k', parseInt(e.target.value))}
            />
            <span className="setting-val">{local.top_k}</span>
          </div>
        </div>

        <div className="setting-row">
          <div>
            <div className="setting-label">Voice Speed</div>
            <div className="setting-desc">Rate of speech synthesis output</div>
          </div>
          <div className="setting-control">
            <input type="range" min="0.5" max="2.0" step="0.1"
              value={local.voice_rate}
              onChange={e => set('voice_rate', parseFloat(e.target.value))}
            />
            <span className="setting-val">{local.voice_rate.toFixed(1)}×</span>
          </div>
        </div>

        <div className="setting-row">
          <div>
            <div className="setting-label">Sidebar Auto-collapse</div>
            <div className="setting-desc">Collapse sidebar when sending a message</div>
          </div>
          <div className="setting-control">
            <label className="toggle-switch">
              <input type="checkbox"
                checked={local.sidebar_auto_collapse}
                onChange={e => set('sidebar_auto_collapse', e.target.checked)}
              />
              <span className="toggle-slider" />
            </label>
          </div>
        </div>

        <div className="setting-row">
          <div>
            <div className="setting-label">Show Timestamps</div>
            <div className="setting-desc">Display time on each message bubble</div>
          </div>
          <div className="setting-control">
            <label className="toggle-switch">
              <input type="checkbox"
                checked={local.show_timestamps}
                onChange={e => set('show_timestamps', e.target.checked)}
              />
              <span className="toggle-slider" />
            </label>
          </div>
        </div>
      </div>

      <button className="btn-save-settings" onClick={() => onSave(local)}>
        💾 Save Settings
      </button>
    </div>
  );
}

/* ─── Image Upload Component ──────────────────────────────────────────────────── */
function ImagePreview({ file, onRemove }) {
  const url = URL.createObjectURL(file);
  useEffect(() => () => URL.revokeObjectURL(url), [url]);
  return (
    <div className="image-preview">
      <img src={url} alt="preview" />
      <button className="remove-image" onClick={onRemove}>✕</button>
    </div>
  );
}

/* ─── App ───────────────────────────────────────────────────────────────────── */
const DEFAULT_SETTINGS = {
  temperature: 0.7,
  max_tokens: 512,
  top_k: 40,
  top_p: 0.9,
  repetition_penalty: 1.1,
  voice_rate: 1.0,
  sidebar_auto_collapse: false,
  show_timestamps: true,
  model: 'local',
  provider: 'local',
};

export default function App() {
  const [conversations, setConversations] = useState({});
  const [currentId, setCurrentId]         = useState(null);
  const [messages, setMessages]           = useState([]);
  const [input, setInput]                 = useState('');
  const [streaming, setStreaming]         = useState(false);
  const [thinking, setThinking]           = useState(false);
  const [activeTab, setActiveTab]         = useState('chat');
  const [modelInfo, setModelInfo]         = useState(null);
  const [memories, setMemories]           = useState([]);
  const [newMem, setNewMem]               = useState({ key: '', value: '' });
  const [logs, setLogs]                   = useState([]);
  const [status, setStatus]               = useState({ battery_level:'—', charging:false, media_volume:'—', brightness:128, system_time:'—' });
  const [searchQuery, setSearchQuery]     = useState('');
  const [voiceOutput, setVoiceOutput]     = useState(false);
  const [isListening, setIsListening]     = useState(false);
  const [theme, setTheme]                 = useState('dark');
  const [sidebarOpen, setSidebarOpen]     = useState(true);
  const [settings, setSettings]           = useState(DEFAULT_SETTINGS);
  const [settingsSaved, setSettingsSaved] = useState(false);
  const [attachedFiles, setAttachedFiles]  = useState([]);
  const [showModelSelector, setShowModelSelector] = useState(false);
  const [uploading, setUploading]         = useState(false);

  const bottomRef      = useRef(null);
  const inputRef       = useRef(null);
  const abortRef       = useRef(null);
  const recognitionRef = useRef(null);
  const fileInputRef   = useRef(null);

  /* Theme */
  useEffect(() => { document.documentElement.setAttribute('data-theme', theme); }, [theme]);

  /* Speech recognition */
  useEffect(() => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return;
    const rec = new SR();
    rec.continuous = false;
    rec.interimResults = false;
    rec.lang = 'en-US';
    rec.onstart  = () => setIsListening(true);
    rec.onend    = () => setIsListening(false);
    rec.onerror  = () => setIsListening(false);
    rec.onresult = e => setInput(p => p + (p ? ' ' : '') + e.results[0][0].transcript);
    recognitionRef.current = rec;
  }, []);

  const speakText = useCallback((text) => {
    if (!voiceOutput) return;
    const clean = text.replace(/<[^>]*>/g, '').replace(/[*#`_\-|]/g, '').trim();
    if (!clean) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(clean);
    u.rate  = settings.voice_rate;
    u.pitch = 1.0;
    window.speechSynthesis.speak(u);
  }, [voiceOutput, settings.voice_rate]);

  useEffect(() => { if (!voiceOutput) window.speechSynthesis.cancel(); }, [voiceOutput]);

  /* Scroll to bottom */
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, thinking]);

  /* Load settings from backend */
  useEffect(() => {
    fetch(`${API}/api/settings`)
      .then(r => r.json())
      .then(s => setSettings({ ...DEFAULT_SETTINGS, ...s }))
      .catch(() => {});
  }, []);

  /* File upload handler */
  const handleFileSelect = async (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;
    setUploading(true);
    const uploaded = [];
    for (const file of files) {
      if (file.size > 10 * 1024 * 1024) continue;
      const formData = new FormData();
      formData.append('file', file);
      try {
        const res = await fetch(`${API}/api/upload`, { method: 'POST', body: formData });
        const data = await res.json();
        uploaded.push({ file: data, name: file.name, type: file.type });
      } catch (err) {
        console.error('Upload failed:', err);
      }
    }
    setAttachedFiles(prev => [...prev, ...uploaded]);
    setUploading(false);
    e.target.value = '';
  };

  const removeAttachedFile = (idx) => {
    setAttachedFiles(prev => prev.filter((_, i) => i !== idx));
  };

  /* Save settings */
  const saveSettings = async (s) => {
    setSettings(s);
    try {
      await fetch(`${API}/api/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(s),
      });
    } catch {}
    setSettingsSaved(true);
    setTimeout(() => setSettingsSaved(false), 2000);
  };

  /* Fetch everything */
  const fetchAll = useCallback(async () => {
    try {
      const [s, m, l, mi, c] = await Promise.all([
        fetch(`${API}/api/status`).then(r => r.json()),
        fetch(`${API}/api/memory`).then(r => r.json()),
        fetch(`${API}/api/logs`).then(r => r.json()),
        fetch(`${API}/api/model/info`).then(r => r.json()),
        fetch(`${API}/api/conversations`).then(r => r.json()),
      ]);
      setStatus(s); setMemories(m); setLogs(l.slice().reverse());
      setModelInfo(mi); setConversations(c);
      if (!currentId && Object.keys(c).length > 0) {
        const ids = Object.keys(c).sort((a, b) => new Date(c[b].updated) - new Date(c[a].updated));
        setCurrentId(ids[0]);
        setMessages(c[ids[0]].messages || []);
      }
    } catch {}
  }, [currentId]);

  useEffect(() => {
    fetchAll();
    const id = setInterval(fetchAll, 5000);
    return () => clearInterval(id);
  }, [fetchAll]);

  /* Conversations */
  const selectConversation = (id) => {
    setCurrentId(id);
    setMessages(conversations[id]?.messages || []);
    setActiveTab('chat');
    if (settings.sidebar_auto_collapse) setSidebarOpen(false);
  };

  const createNewChat = async () => {
    try {
      const res  = await fetch(`${API}/api/conversations/new`, { method: 'POST' });
      const data = await res.json();
      setCurrentId(data.id);
      setMessages([]);
      setActiveTab('chat');
      fetchAll();
    } catch {}
  };

  const deleteChat = async (e, id) => {
    e.stopPropagation();
    try {
      await fetch(`${API}/api/conversations/${id}`, { method: 'DELETE' });
      if (currentId === id) { setCurrentId(null); setMessages([]); }
      fetchAll();
    } catch {}
  };

  /* Voice */
  const toggleListening = () => {
    if (!recognitionRef.current) { alert('Speech recognition not supported. Use Chrome or Edge.'); return; }
    isListening ? recognitionRef.current.stop() : recognitionRef.current.start();
  };

  /* Export / Import */
  const exportHistory = () => {
    const a = document.createElement('a');
    a.href = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(conversations, null, 2));
    a.download = `aios_history_${new Date().toISOString().slice(0,10)}.json`;
    a.click();
  };
  const importHistory = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const fr = new FileReader();
    fr.onload = async (ev) => {
      try {
        JSON.parse(ev.target.result); // validate
        alert('Import successful! (Requires backend persistence support)');
        fetchAll();
      } catch { alert('Invalid JSON file.'); }
    };
    fr.readAsText(file);
  };

  /* Send message */
  const handleSend = async (e, customText = null) => {
    e?.preventDefault();
    const text = (customText ?? input).trim();
    if ((!text && attachedFiles.length === 0) || streaming) return;
    if (!customText) setInput('');
    if (settings.sidebar_auto_collapse) setSidebarOpen(false);

    // Build user message with attachments
    const now = new Date().toISOString();
    const history = [...messages];
    let userContent = text;
    if (attachedFiles.length > 0) {
      const fileRefs = attachedFiles.map(f => `![${f.name}](${f.file.url})`).join('\n');
      userContent = fileRefs + (text ? '\n\n' + text : '');
    }
    const userMsg = { role: 'user', content: userContent, ts: now, files: attachedFiles.map(f => f.file) };

    setMessages(prev => [...prev, userMsg]);
    setAttachedFiles([]);
    setThinking(true);

    const controller = new AbortController();
    abortRef.current = controller;
    let fullOutput = '';

    try {
      const res = await fetch(`${API}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userContent,
          conversation_id: currentId || undefined,
          history: history.filter(m => m.content),
          settings: {
            temperature: settings.temperature,
            max_tokens: settings.max_tokens,
            top_k: settings.top_k,
            top_p: settings.top_p,
            repetition_penalty: settings.repetition_penalty,
          },
        }),
        signal: controller.signal,
      });

      const convId = res.headers.get('X-Conversation-Id');
      if (convId && !currentId) setCurrentId(convId);

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let firstToken = true;

      let shouldBreak = false;
      while (!shouldBreak) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (raw === '[DONE]') { shouldBreak = true; break; }
          try {
            const data = JSON.parse(raw);
            if (data.token) {
              if (firstToken) {
                setThinking(false);
                setStreaming(true);
                setMessages(prev => [...prev, { role: 'model', content: '', ts: new Date().toISOString() }]);
                firstToken = false;
              }
              fullOutput += data.token;
              setMessages(prev => {
                const msgs = [...prev];
                const last = msgs[msgs.length - 1];
                if (last?.role === 'model') msgs[msgs.length - 1] = { ...last, content: last.content + data.token };
                return msgs;
              });
            }
            if (data.conversation_id && !currentId) setCurrentId(data.conversation_id);
          } catch {}
        }
      }
      speakText(fullOutput);
    } catch (err) {
      setThinking(false);
      if (err.name !== 'AbortError') {
        setMessages(prev => [...prev, { role: 'model', content: `⚠️ **Error:** ${err.message}`, ts: new Date().toISOString() }]);
      }
    } finally {
      setThinking(false);
      setStreaming(false);
      fetchAll();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  const stopStream = () => { abortRef.current?.abort(); setStreaming(false); setThinking(false); };

  const handleRegenerate = () => {
    if (messages.length < 2 || streaming) return;
    let idx = -1;
    for (let i = messages.length - 1; i >= 0; i--) { if (messages[i].role === 'user') { idx = i; break; } }
    if (idx === -1) return;
    const text = messages[idx].content;
    setMessages(messages.slice(0, idx));
    handleSend(null, text);
  };

  const addMemory = async (e) => {
    e.preventDefault();
    if (!newMem.key || !newMem.value) return;
    await fetch(`${API}/api/memory`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(newMem) });
    setNewMem({ key: '', value: '' });
    fetchAll();
  };
  const deleteMemory = async (key) => { await fetch(`${API}/api/memory/${key}`, { method: 'DELETE' }); fetchAll(); };

  const filteredConversations = Object.keys(conversations)
    .filter(id => conversations[id].title.toLowerCase().includes(searchQuery.toLowerCase()))
    .sort((a, b) => new Date(conversations[b].updated) - new Date(conversations[a].updated));

  const isEmptyChat = messages.length === 0;
  const wordCount   = input.trim() ? input.trim().split(/\s+/).length : 0;

  return (
    <div className={`app ${sidebarOpen ? '' : 'sidebar-collapsed'}`}>

      {/* ── Sidebar Toggle ── */}
      <button className="sidebar-toggle-btn" onClick={() => setSidebarOpen(o => !o)} title="Toggle Sidebar">
        {sidebarOpen ? '‹' : '›'}
      </button>

      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <span className="logo-icon">🧠</span>
          <div>
            <div className="logo-title">AIOS Hub</div>
            <div className="logo-sub">On-Device Control</div>
          </div>
        </div>

        <div className="sidebar-controls">
          <button className="btn-new-chat" onClick={createNewChat}><span>+</span> New Chat</button>
          <button className="theme-toggle-btn" onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}>
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>
        </div>

        <div className="backup-controls">
          <button className="backup-btn" onClick={exportHistory}>📤 Export</button>
          <label className="backup-btn import-label">
            📥 Import
            <input type="file" accept=".json" onChange={importHistory} style={{ display: 'none' }} />
          </label>
        </div>

        <nav className="sidebar-nav">
          {[
            { id:'chat',    icon:'💬', label:'Agent Chat' },
            { id:'memory',  icon:'🧬', label:'Memory Store' },
            { id:'logs',    icon:'📋', label:'Tool Logs' },
            { id:'model',   icon:'🖥️', label:'System & Model' },
            { id:'settings',icon:'⚙️', label:'Settings' },
          ].map(t => (
            <button key={t.id} className={`nav-item ${activeTab === t.id ? 'active' : ''}`} onClick={() => setActiveTab(t.id)}>
              <span className="nav-icon">{t.icon}</span>
              <span>{t.label}</span>
            </button>
          ))}
        </nav>

        <div className="history-section">
          <div className="history-header">Recent Chats</div>
          <div className="search-container">
            <input className="search-input" placeholder="Search chats…" value={searchQuery} onChange={e => setSearchQuery(e.target.value)} />
            {searchQuery && <button className="clear-search" onClick={() => setSearchQuery('')}>✕</button>}
          </div>
          <div className="history-list">
            {filteredConversations.length === 0
              ? <div className="history-empty">{searchQuery ? 'No matches' : 'No chats yet'}</div>
              : filteredConversations.map(id => (
                <div key={id}
                  className={`history-item ${currentId === id && activeTab === 'chat' ? 'active' : ''}`}
                  onClick={() => selectConversation(id)}
                >
                  <span className="history-icon">💬</span>
                  <span className="history-title" title={conversations[id].title}>{conversations[id].title}</span>
                  <button className="btn-delete-chat" onClick={e => deleteChat(e, id)}>✕</button>
                </div>
              ))
            }
          </div>
        </div>

        <div className="sidebar-status">
          <ModelStatusBadge info={modelInfo} />
          <div className="status-grid">
            <div className="stat"><span>🔋</span><span>{status.battery_level}</span></div>
            <div className="stat"><span>🔊</span><span>{status.media_volume}</span></div>
            <div className="stat"><span>☀️</span><span>{status.brightness}/255</span></div>
            <div className="stat"><span>🕐</span><span>{status.system_time?.slice(11,16)}</span></div>
          </div>
        </div>
      </aside>

      {/* ── Main ── */}
      <main className="main">

        {activeTab === 'chat' && (
          <div className="chat-panel">
            <div className="chat-header">
              <div>
                <h1 className="chat-title">AIOS v2</h1>
                <p className="chat-subtitle">
                  {modelInfo?.loaded ? `🧠 ${modelInfo.params_M}M LLM · ` : '⚡ Tool Mode · '}
                  GQA · RoPE · SwiGLU · {modelInfo?.max_context || 2048}ctx
                </p>
              </div>
              <div className="header-actions">
                {/* Model Selector */}
                <div className="model-selector">
                  <button className="model-selector-btn" onClick={() => setShowModelSelector(!showModelSelector)}>
                    🧠 {settings.model === 'local' ? 'Local LLM' : settings.model}
                  </button>
                  {showModelSelector && (
                    <div className="model-dropdown">
                      {['local', 'gemini-2.5-flash', 'gemini-2.5-pro', 'gpt-4o'].map(m => (
                        <button key={m} className="model-option"
                          onClick={() => { setSettings(s => ({ ...s, model: m })); setShowModelSelector(false); }}>
                          {m === 'local' ? '🧠 Local LLM v2' : m === 'gemini-2.5-flash' ? '⚡ Gemini 2.5 Flash' : m === 'gemini-2.5-pro' ? '🧪 Gemini 2.5 Pro' : '✨ GPT-4o'}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
                <button className={`btn-voice-toggle ${voiceOutput ? 'active' : ''}`}
                  onClick={() => setVoiceOutput(v => !v)}>
                  {voiceOutput ? '🔊' : '🔇'}
                </button>
                <button className="btn-clear"
                  onClick={() => { setMessages([]); setCurrentId(null); }}>
                  Clear
                </button>
              </div>
            </div>

            <div className="messages-area">
              {isEmptyChat && !thinking
                ? <HeroWelcome onPrompt={t => { setInput(t); inputRef.current?.focus(); }} />
                : messages.map((msg, i) => (
                  <div key={i}>
                    {/* Render attached images */}
                    {msg.files && msg.files.length > 0 && (
                      <div className="attachments-row">
                        {msg.files.map((f, fi) => (
                          f.content_type?.startsWith('image/') ? (
                            <img key={fi} src={`${API}${f.url}`} alt={f.filename}
                              className="attached-image" />
                          ) : (
                            <div key={fi} className="attached-file">
                              📎 {f.filename}
                            </div>
                          )
                        ))}
                      </div>
                    )}
                    <MessageBubble
                      msg={{ ...msg, ts: settings.show_timestamps ? msg.ts : null }}
                      isStreaming={streaming && i === messages.length - 1 && msg.role === 'model'}
                      onRegenerate={i === messages.length - 1 ? handleRegenerate : null}
                    />
                  </div>
                ))
              }
              {thinking && <ThinkingDots />}
              <div ref={bottomRef} />
            </div>

            <form className="input-bar" onSubmit={handleSend}>
              <div className="input-wrap">
                {/* Attached files preview */}
                {attachedFiles.length > 0 && (
                  <div className="attached-preview-row">
                    {attachedFiles.map((f, i) => (
                      f.type?.startsWith('image/') ? (
                        <ImagePreview key={i} file={f.file ? new File([], f.name) : null} onRemove={() => removeAttachedFile(i)} />
                      ) : (
                        <div key={i} className="file-chip">
                          📎 {f.name}
                          <button onClick={() => removeAttachedFile(i)}>✕</button>
                        </div>
                      )
                    ))}
                  </div>
                )}
                <textarea
                  ref={inputRef}
                  className="chat-input"
                  placeholder={attachedFiles.length > 0 ? "Add a message or ask about the files..." : "Message AIOS v2... (Enter to send, Shift+Enter for newline)"}
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  rows={1}
                  disabled={streaming || thinking}
                />
                {input && (
                  <span className="word-counter">{wordCount} word{wordCount !== 1 ? 's' : ''}</span>
                )}
              </div>
              {/* File upload */}
              <input type="file" ref={fileInputRef} style={{display:'none'}} multiple
                accept="image/*,.pdf,.txt,.doc,.docx,.csv,.json,.py,.js,.ts,.jsx,.tsx,.html,.css"
                onChange={handleFileSelect} />
              <button type="button" className="btn-attach" onClick={() => fileInputRef.current?.click()}
                disabled={streaming || thinking} title="Attach files">
                📎
              </button>
              <button type="button" className={`btn-mic ${isListening ? 'listening' : ''}`}
                onClick={toggleListening} disabled={streaming || thinking} title="Voice input">
                🎙️
              </button>
              {(streaming || thinking) ? (
                <button type="button" className="btn-stop" onClick={stopStream}>⏹ Stop</button>
              ) : (
                <button type="submit" className="btn-send" disabled={!input.trim() && attachedFiles.length === 0}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
                  </svg>
                </button>
              )}
            </form>
          </div>
        )}

        {activeTab === 'memory' && (
          <div className="panel animate-fade-in">
            <h2 className="panel-title">🧬 Memory Store</h2>
            <p className="panel-sub">Persistent facts AIOS remembers across all sessions</p>
            <form className="add-form" onSubmit={addMemory}>
              <input className="form-input" placeholder="Key (e.g. user_name)" value={newMem.key} onChange={e => setNewMem(p => ({...p, key: e.target.value}))} />
              <input className="form-input" placeholder="Value (e.g. John)" value={newMem.value} onChange={e => setNewMem(p => ({...p, value: e.target.value}))} />
              <button className="btn-add" type="submit">+ Add</button>
            </form>
            <div className="mem-list">
              {memories.length === 0 && <p className="empty-state">No memories stored yet.</p>}
              {memories.map(m => (
                <div key={m.key} className="mem-item">
                  <div><span className="mem-key">{m.key}</span><span className="mem-val">{m.value}</span></div>
                  <button className="btn-del" onClick={() => deleteMemory(m.key)}>✕</button>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'logs' && (
          <div className="panel animate-fade-in">
            <h2 className="panel-title">📋 Tool Execution Logs</h2>
            <p className="panel-sub">Every system tool AIOS executed, in real-time</p>
            <div className="log-list">
              {logs.length === 0 && <p className="empty-state">No tool calls yet. Try a device command!</p>}
              {logs.map((log, i) => (
                <div key={i} className="log-item">
                  <span className="log-time">{log.timestamp?.slice(11,19)}</span>
                  <span className={`log-status ${log.status === 'success' ? 'ok' : 'err'}`}>{log.status}</span>
                  <span className="log-tool">{log.tool}</span>
                  <span className="log-result">{log.result}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'model' && (
          <div className="panel animate-fade-in">
            <h2 className="panel-title">🖥️ System & Model</h2>
            <p className="panel-sub">Hardware, runtime, and neural network details</p>
            {modelInfo ? (
              <div className="model-cards">
                {[
                  { icon:'🏗️', label:'Architecture',   val:'LLaMA Transformer',      sub:'RoPE · SwiGLU · RMSNorm' },
                  { icon:'📊', label:'Parameters',      val:`${modelInfo.params_M ?? '—'}M`, sub:'100% local, no cloud' },
                  { icon:'🧱', label:'Layers / Heads',  val:`${modelInfo.layers ?? '—'} / ${modelInfo.heads ?? '—'}`, sub:`Embed: ${modelInfo.embed_dim ?? '—'}` },
                  { icon:'📚', label:'Vocabulary',      val:`${modelInfo.vocab_size ?? '—'}`, sub:'Character tokenizer' },
                  { icon:modelInfo.loaded ? '✅' : '⚠️', label:'Status', val:modelInfo.loaded ? 'Online' : 'Tool Mode', sub:modelInfo.error || 'Inference ready' },
                  { icon:'💾', label:'Checkpoint',      val:'aios_llm.pth',           sub:modelInfo.exists ? '✅ Found' : '❌ Missing' },
                  { icon:'💻', label:'Host OS',         val:'Windows 11 x64',         sub:'Python 3.14 · Port 8000' },
                  { icon:'⚡', label:'Vite Server',      val:'Dev Mode',               sub:'Hot Module Replacement' },
                ].map(c => (
                  <div key={c.label} className="model-card">
                    <div className="card-icon">{c.icon}</div>
                    <div className="card-label">{c.label}</div>
                    <div className="card-val">{c.val}</div>
                    <div className="card-sub">{c.sub}</div>
                  </div>
                ))}
              </div>
            ) : <p className="empty-state">Connecting to backend…</p>}
          </div>
        )}

        {activeTab === 'settings' && (
          <SettingsPanel settings={settings} onSave={saveSettings} />
        )}

        {settingsSaved && (
          <div className="toast">✅ Settings saved!</div>
        )}
      </main>
    </div>
  );
}
