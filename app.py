"""
app.py — HR Policy RAG Chatbot
Fixed:
  1. Sources HTML rendering correctly (separate st.markdown call)
  2. Chat input full width matching body
  3. Non-functional nav tabs removed from sidebar
"""

import uuid
import streamlit as st

from agents import ask, get_welcome_message
from config import app_config, validate_all
from models import ChatHistory, MessageType, QueryRequest, Role

# ══════════════════════════════════════════════════════════════════════════════
# Page Config
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="HR Assistant",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Fraunces:ital,wght@0,300;0,400;1,300&display=swap');

:root {
    --bg:      #1C1B2E;
    --bg2:     #16152A;
    --bg3:     #211F35;
    --glass:   rgba(255,255,255,0.06);
    --border:  rgba(255,255,255,0.08);
    --border2: rgba(255,255,255,0.13);
    --accent:  #A78BFA;
    --accent2: #C4B5FD;
    --accent3: #7C3AED;
    --text:    #E2E0EE;
    --text2:   rgba(226,224,238,0.55);
    --text3:   rgba(226,224,238,0.3);
    --user-bg: #3D3562;
    --bot-bg:  rgba(255,255,255,0.055);
    --glow:    rgba(124,58,237,0.2);
}

*, *::before, *::after { box-sizing: border-box; }
#MainMenu, footer, header { visibility: hidden; }

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--text);
}

[data-testid="stMainBlockContainer"] {
    padding-top: 0rem !important;
}
/* ── Background ── */
.stApp {
    background: var(--bg);
    background-image:
        radial-gradient(ellipse 80% 60% at 65% 25%, rgba(124,58,237,0.13) 0%, transparent 65%),
        radial-gradient(ellipse 50% 40% at 15% 80%, rgba(236,72,153,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 40% 50% at 88% 70%, rgba(99,102,241,0.09) 0%, transparent 60%);
    min-height: 100vh;
}

/* ── Main content area ── */
/* ── Main content area ── */
.main .block-container {
    max-width: 860px;
    padding-top: 0.5rem !important;
    padding-right: 2rem;
    padding-left: 2rem;
    padding-bottom: 1rem;
    margin: 0 auto;
}

/* Remove default Streamlit top spacing */
section.main > div {
    padding-top: 0rem !important;
}

/* ══════════════════════════════════════
   SIDEBAR — clean, no nav tabs
══════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1rem;
}

/* Logo */
.sb-logo {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin-bottom: 1.5rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--border);
}
.sb-logo-icon {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #5B21B6, #A78BFA);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.95rem;
    box-shadow: 0 0 16px var(--glow);
}
.sb-logo-text {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.3px;
}
.sb-logo-dot { color: var(--accent); }

/* Info cards in sidebar */
.sb-card {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.75rem;
}
.sb-card-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: var(--text3);
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.sb-card-body {
    font-size: 0.8rem;
    color: var(--text2);
    line-height: 1.55;
}
.sb-card-body span {
    color: var(--accent2);
    font-weight: 500;
}

/* New Chat button */
.stButton > button {
    background: rgba(124,58,237,0.15) !important;
    border: 1px solid rgba(167,139,250,0.25) !important;
    border-radius: 10px !important;
    color: var(--accent2) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 0.6rem 1rem !important;
    white-space: normal !important;
    text-align: center !important;
    height: auto !important;
    width: 100% !important;
    transition: all 0.18s ease !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    background: rgba(124,58,237,0.28) !important;
    border-color: rgba(167,139,250,0.45) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(124,58,237,0.2) !important;
}

/* Sidebar footer */
.sb-footer {
    margin-top: 1.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    font-size: 0.65rem;
    color: var(--text3);
    text-align: center;
    letter-spacing: 0.05em;
}

/* ══════════════════════════════════════
   HEADER
══════════════════════════════════════ */
.chat-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0 0.8rem;
    margin-top: 0rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
    animation: fadeDown 0.5s ease both;
}
.chat-header-left {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--text);
}
.online-dot {
    width: 8px; height: 8px;
    background: #34D399;
    border-radius: 50%;
    box-shadow: 0 0 7px #34D399;
    animation: pulse-green 2s ease infinite;
    flex-shrink: 0;
}
.chat-header-right {
    font-size: 0.7rem;
    color: var(--text3);
    letter-spacing: 0.03em;
}

/* ══════════════════════════════════════
   MESSAGES
   FIX: render content + sources as
   TWO separate st.markdown() calls
   so HTML in src_html is never escaped
══════════════════════════════════════ */
.msg-row-user {
    display: flex;
    justify-content: flex-end;
    align-items: flex-end;
    gap: 0.65rem;
    margin: 0.75rem 0;
    animation: slideLeft 0.28s ease both;
}
.msg-row-bot {
    display: flex;
    justify-content: flex-start;
    align-items: flex-start;
    gap: 0.65rem;
    margin: 0.75rem 0;
    animation: slideRight 0.28s ease both;
}

.bubble-user {
    background: var(--user-bg);
    border: 1px solid rgba(167,139,250,0.18);
    border-radius: 18px 18px 4px 18px;
    padding: 0.85rem 1.1rem;
    max-width: 70%;
    font-size: 0.88rem;
    line-height: 1.65;
    color: var(--text);
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    word-break: break-word;
}
.bubble-bot {
    background: var(--bot-bg);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid var(--border2);
    border-radius: 4px 18px 18px 18px;
    padding: 0.9rem 1.15rem;
    max-width: 80%;
    font-size: 0.88rem;
    line-height: 1.8;
    color: var(--text);
    box-shadow: 0 4px 24px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.05);
    word-break: break-word;
}
.bubble-fallback {
    background: rgba(251,191,36,0.055);
    border: 1px solid rgba(251,191,36,0.18);
    border-radius: 4px 18px 18px 18px;
    padding: 0.9rem 1.15rem;
    max-width: 80%;
    font-size: 0.88rem;
    line-height: 1.8;
    color: #FCD34D;
    word-break: break-word;
}

/* Mini avatars */
.mini-av {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.82rem;
    flex-shrink: 0;
}
.mini-av-bot  { background: linear-gradient(135deg,#4C1D95,#7C3AED); box-shadow:0 0 10px var(--glow); }
.mini-av-user { background: linear-gradient(135deg,#3D3562,#5B52A3); }

/* ── Sources bar — rendered as its OWN st.markdown call ── */
.src-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.35rem;
    margin-top: 0.7rem;
    padding-top: 0.65rem;
    border-top: 1px solid var(--border);
}
.src-lbl {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: var(--text3);
    font-weight: 600;
}
.src-pill {
    background: rgba(167,139,250,0.1);
    border: 1px solid rgba(167,139,250,0.2);
    color: var(--accent2);
    border-radius: 20px;
    padding: 0.13rem 0.58rem;
    font-size: 0.68rem;
    font-weight: 500;
}
.src-lat {
    margin-left: auto;
    font-size: 0.65rem;
    color: var(--text3);
}

/* ══════════════════════════════════════
   WELCOME
══════════════════════════════════════ */
.welcome-wrap {
    text-align: center;
    margin-top: 0 !important;
    padding: 2.5rem 0 1.5rem;
    animation: fadeUp 0.6s ease both;
}
.welcome-av {
    width: 72px; height: 72px;
    background: linear-gradient(135deg,#4C1D95,#7C3AED,#A78BFA);
    border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 1.9rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 0 0 10px rgba(124,58,237,0.07), 0 0 30px var(--glow);
    animation: float 4s ease-in-out infinite;
}
.welcome-title {
    font-family: 'Fraunces', serif;
    font-size: 1.55rem;
    font-weight: 300;
    color: var(--text);
    margin-bottom: 0.45rem;
}
.welcome-title em { font-style: italic; color: var(--accent2); }
.welcome-sub {
    font-size: 0.84rem;
    color: var(--text2);
    max-width: 400px;
    margin: 0 auto 2rem;
    line-height: 1.65;
}
.suggest-lbl {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text3);
    font-weight: 600;
    margin-bottom: 0.65rem;
    text-align: left;
}

/* ══════════════════════════════════════
   CHAT INPUT — FIX: full width matching body
══════════════════════════════════════ */
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 0px !important;
    left: 0 !important;
    right: 0 !important;
    padding: 0.875rem 2rem 1.25rem !important;
    background: linear-gradient(to top, var(--bg) 65%, transparent) !important;
    z-index: 999 !important;
    display: flex !important;
    justify-content: center !important;
}
[data-testid="stChatInput"] > div {
    width: 100% !important;
    margin: 0 0 0 400px !important;       /* center horizontally */
    max-width: 860px !important;
    # background: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(167,139,250,0.05) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.9rem !important;
    caret-color: var(--accent) !important;
    min-height: 28px !important;
    width: 100% !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    # color: var(--text3) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* Typing indicator */
.typing-row {
    display: flex;
    align-items: flex-start;
    gap: 0.65rem;
    margin: 0.75rem 0;
    animation: slideRight 0.28s ease both;
}
.typing-bubble {
    background: var(--bot-bg);
    border: 1px solid var(--border2);
    border-radius: 4px 18px 18px 18px;
    padding: 0.8rem 1.05rem;
    display: flex;
    gap: 5px;
    align-items: center;
}
.dot {
    width: 6px; height: 6px;
    background: var(--accent);
    border-radius: 50%;
    animation: bounce 1.1s ease infinite;
}
.dot:nth-child(2) { animation-delay: 0.18s; }
.dot:nth-child(3) { animation-delay: 0.36s; }

/* ══════════════════════════════════════
   ANIMATIONS
══════════════════════════════════════ */
@keyframes fadeDown   { from{opacity:0;transform:translateY(-10px)} to{opacity:1;transform:none} }
@keyframes fadeUp     { from{opacity:0;transform:translateY(14px)}  to{opacity:1;transform:none} }
@keyframes slideLeft  { from{opacity:0;transform:translateX(16px)}  to{opacity:1;transform:none} }
@keyframes slideRight { from{opacity:0;transform:translateX(-16px)} to{opacity:1;transform:none} }
@keyframes float      { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-7px)} }
@keyframes pulse-green{
    0%,100%{box-shadow:0 0 5px #34D399;opacity:1}
    50%    {box-shadow:0 0 11px #34D399;opacity:0.7}
}
@keyframes bounce {
    0%,60%,100%{transform:translateY(0)}
    30%        {transform:translateY(-5px)}
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Session State
# ══════════════════════════════════════════════════════════════════════════════

def init_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "history" not in st.session_state:
        st.session_state.history = ChatHistory(session_id=st.session_state.session_id)
    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []
    if "config_valid" not in st.session_state:
        try:
            validate_all()
            st.session_state.config_valid = True
            st.session_state.config_error = None
        except Exception as e:
            st.session_state.config_valid = False
            st.session_state.config_error = str(e)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — no non-functional nav tabs
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:

        st.markdown("""
        <div class="sb-logo">
            <div class="sb-logo-icon">🤝</div>
            <div class="sb-logo-text">HR<span class="sb-logo-dot">.</span>ai</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("＋  New Chat", use_container_width=True):
            st.session_state.session_id       = str(uuid.uuid4())
            st.session_state.history          = ChatHistory(session_id=st.session_state.session_id)
            st.session_state.display_messages = []
            st.rerun()

        st.markdown("""
        <div class="sb-card">
            <div class="sb-card-title">About</div>
            <div class="sb-card-body">
                Ask anything about company HR policies.<br><br>
                Answers are grounded in <span>official policy documents</span>
                and cited with sources.
            </div>
        </div>

        <div class="sb-card">
            <div class="sb-card-title">Topics I can help with</div>
            <div class="sb-card-body">
                🏖️ Leave &amp; Time Off<br>
                💰 Reimbursements<br>
                🏠 Work From Home<br>
                📈 Performance Reviews<br>
                👶 Parental Leave<br>
                🏥 Health Benefits<br>
                📝 Resignation &amp; Notice<br>
                🎓 Learning &amp; Development
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sb-footer">
            Llama 3.1 · Groq · v1.0
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Message Renderer
# FIX: content and sources rendered as SEPARATE st.markdown() calls
# so sources HTML is never double-escaped
# ══════════════════════════════════════════════════════════════════════════════

def render_message(msg: dict):
    role         = msg["role"]
    content      = msg["content"]
    sources      = msg.get("sources", [])
    message_type = msg.get("message_type", MessageType.ANSWER)
    latency_ms   = msg.get("latency_ms")

    if role == "user":
        # Escape user input — never trust user HTML
        safe = content.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        st.markdown(f"""
        <div class="msg-row-user">
            <div class="bubble-user">{safe}</div>
            <div class="mini-av mini-av-user">👤</div>
        </div>
        """, unsafe_allow_html=True)

    elif role == "assistant":
        bubble = "bubble-fallback" if message_type == MessageType.FALLBACK else "bubble-bot"

        # Escape LLM content for safety
        safe = content.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        # Restore newlines as <br> for readability
        safe = safe.replace("\n", "<br>")

        # ── FIX: render bubble content first ──
        st.markdown(f"""
        <div class="msg-row-bot">
            <div class="mini-av mini-av-bot">🤝</div>
            <div class="{bubble}">{safe}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── FIX: render sources as completely separate st.markdown call ──
        # This is our own HTML — never escaped, always renders correctly
        if sources:
            seen  = set()
            pills = ""
            for s in sources:
                if s.source_file not in seen:
                    seen.add(s.source_file)
                    pills += f'<span class="src-pill">📄 {s.source_file}</span>'
            lat = f'<span class="src-lat">⚡ {latency_ms:.0f}ms</span>' if latency_ms else ""
            st.markdown(f"""
            <div style="margin-left:48px; margin-top:-0.5rem; margin-bottom:0.25rem;">
                <div class="src-bar">
                    <span class="src-lbl">Sources</span>
                    {pills}
                    {lat}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Suggested Questions
# ══════════════════════════════════════════════════════════════════════════════

def render_suggested_questions():
    from prompts import SUGGESTED_QUESTIONS
    st.markdown('<div class="suggest-lbl">✦ Try asking</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        with cols[i % 2]:
            if st.button(q, key=f"sq_{i}"):
                return q
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Handle Question
# ══════════════════════════════════════════════════════════════════════════════

def handle_question(question: str):
    st.session_state.display_messages.append({
        "role": "user", "content": question
    })
    st.session_state.history.add(Role.USER, question)

    request = QueryRequest(
        question   = question,
        session_id = st.session_state.session_id
    )

    placeholder = st.empty()
    placeholder.markdown("""
    <div class="typing-row">
        <div class="mini-av mini-av-bot">🤝</div>
        <div class="typing-bubble">
            <div class="dot"></div><div class="dot"></div><div class="dot"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        response = ask(request, st.session_state.history)
    except Exception as e:
        placeholder.empty()
        err = f"Something went wrong: {str(e)}"
        st.session_state.display_messages.append({
            "role": "assistant", "content": err,
            "sources": [], "message_type": MessageType.ERROR, "latency_ms": None
        })
        st.session_state.history.add(Role.ASSISTANT, err)
        return

    placeholder.empty()
    st.session_state.display_messages.append({
        "role":         "assistant",
        "content":      response.answer,
        "sources":      response.sources,
        "message_type": response.message_type,
        "latency_ms":   response.latency_ms,
    })
    st.session_state.history.add(Role.ASSISTANT, response.answer)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    init_session()
    render_sidebar()

    st.markdown("""
    <div class="chat-header">
        <div class="chat-header-left">
            <div class="online-dot"></div>
            Chat
        </div>
        <div class="chat-header-right">HR Policy Assistant · Online</div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.config_valid:
        st.error(f"⚠️ {st.session_state.config_error}")
        st.stop()

    if not st.session_state.display_messages:
        st.markdown("""
        <div class="welcome-wrap">
            <div class="welcome-av">🤝</div>
            <div class="welcome-title">Hello, I'm <em>Aria</em></div>
            <div class="welcome-sub">
                Your HR Policy Assistant — ask me anything about leave,
                benefits, reimbursements, code of conduct, or any company policy.
            </div>
        </div>
        """, unsafe_allow_html=True)

        suggested = render_suggested_questions()
        if suggested:
            handle_question(suggested)
            st.rerun()

    for msg in st.session_state.display_messages:
        render_message(msg)

    if user_input := st.chat_input("Type a message..."):
        handle_question(user_input)
        st.rerun()


if __name__ == "__main__":
    main()
