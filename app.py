import re
import time
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Config ---
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

MAX_CONTEXT_CHARS = 8000
SHOW_DEBUG = False

# Assistant chat-avatar image, shared with the portfolio site's chat FAB
# (keeps the two experiences visually tied together).
ASSISTANT_AVATAR = "https://nikhileshnarkhede.github.io/portfolio/Image/chatbot-icon.jpeg"

# How long to pause between each streamed chunk (seconds).
# 0.025 ≈ 40 visible tokens/sec — feels natural, like ChatGPT.
# Raise to 0.04 for slower/more dramatic; lower to 0.01 for snappier.
STREAM_DELAY = 0.025

# ---------------------------------------------------------------------------
# ALLOWED URLS — the only links the chatbot is ever permitted to show.
# Any URL the LLM produces that is NOT in this set is silently stripped.
# ---------------------------------------------------------------------------
ALLOWED_URLS = {
    # Profile
    "https://www.linkedin.com/in/nikhileshnarkhede",
    "https://github.com/nikhileshnarkhede",
    "https://nikhileshnarkhede.github.io/portfolio/",
    # Research
    "https://doi.org/10.3390/jcs9060271",
    "https://doi.org/10.1007/s12008-026-02637-y",
    # Projects
    "https://nikhileshnarkhede.github.io/portfolio/projects/job-application-assistant.html",
    "https://nikhileshnarkhede.github.io/portfolio/projects/chat-youtube-videos.html",
    "https://nikhileshnarkhede.github.io/portfolio/projects/market-sentiment-prediction.html",
    "https://nikhileshnarkhede.github.io/portfolio/projects/nlp-text-classification.html",
    "https://nikhileshnarkhede.github.io/portfolio/projects/credit-risk-prediction.html",
    "https://nikhileshnarkhede.github.io/portfolio/projects/music-genre-classification.html",
    "https://nikhileshnarkhede.github.io/portfolio/projects/ml-business-forecasting.html",
    "https://nikhileshnarkhede.github.io/portfolio/projects/computer-vision.html",
    "https://nikhileshnarkhede.github.io/portfolio/projects/supply-chain-tracker.html",
    "https://nikhileshnarkhede.github.io/portfolio/projects/data-engineering-visualization.html",
    # Live app (this chatbot's own deployment)
    "https://nikhileshportfoliochatbot.streamlit.app/",
    # Certifications — verified credential links
    "https://verify.eicta.digitalcredentials.in/65e424b5-68ff-44eb-b974-fc73e27add6c",
    "https://verify.skilljar.com/c/549584kt98m9",
    "https://verify.skilljar.com/c/pcis3ygdauaw",
    "https://verify.skilljar.com/c/fvon5s9cdv7b",
    "https://verify.skilljar.com/c/ste8f6xptntw",
    "https://www.linkedin.com/learning/certificates/0a8d3abcc8338a4f3f9a21e604bebb32520811dda39a7ee696f921067ba8df11/",
    "https://verify.skilljar.com/c/exdsh88k8m5q",
    "https://verify.skilljar.com/c/qvr5g76rd8kq",
    "https://verify.skilljar.com/c/yiy2bcayhcgk",
    "https://verify.skilljar.com/c/f5u5x52ubngk",
    "https://verify.skilljar.com/c/hdsginsdk2kh",
    # Newsletter
    "https://www.linkedin.com/newsletters/weights-real-life-7447323018998988800/",
    # Project GitHub repos
    "https://github.com/nikhileshnarkhede/Chat_with_YouTube_Video",
    "https://github.com/nikhileshnarkhede/High-Perform-Scientific-Compute",
    "https://github.com/nikhileshnarkhede/SymbOptAI",
}

# ---------------------------------------------------------------------------
# URL SANITIZER
# Scans the LLM response for any markdown links [text](url) or bare URLs
# and removes those that are not in ALLOWED_URLS. This is a hard safety net
# that fires even if the LLM ignores the prompt instructions.
# ---------------------------------------------------------------------------
def sanitize_links(text: str) -> str:
    """Remove any markdown links or bare URLs not present in ALLOWED_URLS."""

    allowed_norm = {u.rstrip("/") for u in ALLOWED_URLS}

    def _check_markdown_link(match):
        label = match.group(1)
        url = match.group(2).strip().rstrip(")")
        if url.rstrip("/") in allowed_norm:
            return match.group(0)          # keep as-is
        return label                        # keep label text, drop the bad link

    def _check_bare_url(match):
        url = match.group(0).strip()
        if url.rstrip("/") in allowed_norm:
            return match.group(0)
        return ""                           # remove silently

    # 1) Handle markdown links: [label](url)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', _check_markdown_link, text)

    # 2) Handle bare URLs not already inside markdown syntax
    text = re.sub(r'(?<!\()(https?://[^\s\)\"\'<>]+)', _check_bare_url, text)

    return text


@st.cache_resource(show_spinner=False)
def get_llm(model_name: str):
    """Lazy + cached LLM init, one cached instance per model name."""
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=model_name,
        api_key=GROQ_API_KEY,
        temperature=0.3,
    )


# ---------------------------------------------------------------------------
# MODEL FALLBACK CHAIN
# Groq's free tier gives each model its OWN separate RPM/RPD/TPM/TPD quota.
# Rather than showing an error the moment the primary model is exhausted,
# retry the same request on the next general-purpose chat model in this
# list. TTS models (orpheus-*), transcription models (whisper-*), and
# classifier-only models (llama-prompt-guard-2-*) are excluded -- they can't
# serve open-ended chat. Ordered primary -> fallbacks:
#   openai/gpt-oss-20b    30 RPM / 1K RPD / 8K TPM / 200K TPD  (primary)
#   openai/gpt-oss-120b   30 RPM / 1K RPD / 8K TPM / 200K TPD  (bigger, same quota)
#   qwen/qwen3.6-27b      30 RPM / 1K RPD / 8K TPM / 200K TPD
#   groq/compound-mini    30 RPM /  250 RPD / 70K TPM           (much higher TPM, lower RPD)
# The session remembers whichever model last worked, so once a switch
# happens the app doesn't keep re-hitting the exhausted model every turn.
# ---------------------------------------------------------------------------
MODEL_CHAIN = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "groq/compound-mini",
]


def _is_rate_limited(error_msg: str) -> bool:
    m = error_msg.lower()
    return "rate_limit" in m or "429" in m or "rate limit" in m


@st.cache_resource(show_spinner=False)
def get_store():
    """Lazy + cached vector store init."""
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return FAISS.load_local("./faiss_db", embedder, allow_dangerous_deserialization=True)


@st.cache_resource(show_spinner=False)
def get_answer_chain(model_name: str):
    """Prompt + LLM chain cached per model to avoid rebuilding each rerun."""
    return prompt | get_llm(model_name) | StrOutputParser()

# ---------------------------------------------------------------------------
# TYPE-AWARE RETRIEVAL
# resume.txt is split by ingest.py into small, single-topic chunks tagged
# with a `chunk_type` in their metadata (experience, project, skills,
# research_project, publication, certification_group, education, ...).
# For category questions ("all your projects", "your skills") we pull EVERY
# chunk of that type so answers are complete; for open-ended questions we
# fall back to MMR for relevant + diverse coverage.
# Routing runs on the RAW user question (not the keyword-expanded one) to
# avoid the expansion keywords triggering the wrong category.
# ---------------------------------------------------------------------------
TYPE_ROUTES = [
    (("current job", "current role", "current company", "currently working",
      "where do you work", "where are you working", "choir", "sentrix", "uptimepowr"),
     ["experience"]),
    (("skill", "technolog", "tool", "stack", "framework", "librar", "language", "proficien"),
     ["skills"]),
    (("research", "publication", "paper", "pinn", "physics", "symbolic", "journal",
      "igbt", "rul", "remaining useful life", "mlflow"),
     ["research_project", "publication", "experience"]),
    (("education", "degree", "university", "gpa", "coursework", "school",
      "master", "bachelor", "stud", "colleg", "academic"),
     ["education"]),
    (("project", "built", "build", "application", "portfolio", "demo"),
     ["project"]),
    (("experience", "work", "job", "role", "position", "intern", "employ", "compan"),
     ["experience"]),
    (("certif", "credential", "course", "training", "license"),
     ["certification_group"]),
    (("newsletter", "writing", "weights & real", "blog"),
     ["newsletter"]),
    (("presentation", "conference", "symposium", "3mt", "thesis"),
     ["presentation"]),
    (("volunteer", "leadership", "mentor", "advisor"),
     ["leadership_role"]),
    (("recommend", "reference", "endorse"),
     ["recommendation"]),
]


def detect_types(text):
    t = text.lower()
    for keywords, types in TYPE_ROUTES:
        if any(k in t for k in keywords):
            return types
    return None


def retrieve(query, search_text=None):
    """Return the most useful chunks for `query`.
    `query`       -> raw user question, used only for category routing.
    `search_text` -> (optional) keyword-expanded text, used for ranking.
    """
    db = get_store()
    search_text = search_text or query
    types = detect_types(query)
    if types:
        try:
            docs = db.similarity_search(
                search_text, k=14, fetch_k=120, filter={"chunk_type": types}
            )
            if docs:
                return docs
        except Exception:
            pass  # fall through to MMR on any version/filter issue
    # Open-ended question (or filter returned nothing): MMR over everything.
    return db.max_marginal_relevance_search(
        search_text, k=10, fetch_k=30, lambda_mult=0.6
    )

# --- Query Enhancement ---
def enhance_query(query: str) -> str:
    """Expand query with relevant keywords for better retrieval"""
    query_lower = query.lower()

    if any(word in query_lower for word in ["skill", "technolog", "tool", "language"]):
        return f"{query} technical skills programming frameworks libraries"
    elif any(word in query_lower for word in ["project", "built", "developed", "created"]):
        return f"{query} projects applications systems built developed"
    elif any(word in query_lower for word in ["experience", "work", "job", "role", "position"]):
        return f"{query} experience work employment roles responsibilities"
    elif any(word in query_lower for word in ["research", "publication", "paper", "study"]):
        return f"{query} research publications papers studies machine learning"
    elif any(word in query_lower for word in ["education", "degree", "university", "study"]):
        return f"{query} education degree university certification"
    else:
        return query

# --- Format docs with better structure ---
def format_docs(retrieved_docs):
    """Format retrieved documents with clear sections"""
    if not retrieved_docs:
        return "No relevant information found."

    formatted = []
    for i, doc in enumerate(retrieved_docs, 1):
        content = doc.page_content.strip()
        formatted.append(f"[Section {i}]\n{content}")

    context = "\n\n".join(formatted)
    return context[:MAX_CONTEXT_CHARS]

# --- Format conversation history with summarization ---
def format_chat_history(messages):
    """Format chat history - uses summary + recent messages"""
    if not messages or len(messages) == 0:
        if st.session_state.conversation_summary:
            return f"[Earlier conversation summary]\n{st.session_state.conversation_summary}\n\n[Current conversation]\nNo messages yet."
        return "No previous conversation."

    history_parts = []

    if st.session_state.conversation_summary:
        history_parts.append(f"[Earlier conversation summary]\n{st.session_state.conversation_summary}\n")

    if messages:
        history_parts.append("[Recent conversation]")
        for msg in messages:
            role = "Recruiter" if msg["role"] == "user" else "Nikhilesh"
            history_parts.append(f"{role}: {msg['content']}")

    return "\n".join(history_parts)

# --- Prompt Template ---
prompt = PromptTemplate(
    template="""
        You are Nikhilesh, speaking directly to a recruiter about your own background, skills, and experience.
        You are NOT a third-party assistant — you ARE Nikhilesh. Use "I", "my", "me" naturally.

        Your personality:
        - Caring: You genuinely want the recruiter to feel heard and valued. Acknowledge their questions warmly before diving in.
        - Optimistic: You see the bright side — frame your experience as growth, learning, and impact.
        - Assertive: You own your achievements confidently. Don't undersell yourself. State what you've done clearly.
        - Confident: You speak with certainty. No hedging words like "maybe", "I think", or "possibly" unless truly unsure.
        - Soft tone: Your confidence doesn't come across as aggressive. It feels approachable, genuine, and easy to connect with.
        - Easy to understand: Keep it simple. Avoid unnecessary jargon. If you use a technical term, briefly explain what it means.

        **IMPORTANT: Use the conversation history below to maintain context and answer follow-up questions naturally.**

        Previous Conversation:
        {chat_history}

        Tone examples:
        - Instead of: "I have experience in machine learning."
        - Say: "Machine learning has been a big part of my journey — I've built models that actually solved real problems, like predicting material properties with over 99% accuracy."

        - Instead of: "I worked on a RAG project."
        - Say: "One project I really enjoyed was building a conversational AI system — it lets users ask questions about videos and get accurate answers instantly. Super rewarding to build end to end."

        Rules:
        - Answer ONLY from the provided resume context. Never make up information.
        - Use the conversation history to understand follow-up questions like "tell me more", "what about that project?", "and the others?"
        - If the context doesn't cover the question, say it warmly:
          "That's a great question! I don't have that detail here right now, but I'd love to chat about it directly — feel free to reach out!"
        - Keep answers focused but not too short. Give enough detail to impress, but don't overwhelm.
        - End with a warm, inviting line when appropriate — like offering to elaborate or connect.

        === STRICT URL RULES (read carefully) ===
        - You may ONLY include a URL if you can see the EXACT, COMPLETE URL string printed word-for-word inside the Resume Context sections above.
        - Copy the URL character-by-character. Do NOT paraphrase, guess, reconstruct, or invent any URL.
        - If you cannot find the exact URL for a project inside the Resume Context, do NOT include any link for that project. Simply describe the project without a link.
        - NEVER fabricate, shorten, or modify a URL. No partial links, no placeholder links, no assumed URLs.
        - Format confirmed URLs as a markdown link: [View project](EXACT_URL_FROM_CONTEXT)
        - If multiple projects are mentioned, include each project's URL only if it appears verbatim in the context.
        - When in doubt — omit the link entirely. A missing link is far better than a broken or wrong one.
        ==========================================

        About Me (use this when asked who you are or about your background):
        I am a Master's student in Data Science. I hold a Bachelor's degree in Mechanical Engineering and an Advanced Certification in Artificial Intelligence and Machine Learning from IIT Kanpur. I bring a strong academic foundation and a deep enthusiasm for statistical modeling, mathematics, and using data to solve real-world problems. My approach spans the entire data science lifecycle — from data ingestion and preprocessing to model deployment and performance optimization — and is grounded in both theory and practical application. I excel at translating complex, ambiguous domain problems into clear, impactful machine learning tasks, and I specialize in building interpretable, scalable, and production-ready models that deliver measurable results. I am equally adept at communicating data-driven insights to both technical and non-technical audiences. I have a strong interest in research and enjoy working at the intersection of data, experimentation, and domain expertise to develop innovative, AI-driven solutions. For me, data science is more than just a discipline — it is a powerful tool for transforming ideas into actionable strategies and real-world impact.

        Use the retrieved resume context below for project and experience details.
        - When asked about projects, list relevant projects and include available links from context.
        - When asked about experience, prioritize current/recent ML roles first, then supporting roles briefly.

        Resume Context:
        {context}

        Question: {question}
    """,
    input_variables=["context", "question", "chat_history"]
)

# --- Streamlit UI ---
st.set_page_config(page_title="Chat with Nikhilesh", page_icon="N")

st.markdown("""
<style>
    :root {
        --black: #000000;
        --light-bg: #f5f5f7;
        --near-black: #1d1d1f;
        --white: #ffffff;
        --apple-blue: #0071e3;
        --link-light: #0066cc;
        --link-dark: #2997ff;
        --transition: all 0.3s ease;
        --text-on-dark: #ffffff;
        --text-secondary-dark: rgba(255, 255, 255, 0.72);
        --ds-1: #272729;
        --ds-4: #2a2a2d;
        --shadow-card: rgba(0, 0, 0, 0.22) 3px 5px 30px 0px;
        --font-display: -apple-system, "SF Pro Display", "Helvetica Neue", Helvetica, Arial, sans-serif;
        --font-body: -apple-system, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif;
    }

    /* Streamlit ships its OWN theming variables (--text-color, etc.)
       that many built-in components read internally (alerts, markdown
       lists, captions) — separate from our custom vars above. Without
       overriding these too, those components keep Streamlit's default
       dark-on-light text color, which is invisible on our black page. */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        --text-color: #ffffff;
        color: #ffffff;
    }

    /* ============================================================
       PORTFOLIO-MATCHED UI — aurora glow, border-beam avatar,
       glass nav strip, button lift+bloom. Pure CSS only (Streamlit
       sandboxes <script>, so no Three.js/GSAP here) — these effects
       mirror the main portfolio site's language using CSS animation
       + backdrop-filter instead of a JS engine.
       ============================================================ */
    @property --beam {
        syntax: '<angle>';
        initial-value: 0deg;
        inherits: false;
    }

    /* soft aurora glow baked into the page background itself (not a
       separate positioned layer) — backgrounds always paint behind an
       element's own content, so this can never end up on top of text */
    .stApp {
        background:
            radial-gradient(circle at 6% -4%, rgba(0, 113, 227, 0.16), transparent 30%),
            radial-gradient(circle at 96% 104%, rgba(41, 151, 255, 0.13), transparent 30%),
            var(--black);
    }

    /* slim glass nav strip, echoes the portfolio's navbar */
    .chat-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        max-width: 860px;
        margin: 0 auto 1.1rem auto;
        padding: 0.6rem 0.2rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .chat-nav-word {
        font-family: var(--font-display);
        font-size: 0.9375rem;
        font-weight: 600;
        color: var(--white);
        text-decoration: none;
        letter-spacing: -0.16px;
    }
    .chat-nav-back {
        font-family: var(--font-body);
        font-size: 0.8125rem;
        color: var(--link-dark);
        text-decoration: none;
        letter-spacing: -0.1px;
        padding: 6px 14px;
        border: 1px solid rgba(41, 151, 255, 0.35);
        border-radius: 980px;
        transition: var(--transition);
    }
    .chat-nav-back:hover {
        background: rgba(41, 151, 255, 0.12);
        border-color: var(--link-dark);
        color: #ffffff;
    }

    /* avatar with a rotating conic border-beam, matching the FAB ring
       on the main portfolio site */
    .chat-hero-avatar-wrap {
        display: flex;
        justify-content: center;
        margin-bottom: 0.6rem;
    }
    .chat-hero-avatar {
        position: relative;
        width: 64px; height: 64px;
    }
    .chat-hero-avatar img {
        width: 100%; height: 100%;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid rgba(255,255,255,0.14);
        display: block;
    }
    .chat-hero-avatar::before {
        content: '';
        position: absolute;
        inset: -4px;
        border-radius: 50%;
        padding: 2px;
        pointer-events: none;
        background: conic-gradient(from var(--beam),
            rgba(41, 151, 255, 0) 0deg,
            rgba(41, 151, 255, 0.9) 120deg,
            rgba(142, 200, 255, 1) 160deg,
            rgba(41, 151, 255, 0.9) 200deg,
            rgba(41, 151, 255, 0) 330deg);
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        mask-composite: exclude;
    }
    @media (prefers-reduced-motion: no-preference) {
        .chat-hero-avatar::before { animation: chatBeam 3.6s linear infinite; }
    }
    @keyframes chatBeam { to { --beam: 360deg; } }

    /* footer, matches the portfolio's dark footer bar */
    .chat-footer {
        position: relative;
        z-index: 1;
        max-width: 860px;
        margin: 1.6rem auto 0.4rem auto;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.10);
        text-align: center;
        font-family: var(--font-body);
        font-size: 0.78rem;
        color: rgba(255,255,255,0.42);
    }
    .chat-footer a {
        color: rgba(255,255,255,0.6);
        text-decoration: none;
    }
    .chat-footer a:hover { color: var(--link-dark); }

    .stApp {
        color: var(--text-on-dark);
        font-family: var(--font-body);
        overflow-x: clip;
    }

    header[data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0.8) !important;
        backdrop-filter: saturate(180%) blur(20px);
        -webkit-backdrop-filter: saturate(180%) blur(20px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    header[data-testid="stHeader"] * { color: var(--white) !important; }

    .main {
        background: transparent;
        color: var(--text-on-dark);
    }

    h1 {
        color: var(--white);
        text-align: center;
        font-family: var(--font-display);
        font-size: clamp(2rem, 5vw, 3rem);
        font-weight: 600;
        line-height: 1.07;
        letter-spacing: -0.28px;
        margin-bottom: 0.75rem;
    }

    .hero-tagline {
        text-align: center;
        font-family: var(--font-display);
        font-size: clamp(1rem, 2.5vw, 1.3125rem);
        font-weight: 400;
        line-height: 1.19;
        letter-spacing: 0.231px;
        color: rgba(255,255,255,0.85);
        margin: 0 0 0.5rem 0;
    }

    .caption-text {
        text-align: center;
        font-family: var(--font-body);
        color: rgba(255,255,255,0.65);
        font-size: 1.0625rem;
        line-height: 1.47;
        letter-spacing: -0.374px;
        margin-bottom: 1rem;
    }

    .contact-card {
        max-width: 860px;
        margin: 0 auto 0.9rem auto;
        background: var(--ds-1);
        border-radius: 12px;
        box-shadow: var(--shadow-card);
        padding: 1rem 1.2rem;
        text-align: center;
    }

    .contact-card h2 {
        color: var(--white);
        margin: 0 0 0.2rem 0;
        font-size: 1.25rem;
        font-family: var(--font-display);
        font-weight: 600;
        letter-spacing: -0.224px;
    }

    .contact-meta {
        color: var(--text-secondary-dark);
        font-size: 0.85rem;
        margin: 0.15rem 0;
        letter-spacing: -0.12px;
    }

    .contact-links {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 1rem;
        margin-top: 0.6rem;
    }

    .contact-links a {
        color: #2997ff;
        text-decoration: none;
        font-size: 0.9rem;
        letter-spacing: -0.12px;
        transition: color 0.2s ease, opacity 0.2s ease;
    }

    .contact-links a:hover { text-decoration: underline; opacity: 0.85; }

    [data-testid="stChatMessage"] {
        border-radius: 12px;
        margin-bottom: 0.6rem;
    }

    [data-testid="stChatMessage"][data-st-chat-message-role="assistant"] {
        background: #272729;
    }

    [data-testid="stChatMessage"][data-st-chat-message-role="user"] {
        background: #1d1d1f;
    }

    [data-testid="stChatInput"] input {
        background: #1d1d1f !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.16) !important;
        border-radius: 8px !important;
    }

    [data-testid="stChatInput"] input:focus {
        border-color: #0071e3 !important;
        outline: 2px solid var(--apple-blue) !important;
        outline-offset: 1px;
        box-shadow: 0 0 0 2px rgba(0, 113, 227, 0.35) !important;
    }

    .stButton button:focus-visible,
    .contact-links a:focus-visible {
        outline: 2px solid var(--apple-blue);
        outline-offset: 2px;
        border-radius: 980px;
    }

    .stButton button {
        background: transparent;
        color: var(--link-dark);
        border: 1px solid var(--link-dark);
        border-radius: 980px;
        font-family: var(--font-body);
        font-size: 1.0625rem;
        font-weight: 400;
        line-height: 1;
        padding: 8px 18px;
        letter-spacing: -0.374px;
        transition: var(--transition), transform 0.22s cubic-bezier(.34,1.56,.64,1), box-shadow 0.22s ease;
    }

    .stButton button:hover {
        color: var(--white);
        border-color: rgba(255,255,255,0.70);
        background: rgba(255,255,255,0.06);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.28), 0 0 16px rgba(41,151,255,0.18);
    }

    [data-testid="stExpander"] {
        background: #272729;
        border-radius: 12px;
        box-shadow: var(--shadow-card);
    }

    [data-testid="stExpander"] * {
        color: var(--text-on-dark);
    }

    /* Explicit, high-priority overrides for every native Streamlit text
       container (markdown, alerts, captions, lists) — !important because
       Streamlit's own injected stylesheet otherwise wins these on
       specificity/order in some versions. */
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] em,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p,
    [data-testid="stText"],
    [data-testid="stAlertContentInfo"],
    [data-testid="stAlertContentInfo"] p,
    [data-testid="stAlertContentWarning"],
    [data-testid="stAlertContentWarning"] p,
    [data-testid="stAlertContentError"],
    [data-testid="stAlertContentError"] p,
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] li {
        color: #ffffff !important;
    }

    hr {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.12);
        margin: 0.9rem 0 1rem 0;
    }

    .block-container {
        position: relative;
        z-index: 1;
        max-width: 960px;
        /* Streamlit's own header bar is position:fixed and sits on top of
           normal page flow — without enough top padding here, our nav
           strip renders partially UNDERNEATH it (only the bottom sliver
           visible). This clears it on both desktop and mobile header heights. */
        padding-top: 4.75rem;
        padding-bottom: 1.2rem;
    }

    [data-testid="stChatMessageContent"] p {
        font-size: 1.0625rem;
        line-height: 1.47;
        letter-spacing: -0.374px;
    }

    [data-testid="stHorizontalBlock"] {
        gap: 0.5rem;
    }

    .stButton > button {
        min-height: 40px;
    }

    [data-testid="stChatInput"] {
        padding-top: 0.45rem;
        background: linear-gradient(to top, rgba(0,0,0,0.85), rgba(0,0,0,0));
    }

    [data-testid="stChatInput"] input {
        font-size: 16px !important;
        min-height: 44px;
    }

    @media (max-width: 768px) {
        .chat-nav {
            padding: 0.5rem 0.1rem;
            margin-bottom: 0.8rem;
        }

        .chat-nav-word {
            font-size: 0.8125rem;
        }

        .chat-nav-back {
            font-size: 0.72rem;
            padding: 5px 11px;
            white-space: nowrap;
        }

        .chat-hero-avatar {
            width: 48px;
            height: 48px;
        }

        .chat-hero-avatar-wrap {
            margin-bottom: 0.4rem;
        }

        .chat-footer {
            font-size: 0.7rem;
            margin-top: 1.2rem;
        }

        .block-container {
            padding-top: 3.75rem;
            padding-left: 0.7rem;
            padding-right: 0.7rem;
            padding-bottom: 1rem;
        }

        h1 {
            font-size: 1.45rem;
            margin-bottom: 0.45rem;
        }

        .caption-text {
            font-size: 0.88rem;
            margin-bottom: 0.55rem;
        }

        .contact-card {
            padding: 0.8rem 0.8rem;
            margin-bottom: 0.7rem;
        }

        .contact-card h2 {
            font-size: 1.05rem;
        }

        .contact-meta {
            font-size: 0.8rem;
        }

        .contact-links {
            gap: 0.6rem;
            flex-direction: column;
            align-items: center;
        }

        [data-testid="stChatMessage"] {
            margin-bottom: 0.45rem;
            border-radius: 7px;
        }

        [data-testid="stChatMessageContent"] p {
            font-size: 0.94rem;
            line-height: 1.5;
        }

        .stButton > button {
            width: 100%;
            font-size: 0.85rem;
            min-height: 42px;
            padding: 0.4rem 0.7rem;
        }

        [data-testid="stExpander"] {
            border-radius: 7px;
        }
    }

    @media (max-width: 480px) {
        .chat-nav-word {
            font-size: 0.75rem;
        }

        .chat-nav-back {
            font-size: 0.68rem;
            padding: 4px 9px;
        }

        .chat-hero-avatar {
            width: 42px;
            height: 42px;
        }

        .chat-footer {
            font-size: 0.66rem;
            line-height: 1.6;
        }

        .block-container {
            padding-top: 3.5rem;
            padding-left: 0.55rem;
            padding-right: 0.55rem;
        }

        h1 {
            font-size: 1.28rem;
            letter-spacing: -0.2px;
        }

        .contact-card h2 {
            font-size: 1rem;
        }

        .contact-links a {
            font-size: 0.84rem;
        }

        [data-testid="stChatMessageContent"] p {
            font-size: 0.91rem;
        }

        [data-testid="stChatInput"] input {
            min-height: 46px;
        }
    }

    @media (prefers-reduced-motion: no-preference) {
        @keyframes chatFadeUp {
            from { opacity: 0; transform: translateY(16px); }
            to   { opacity: 1; transform: none; }
        }

        h1            { animation: chatFadeUp 0.6s cubic-bezier(0.22,0.61,0.36,1) both; }
        .caption-text { animation: chatFadeUp 0.6s cubic-bezier(0.22,0.61,0.36,1) 0.06s both; }
        .contact-card { animation: chatFadeUp 0.6s cubic-bezier(0.22,0.61,0.36,1) 0.12s both; }

        [data-testid="stChatMessage"] {
            animation: chatFadeUp 0.42s cubic-bezier(0.22,0.61,0.36,1) both;
        }

        [data-testid="stHorizontalBlock"] [data-testid="column"] .stButton {
            animation: chatFadeUp 0.5s cubic-bezier(0.22,0.61,0.36,1) both;
        }
        [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(1) .stButton { animation-delay: 0.10s; }
        [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(2) .stButton { animation-delay: 0.18s; }
        [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(3) .stButton { animation-delay: 0.26s; }
        [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(4) .stButton { animation-delay: 0.34s; }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="chat-nav">
    <a class="chat-nav-word" href="https://nikhileshnarkhede.github.io/portfolio/" target="_blank">Nikhilesh Narkhede</a>
    <a class="chat-nav-back" href="https://nikhileshnarkhede.github.io/portfolio/" target="_blank">&larr; Back to Portfolio</a>
</div>
<div class="chat-hero-avatar-wrap">
    <div class="chat-hero-avatar">
        <img src="https://nikhileshnarkhede.github.io/portfolio/Image/chatbot-icon.jpeg" alt="">
    </div>
</div>
""", unsafe_allow_html=True)

st.title("Chat with Nikhilesh")
st.markdown('<p class="hero-tagline">Data Science Researcher &amp; ML/DL Engineer</p>', unsafe_allow_html=True)

# --- Contact Card ---
st.markdown("""
<div class="contact-card">
    <h2>Nikhilesh Narkhede</h2>
    <p class="contact-meta">US Work Authorized | +1 508-509-3697</p>
    <p class="contact-meta">narkhede.nikhilesh@gmail.com</p>
    <div class="contact-links">
        <a href="https://www.linkedin.com/in/nikhileshnarkhede" target="_blank">LinkedIn</a>
        <a href="https://github.com/nikhileshnarkhede" target="_blank">GitHub</a>
        <a href="https://nikhileshnarkhede.github.io/portfolio/" target="_blank">Portfolio</a>
    </div>
</div>
<hr>
""", unsafe_allow_html=True)
st.markdown('<p class="caption-text">Ask me about my skills, projects, research, and experience.</p>', unsafe_allow_html=True)

with st.expander("About this chatbot", expanded=False):
    st.info("""
    Powered by Groq — auto-switches across multiple free-tier models
    (GPT-OSS 20B/120B, Qwen3.6-27B, Compound-Mini) if one hits its rate limit

    Features:
    - Automatic model fallback on rate limits
    - Fast responses
    - Streaming responses
    - Conversation memory
    - Auto-summarization (prevents token overflow)
    - Query expansion for better answers

    Questions? Email: narkhede.nikhilesh@gmail.com
    """)

# --- Quick Action Buttons ---
button_questions = {
    "Skills": "What are all your technical skills?",
    "Projects": "Tell me about all the projects you have built.",
    "Research": "Tell me about your research work.",
    "Experience": "Walk me through your work experience."
}

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "triggered_question" not in st.session_state:
    st.session_state.triggered_question = None
if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = ""
if "runtime_ready" not in st.session_state:
    st.session_state.runtime_ready = False
if "active_model_idx" not in st.session_state:
    st.session_state.active_model_idx = 0

# --- Auto-summarize ---
def auto_summarize_if_needed():
    message_count = len(st.session_state.messages)

    if message_count > 12:
        messages_to_summarize = st.session_state.messages[:-6]

        conv_text = ""
        for msg in messages_to_summarize:
            role = "Recruiter" if msg["role"] == "user" else "Nikhilesh"
            conv_text += f"{role}: {msg['content']}\n\n"

        summary_prompt = f"""Briefly summarize this conversation between a recruiter and Nikhilesh in 2-3 sentences.
Focus on: topics discussed, projects mentioned, skills asked about.

Conversation:
{conv_text}

Brief summary:"""

        try:
            model_name = MODEL_CHAIN[st.session_state.active_model_idx]
            summary = get_llm(model_name).invoke(summary_prompt)
            st.session_state.conversation_summary = summary.content if hasattr(summary, 'content') else str(summary)
            st.session_state.messages = st.session_state.messages[-6:]
        except Exception:
            st.session_state.conversation_summary = "Earlier conversation covered various topics about Nikhilesh's background."
            st.session_state.messages = st.session_state.messages[-6:]

# Only show buttons if chat is empty
if len(st.session_state.messages) == 0:
    cols = st.columns(4)
    for i, (label, question) in enumerate(button_questions.items()):
        with cols[i]:
            if st.button(label, key=f"btn_{i}", use_container_width=True, help=question):
                st.session_state.triggered_question = question

# --- Display Chat History ---
for msg in st.session_state.messages:
    avatar = ASSISTANT_AVATAR if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# --- Chat Input ---
user_input = st.chat_input("Ask something about my background...")

if st.session_state.triggered_question and not user_input:
    user_input = st.session_state.triggered_question
    st.session_state.triggered_question = None

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        chain_input = {
            "question": user_input,
            "chat_history": format_chat_history(st.session_state.messages)
        }

        if not st.session_state.runtime_ready:
            with st.spinner("Initializing AI engine..."):
                get_store()
                get_answer_chain(MODEL_CHAIN[st.session_state.active_model_idx])
            st.session_state.runtime_ready = True

        enhanced = enhance_query(user_input)
        if SHOW_DEBUG and enhanced != user_input:
            with st.expander("Enhanced Query (Debug)", expanded=False):
                st.caption(f"Original: {user_input}")
                st.caption(f"Enhanced: {enhanced}")

        docs = retrieve(user_input, enhanced)
        context = format_docs(docs)

        if SHOW_DEBUG:
            with st.expander("Retrieved Context (Debug)", expanded=False):
                st.text(context[:1000] + "..." if len(context) > 1000 else context)

        if SHOW_DEBUG and (len(st.session_state.messages) > 0 or st.session_state.conversation_summary):
            with st.expander("Conversation Memory (Debug)", expanded=False):
                st.text(chain_input["chat_history"])
                if st.session_state.conversation_summary:
                    st.info("Auto-summarization active: keeping last 6 messages plus summary to save tokens.")

        model_note_placeholder = st.empty()
        answer_placeholder = st.empty()
        full_answer = ""

        start_idx = st.session_state.active_model_idx
        attempt_order = list(range(start_idx, len(MODEL_CHAIN))) + list(range(0, start_idx))

        success = False
        last_error = None

        for attempt_i, idx in enumerate(attempt_order):
            model_name = MODEL_CHAIN[idx]
            full_answer = ""

            if attempt_i > 0:
                model_note_placeholder.caption("Primary model is at capacity, switching to a backup model.")
                answer_placeholder.markdown("...")

            try:
                for chunk in get_answer_chain(model_name).stream({
                    "context": context,
                    "question": user_input,
                    "chat_history": chain_input["chat_history"]
                }):
                    full_answer += chunk
                    answer_placeholder.markdown(full_answer + "...")  # streaming cursor
                    time.sleep(STREAM_DELAY)                          # paced rendering

                # Safety net: strip any URL not in the allowlist
                full_answer = sanitize_links(full_answer)

                # Final render, no cursor
                answer_placeholder.markdown(full_answer)
                model_note_placeholder.empty()

                st.session_state.active_model_idx = idx
                success = True
                break

            except Exception as e:
                last_error = e
                err_msg = str(e)
                if _is_rate_limited(err_msg) and attempt_i < len(attempt_order) - 1:
                    continue  # this model is exhausted, try the next one
                break  # non-rate-limit error, or the whole chain is exhausted

        if not success:
            error_msg = str(last_error) if last_error else "Unknown error"

            if _is_rate_limited(error_msg):
                st.error("All Models at Capacity")
                st.warning("""
                Every model in my fallback chain (GPT-OSS 20B/120B, Qwen3.6-27B, Compound-Mini) has hit Groq's free-tier limit right now.

                **Options:**
                1. Wait a few minutes and try again - limits reset per-minute and per-day
                2. Email me directly at narkhede.nikhilesh@gmail.com

                This usually happens during high traffic. Sorry for the inconvenience!
                """)
                full_answer = "I've hit rate limits across all my backup models right now. Please try again in a few minutes or contact me directly!"

            elif "api" in error_msg.lower() or "connection" in error_msg.lower():
                st.error("API Connection Error")
                st.warning("""
                There was a problem connecting to the AI service.

                **Options:**
                1. Refresh the page and try again
                2. Contact me directly at narkhede.nikhilesh@gmail.com
                """)
                full_answer = "I'm having trouble connecting right now. Please try refreshing the page or contact me directly!"

            else:
                st.error("Unexpected Error")
                st.warning(f"""
                Something went wrong. Here's the error:

                ```
                {error_msg[:200]}
                ```

                Please contact me at narkhede.nikhilesh@gmail.com
                """)
                full_answer = "I encountered an error. Please contact me directly to discuss your questions!"

    st.session_state.messages.append({"role": "assistant", "content": full_answer})
    auto_summarize_if_needed()

# --- Footer (always rendered, matches the portfolio's dark footer) ---
st.markdown("""
<div class="chat-footer">
    © 2026 Nikhilesh Narkhede &middot;
    <a href="https://nikhileshnarkhede.github.io/portfolio/" target="_blank">Portfolio</a> &middot;
    <a href="https://github.com/nikhileshnarkhede" target="_blank">GitHub</a> &middot;
    <a href="https://www.linkedin.com/in/nikhileshnarkhede" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
