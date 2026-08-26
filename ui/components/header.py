"""
header.py - the chrome: nav strip, avatar, title, contact card, footer.

Pure presentation. Every string and URL comes from `config.ui`, so the contact
details and links are editable without touching Python - the same rule the rest
of the project follows.

The markup is the original app.py's, unchanged. Streamlit sandboxes <script>,
so the portfolio site's effects (the rotating conic border-beam on the avatar,
the aurora glow) are pure CSS in `ui/styles.css`.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from portfolio_chatbot.config import AppConfig

STYLES = Path(__file__).resolve().parents[1] / "styles.css"


@st.cache_data(show_spinner=False)
def _css() -> str:
    return STYLES.read_text(encoding="utf-8")


def inject_styles() -> None:
    st.markdown(f"<style>{_css()}</style>", unsafe_allow_html=True)


def nav(cfg: AppConfig) -> None:
    st.markdown(f"""
<div class="chat-nav">
    <a class="chat-nav-word" href="{cfg.ui.portfolio_url}" target="_blank">{cfg.ui.contact.name}</a>
    <a class="chat-nav-back" href="{cfg.ui.portfolio_url}" target="_blank">&larr; Back to Portfolio</a>
</div>
<div class="chat-hero-avatar-wrap">
    <div class="chat-hero-avatar">
        <img src="{cfg.ui.assistant_avatar}" alt="">
    </div>
</div>
""", unsafe_allow_html=True)


def hero(cfg: AppConfig) -> None:
    st.title(cfg.ui.heading)
    st.markdown(f'<p class="hero-tagline">{cfg.ui.tagline}</p>', unsafe_allow_html=True)

    links = "\n".join(
        f'        <a href="{link.url}" target="_blank">{link.label}</a>'
        for link in cfg.ui.contact.links
    )
    meta = "\n".join(f'    <p class="contact-meta">{line}</p>' for line in cfg.ui.contact.meta)

    st.markdown(f"""
<div class="contact-card">
    <h2>{cfg.ui.contact.name}</h2>
{meta}
    <div class="contact-links">
{links}
    </div>
</div>
<hr>
""", unsafe_allow_html=True)
    st.markdown(f'<p class="caption-text">{cfg.ui.caption}</p>', unsafe_allow_html=True)


def about(cfg: AppConfig) -> None:
    models = ", ".join(m.split("/")[-1] for m in cfg.llm.model_chain)
    with st.expander("About this chatbot", expanded=False):
        st.info(f"""
Powered by Groq, with automatic fallback across {len(cfg.llm.model_chain)} free-tier
models ({models}) when one hits its rate limit.

Built as a LangGraph pipeline: query expansion, type-aware retrieval, generation,
and a hard URL allowlist that strips any link not present in the source resume.

Conversation memory is summarized past {cfg.memory.summarize_after_n_messages}
messages to stay inside the token budget.

Questions? Email {cfg.ui.contact.meta[-1] if cfg.ui.contact.meta else ''}
""")


def footer(cfg: AppConfig) -> None:
    links = " &middot;\n    ".join(
        f'<a href="{link.url}" target="_blank">{link.label}</a>'
        for link in cfg.ui.contact.links
    )
    st.markdown(f"""
<div class="chat-footer">
    &copy; 2026 {cfg.ui.contact.name} &middot;
    {links}
</div>
""", unsafe_allow_html=True)


__all__ = ["inject_styles", "nav", "hero", "about", "footer"]
