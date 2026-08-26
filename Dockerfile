# ---------------------------------------------------------------------------
# Portfolio chatbot - shareable image.
#
# Designed so somebody else can run it on THEIR data with THEIR key:
#
#   docker run -p 8501:8501 --env-file .env \
#     -v "$PWD/resume.txt:/app/data/raw/resume.txt:ro" \
#     -v chatbot-index:/app/data/index \
#     nnarkhede/portfolio-chatbot:latest
#
# Two consequences follow from the resume arriving at RUN time, and they are
# what shape this file:
#
# 1. The index cannot be the only thing baked in, because the vectors depend on
#    a file that does not exist yet at build time. So the entrypoint ingests on
#    startup when the index for the current resume is missing.
#
# 2. The EMBEDDING MODEL still must be baked, or that first-run ingest also has
#    to download 90 MB of MiniLM before it can begin. The model is the slow,
#    constant part; the index is the fast, variable part. Bake the constant.
#
# The default resume is ingested at build time as well, so running the image
# with no mount starts instantly - only a substituted resume pays the ingest.
# ---------------------------------------------------------------------------

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

WORKDIR /build

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# Dependencies first: this layer is cached until requirements.txt changes, so
# editing a prompt does not re-download torch.
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

COPY configs/ ./configs/
COPY prompts/ ./prompts/
COPY data/raw/ ./data/raw/
COPY src/ ./src/
COPY scripts/ ./scripts/

# Ingest the DEFAULT resume here. This warms the HuggingFace cache at /opt/hf -
# which the runtime stage copies, so the model is never fetched over the
# network - and it means the image runs instantly when nobody mounts a resume.
RUN PATH=/install/bin:$PATH PYTHONPATH=/install/lib/python3.12/site-packages:/build/src \
    python scripts/ingest.py && test -d data/index

# ---------------------------------------------------------------------------

FROM python:3.12-slim AS runtime

# HF_HUB_OFFLINE is not an optimisation, it is a correctness fix. The model is
# in the image, but sentence-transformers still calls the Hub on every cold
# start to check the repo for updates - and in a container with no outbound
# network that call does not fail fast, it HANGS until the socket times out.
# The page loads, the first question spins forever, and the container reports
# healthy throughout.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /opt/hf /opt/hf
COPY --from=builder /build/data/index ./data/index

COPY configs/ ./configs/
COPY prompts/ ./prompts/
COPY data/raw/ ./data/raw/
COPY src/ ./src/
COPY ui/ ./ui/
COPY eval/ ./eval/
COPY scripts/ ./scripts/
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY docker-entrypoint.sh /usr/local/bin/entrypoint.sh

ARG RUN_FINGERPRINT=unknown
ARG BUILT_AT=unknown
LABEL org.opencontainers.image.title="nnarkhede/portfolio-chatbot" \
      org.opencontainers.image.description="Recruiter-facing RAG chatbot" \
      chatbot.run_fingerprint="${RUN_FINGERPRINT}" \
      chatbot.built_at="${BUILT_AT}"
ENV CHATBOT_RUN_FINGERPRINT="${RUN_FINGERPRINT}"

# /opt/hf must be chowned too. huggingface_hub writes lock files under HF_HOME
# whenever it resolves a repo - even a fully cached one - so leaving it
# root-owned is a PermissionError raised the first time somebody asks a
# question, long after the container reported healthy.
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /app/runs /app/eval/reports /app/eval/ratings \
 && chmod +x /usr/local/bin/entrypoint.sh \
 && chown -R app:app /app /opt/hf

USER app
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).read()==b'ok' else 1)"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
