#!/bin/sh
# ---------------------------------------------------------------------------
# Build the index for whatever resume is actually mounted, then start the app.
#
# `scripts/ingest.py` is a no-op when the index for the current fingerprint
# already exists, so the common paths cost nothing:
#
#   no resume mounted      -> the baked index matches  -> starts immediately
#   resume mounted, warm   -> index volume has it      -> starts immediately
#   resume mounted, cold   -> ingests once (~30-60s)   -> then starts
#
# That last case is why the index fingerprint hashes the resume's CONTENT and
# not its path. Hashing the path made a substituted resume collide with the
# baked index: the container would come up instantly and answer every question
# out of the wrong person's vectors, with nothing anywhere reporting a problem.
# ---------------------------------------------------------------------------
set -e

RESUME="${CHATBOT_RESUME:-/app/data/raw/resume.txt}"

if [ ! -f "$RESUME" ]; then
    echo "FATAL: no resume at $RESUME" >&2
    echo "Mount one:  -v \"\$PWD/resume.txt:/app/data/raw/resume.txt:ro\"" >&2
    exit 1
fi

echo "chatbot: resume    $RESUME ($(wc -c < "$RESUME") bytes)"

# The build report goes to `docker logs`, which is what you want to be able to
# read while waiting on a first-run ingest.
if ! python scripts/ingest.py; then
    echo "FATAL: ingest failed - see the error above." >&2
    echo "The resume must be the tagged format this project expects;" >&2
    echo "compare yours against data/raw/resume.txt inside the image." >&2
    exit 1
fi

if [ -z "$GROQ_API_KEY" ] && [ ! -f /run/secrets/groq_api_key ] \
   && [ -z "$GROQ_API_KEY_FILE" ]; then
    # Not fatal. The app renders its own "no key configured" page, which tells
    # somebody exactly what to do - far better than a container that exits
    # before it can show them anything.
    echo "chatbot: WARNING - no GROQ_API_KEY. Pass one with --env-file .env" >&2
fi

echo "chatbot: starting  $*"
exec "$@"
