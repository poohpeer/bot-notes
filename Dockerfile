# Multi-stage: one Dockerfile, two targets (app, app-heavy) so shared code
# is never duplicated — see docs/architecture/06-deployment.md, "Образы".

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

# Without this, stdout/stderr are block-buffered in a container and a
# SIGTERM/SIGKILL (a failed liveness probe, an OOM kill) can drop everything
# written since the last flush — logs that would explain the crash never
# reach `kubectl logs` at all.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies before source: the layer cache survives every code change.
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev

COPY src ./src
COPY migrations ./migrations
COPY alembic.ini ./
# One-off maintenance scripts (docs/architecture/03-ingest.md, "Бэкфилл
# subjects") - not run by any CMD here, just baked in so they can be
# `kubectl exec`'d against a live pod without a separate deploy path.
COPY tools ./tools
RUN uv sync --no-dev --no-editable

# ---------------------------------------------------------------------------
# app — the bot and the fast worker. No ffmpeg, no yt-dlp, no Whisper: this
# image must start in seconds, not the better part of a minute.
# ---------------------------------------------------------------------------
FROM base AS app

CMD ["uv", "run", "--no-sync", "python", "-m", "notes_bot.cli.bot"]

# ---------------------------------------------------------------------------
# app-heavy — the heavy worker: yt-dlp downloads, ffmpeg audio extraction,
# faster-whisper transcription. Kept out of `app` so the bot's single replica
# doesn't pay for a stack it never touches.
# ---------------------------------------------------------------------------
FROM base AS app-heavy

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN uv sync --no-dev --extra heavy --no-editable

# faster-whisper's weights, baked in rather than left to download from
# Hugging Face on first use: observed live, a worker pod that had to fetch
# them cold ate enough of a 2-minute reel's transcription budget that the
# whole job hit RQ's job_timeout. `WHISPER_MODEL` matches
# Settings.whisper_model's own default ("small") so the image and a default
# deployment agree on which weights are worth pre-loading; an override still
# falls back to downloading on first use, same as before this existed.
#
# WHISPER_CACHE_BUST is never read for its value, only for the fact that it
# changed — ci.yml passes a fresh one on every build (the run id) so this
# layer never survives Docker's own cache and always re-asks Hugging Face
# for the current weights, the same way a plain `pip install` with no
# version pin would. huggingface_hub's own conditional GETs make a build
# where nothing changed upstream cheap; the point is never being able to
# silently drift behind because a layer happened to cache the same command.
ARG WHISPER_MODEL=small
ARG WHISPER_CACHE_BUST=0
RUN echo "cache-bust ${WHISPER_CACHE_BUST}" \
    && uv run --no-sync python -c \
        "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', compute_type='int8')"

CMD ["uv", "run", "--no-sync", "python", "-m", "notes_bot.cli.worker", "heavy"]
