# Multi-stage: one Dockerfile, two targets (app, app-heavy) so shared code
# is never duplicated — see docs/architecture/06-deployment.md, "Образы".

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

WORKDIR /app

# Dependencies before source: the layer cache survives every code change.
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev

COPY src ./src
COPY migrations ./migrations
COPY alembic.ini ./
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

CMD ["uv", "run", "--no-sync", "python", "-m", "notes_bot.cli.worker", "heavy"]
