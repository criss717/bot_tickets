FROM python:3.12-slim

ARG APP_UID=1000
ARG APP_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Europe/Madrid

WORKDIR /app

# Configure Spain timezone so receipt dates and monthly closures align locally
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged runtime user (non-root)
RUN groupadd --gid ${APP_GID} appgroup \
    && useradd --uid ${APP_UID} --gid ${APP_GID} --shell /usr/sbin/nologin --create-home appuser

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and prepare persistent mount directories owned by appuser
COPY --chown=appuser:appgroup src/ ./src/
RUN mkdir -p /app/data/images /app/export/fotos && chown -R appuser:appgroup /app

USER appuser

CMD ["python", "src/main.py"]
