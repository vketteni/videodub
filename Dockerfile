# Multi-stage build for optimal size and security
FROM python:3.11-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.7.1

# Set working directory for Poetry
WORKDIR /app

# Copy poetry files
COPY pyproject.toml poetry.lock* ./

# Configure poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Export dependencies to requirements.txt and install with pip
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes && \
    pip install -r requirements.txt && \
    rm -rf $POETRY_CACHE_DIR

# Production stage
FROM python:3.11-slim

# Install system dependencies (external deps)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install yt-dlp
RUN pip install yt-dlp

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder --chown=app:app /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder --chown=app:app /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=app:app . .

# Create output directory (before switching user)
RUN mkdir -p /app/output && chown app:app /app/output

# Switch to non-root user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import videodub; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "videodub", "--help"]