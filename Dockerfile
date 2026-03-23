# ----------------------------
# Base: Python + OS packages
# ----------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # keep numerical libs predictable across CPUs
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

# system deps (build tools, git, and basics for manylinux wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ----------------------------
# Builder: install deps
# ----------------------------
FROM base AS builder

# Copy only the dependency manifests first to leverage Docker layer caching
COPY pyproject.toml README.md ./
# If you keep benchmark YAMLs inside package data, copy those too:
COPY ofb/ofb/__init__.py ofb/__init__.py 2>/dev/null || true

# Create a virtualenv to hold all deps (and project) for easy copying
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ARG to include optional extras (e.g., ".[opendss]" or ".[dev,opendss]")
# Tip: Use as `--build-arg EXTRAS=[opendss]`
ARG EXTRAS=
# Install project in editable mode later; first install bare deps so cache hits
RUN pip install --upgrade pip wheel setuptools

# Preinstall project dependencies without sources to maximize cache
# If you pin dependencies, consider using uv/poetry export; here we rely on PEP 621
# We'll install the actual package after copying the source in the next stage.

# ----------------------------
# Dev image (editable install)
# ----------------------------
FROM builder AS dev

# Bring in the rest of the repo
COPY . /app

# Install with extras if provided (e.g., EXTRAS=[dev,opendss])
ARG EXTRAS
RUN pip install -e .${EXTRAS:+$EXTRAS}

# Nice default: show CLI help if someone runs the container without args
ENTRYPOINT ["ofb"]
CMD ["--help"]

# Usage:
# docker build -t ofb-dev --target dev --build-arg EXTRAS=[dev,opendss] .
# docker run --rm -it -v "$PWD:/app" ofb-dev run-benchmark benchmarks/configs/baseline_freq.yaml

# ----------------------------
# Runtime image (slim, non-root)
# ----------------------------
FROM base AS runtime

# Copy the virtualenv from builder (if you preinstalled anything there)
# But we'll install directly here for clarity
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy minimal project files needed to install (source + metadata)
COPY pyproject.toml README.md /app/
COPY ofb /app/ofb

# Build arg to include optional runtime extras (e.g., [opendss])
ARG EXTRAS
RUN pip install --upgrade pip wheel setuptools \
 && pip install "/app"${EXTRAS:+$EXTRAS}

# Create a non-root user
RUN useradd -ms /bin/bash runner
USER runner

WORKDIR /work

# Default entrypoint: CLI available as `ofb`
ENTRYPOINT ["ofb"]
CMD ["--help"]

# Usage:
# docker build -t ofb:latest --target runtime --build-arg EXTRAS=[opendss] .
# docker run --rm -it -v "$PWD:/work" ofb:latest run-benchmark benchmarks/configs/baseline_freq.yaml
