# ----------------------------
# Base: Python + OS packages
# ----------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

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

COPY pyproject.toml README.md ./
COPY src/ src/

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ARG EXTRAS=
RUN pip install --upgrade pip wheel setuptools

# ----------------------------
# Dev image (editable install)
# ----------------------------
FROM builder AS dev

COPY . /app

ARG EXTRAS
RUN pip install -e ".${EXTRAS:+[$EXTRAS]}"

ENTRYPOINT ["ofb"]
CMD ["--help"]

# Usage:
# docker build -t ofb-dev --target dev --build-arg EXTRAS=dev .
# docker run --rm -it -v "$PWD:/app" ofb-dev run examples/quick_smoke.yaml

# ----------------------------
# Runtime image (slim, non-root)
# ----------------------------
FROM base AS runtime

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md /app/
COPY src/ /app/src/

ARG EXTRAS
RUN pip install --upgrade pip wheel setuptools \
 && pip install "/app${EXTRAS:+[$EXTRAS]}"

RUN useradd -ms /bin/bash runner
USER runner

WORKDIR /work

ENTRYPOINT ["ofb"]
CMD ["--help"]

# Usage:
# docker build -t ofb:latest --target runtime .
# docker run --rm -it -v "$PWD:/work" ofb:latest run examples/quick_smoke.yaml
