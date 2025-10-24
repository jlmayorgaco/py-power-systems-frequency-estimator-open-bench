# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base

# ---- Reproducibility + deterministic BLAS threading ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Thread caps to avoid run-to-run variability
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# Minimal OS deps (tini = proper PID 1)
RUN apt-get update -y \
 && apt-get install -y --no-install-recommends tini ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# ---- Non-root user ----
ARG UID=10001
ARG GID=10001
RUN groupadd -g "${GID}" app \
 && useradd -m -u "${UID}" -g app appuser

WORKDIR /app

# Copy only metadata first for better build cache on code changes
COPY pyproject.toml README.md LICENSE ./
# If you have requirements*.txt, uncomment and copy them too for better caching
# COPY requirements*.txt ./

# Install project; prefer full extras if available, fallback to base
RUN python -m pip install --upgrade pip \
 && (python -m pip install -e ".[full]" || python -m pip install -e .)

# Now copy the rest of the source (keeps cache when deps unchanged)
COPY . /app

# Entrypoint + default command
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER appuser

# tini as PID 1, then your entrypoint
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]

# Default: show CLI help (adjust module name if needed, see note below)
CMD ["python", "-m", "pyopenfreqbench", "--help"]
