FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp \
    TMPDIR=/tmp

WORKDIR /app

COPY requirements.txt /app/requirements.txt
# --no-deps: openjiuwen's optional extras (transformers, vector DBs, ...) are
# never imported by this harness; skipping them keeps the image far below the
# 4GB limit and the build under 1000s. httpx is installed explicitly below.
RUN pip install --no-cache-dir --no-deps openjiuwen==0.1.19 \
    && pip install --no-cache-dir httpx==0.28.1

COPY harness /app/harness
COPY submission /app/submission
COPY run_agent.py /app/run_agent.py

RUN useradd --create-home --home-dir /tmp --uid 10001 --shell /usr/sbin/nologin agentuser \
    && chown -R agentuser:agentuser /app

USER agentuser

ENTRYPOINT ["python"]
CMD ["/app/run_agent.py", "--tasks-dir", "/data/tasks"]
