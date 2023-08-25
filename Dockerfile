FROM python:3.11-slim AS builder
WORKDIR /app
# Create a virtual environment and install the last version of pip and setuptools 
RUN python -m venv .venv && .venv/bin/pip install --no-cache-dir -U pip setuptools
COPY requirements-worker.txt *.whl ./
RUN .venv/bin/pip install --no-cache-dir $( ( find . -type f -name "*.whl" | grep . ) || echo armonik ) -r requirements-worker.txt && find /app/.venv \( -type d -a -name test -o -name tests \) -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) -exec rm -rf '{}' \+

FROM python:3.11-slim
WORKDIR /app
RUN groupadd --gid 5000 armonikuser && useradd --home-dir /home/armonikuser --create-home --uid 5000 --gid 5000 --shell /bin/sh --skel /dev/null armonikuser && mkdir /cache && chown armonikuser: /cache
USER armonikuser
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 WORKER=True ENV=dev
COPY --from=builder /app /app
COPY ./linpyk/common ./linpyk/common
COPY ./linpyk/worker.py ./linpyk/__main__.py
ENTRYPOINT ["python", "-m", "linpyk"]
