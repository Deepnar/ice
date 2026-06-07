# 1. Docker services
cd docker && docker compose up -d && cd ..

# 2. Background model (wait for "Application startup complete")
vllm-bg

# 3. Celery worker (wait for "celery@orien ready.")
uv run celery -A src.workers.celery_app worker --loglevel=info

# 4. (optional) Proxy if you want to also test Open WebUI
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload