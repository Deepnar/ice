```
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload 
```

```
vllm-bg
```

```
uv run celery -A src.workers.celery_app worker --loglevel=info  
```

```
cd docker
docker compose down
docker compose up -d
cd ..
```
