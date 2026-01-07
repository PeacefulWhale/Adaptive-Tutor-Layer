# Adaptive Tutor Layer (Django API)

Minimal Django backend for the V0 tutoring flow.

## Local dev (venv)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/manage.py makemigrations
.venv/bin/python src/manage.py migrate
.venv/bin/python src/manage.py runserver
```

Create an admin user:

```bash
.venv/bin/python src/manage.py createsuperuser
```

Add an active prompt (via the admin UI) by going to:

- http://127.0.0.1:8000/admin/

## Docker (SQLite persistence)

Build the image:

```bash
docker build -t adaptive-tutor-api .
```

Run with a bind mount so `db.sqlite3` persists on your host:

```bash
docker run -p 8000:8000 \
  -v "$(pwd)/src/db.sqlite3:/app/src/db.sqlite3" \
  -e LLM_API_URL="https://api.x.ai/v1/chat/completions" \
  -e LLM_API_KEY="your-key-here" \
  -e LLM_TIMEOUT_SECONDS="15" \
  -e LLM_DEFAULT_MODEL="grok-2-latest" \
  adaptive-tutor-api
```

Note: The Docker image runs `python src/manage.py migrate --noinput` automatically on startup.

## API

`POST /api/tutor/respond`

```json
{
  "user_id": "user-123",
  "conversation_id": "optional-uuid",
  "question_text": "What is photosynthesis?",
  "model": "optional",
  "temperature": 0.7,
  "max_tokens": 512
}
```
