FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
VOLUME ["/app/src/db.sqlite3"]

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "src/manage.py", "runserver", "0.0.0.0:8000"]
