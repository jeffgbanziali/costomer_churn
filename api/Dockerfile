FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Source partagée (preprocessing, evaluation)
COPY src/ /app/src/

# Code de l'API
COPY api/ /app/api/

# models/ est monté en volume depuis docker-compose
# Pour test standalone : décommenter la ligne suivante
# COPY models/ /app/models/

EXPOSE 8000

# WORKDIR dans api/ pour que "from auth import" et "from schemas import" fonctionnent
WORKDIR /app/api

HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
