FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pandas folium numpy

COPY . .

CMD ["sh", "-c", "rm -f last_run.json && python pull_data.py --use-dump && python generate_map.py"]
