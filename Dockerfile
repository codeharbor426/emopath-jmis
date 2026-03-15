FROM python:3.11-slim

WORKDIR /app

# system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# copy dependency file first (better Docker caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# copy project files
COPY . .

ENV PYTHONPATH=/app

CMD ["bash"]