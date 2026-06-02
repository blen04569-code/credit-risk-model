# Use a lightweight official Python production image footprint
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging logs stream
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system utilities needed for package builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first to make optimal use of Docker layer caching rules
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy structural codebase elements into directory environment mapping
COPY src/ /app/src/

EXPOSE 8000

# Fire up our enterprise web listener engine bound to standard multi-routing ports
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]