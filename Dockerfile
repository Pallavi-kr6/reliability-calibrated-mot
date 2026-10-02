# CPU image. Build:  docker build -t rcamot .      Run the demo:  docker run --rm -v "$PWD/results:/app/results" rcamot
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY . .
ENV PYTHONUNBUFFERED=1
# Default: weights-free synthetic demo (no datasets, no GPU). Override for real runs, e.g.
#   docker run --rm -v /path/to/data:/app/data rcamot python scripts/reproduce.py --full --embedder colorhist
CMD ["python", "scripts/reproduce.py", "--quick"]
