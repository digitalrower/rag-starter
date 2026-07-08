FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ ./src/
COPY data/ ./data/
RUN pip install --no-cache-dir --no-deps .

RUN useradd --create-home appuser \
    && mkdir -p /app/chroma_db \
    && chown -R appuser:appuser /app
USER appuser
RUN python -c "import chromadb.utils.embedding_functions as ef; ef.DefaultEmbeddingFunction()(['warmup'])"

CMD ["python", "-m", "rag_starter.query", "What are Claude's models?"]