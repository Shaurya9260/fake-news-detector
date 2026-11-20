FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/local/share/nltk_data \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check \
    && python -m nltk.downloader -q -e -d "$NLTK_DATA" stopwords wordnet

RUN useradd --create-home --uid 10001 appuser
COPY app.py database.py utils.py model.pkl vectorizer.pkl ./
COPY tests/ ./tests/
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=5s --timeout=4s --start-period=20s --retries=12 \
    CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()"]
CMD ["sh", "-c", "python -m database && exec streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true"]
