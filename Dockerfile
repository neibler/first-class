FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app
WORKDIR /app
RUN useradd --create-home --uid 10001 appuser

FROM base AS api
COPY backend/investment_ai/requirements.txt /app/requirements.txt
COPY backend/news_research/requirements.txt /app/news-requirements.txt
RUN python -m pip install -r requirements.txt -r news-requirements.txt
COPY backend/main.py /app/backend/main.py
COPY backend/investment_ai/ /app/backend/investment_ai/
COPY backend/news_research/ /app/backend/news_research/
COPY config/agent.yaml config/etf_news_map.yaml /app/config/
USER appuser
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS dashboard
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
COPY frontend/requirements.txt /app/requirements.txt
RUN python -m pip install -r requirements.txt
COPY frontend/ /app/frontend/
COPY .streamlit/config.toml /app/.streamlit/config.toml
USER appuser
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
