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
# Investment inference explicitly loads on CPU; avoid unused CUDA packages.
RUN python -m pip install 'torch>=2.3,<3' --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements.txt -r news-requirements.txt
COPY backend/main.py /app/backend/main.py
COPY backend/research_api.py backend/research_schemas.py backend/research_service.py /app/backend/
COPY backend/investment_ai/ /app/backend/investment_ai/
COPY backend/news_research/ /app/backend/news_research/
# 결과 JSON 을 읽어 내보내는 /performance 라우터. 분석 실행용 무거운 의존성
# (scipy/statsmodels/shap/matplotlib)은 설치하지 않는다 — 분석은 컨테이너 밖에서
# 돌리고 results/ 를 RESULTS_DIR 로 마운트한다.
COPY backend/performance/ /app/backend/performance/
COPY shared/trained_models_manifest.json /app/shared/trained_models_manifest.json
COPY config/agent.yaml config/etf_news_map.yaml config/serving.json config/investment_ai.json config/multi_model.json /app/config/
ENV RESULTS_DIR=/results
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
