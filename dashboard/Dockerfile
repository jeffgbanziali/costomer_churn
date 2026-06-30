FROM python:3.11-slim

RUN pip install --no-cache-dir \
    streamlit>=1.35.0 \
    requests>=2.31.0 \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    plotly>=5.15.0 \
    joblib \
    scikit-learn

WORKDIR /app

COPY dashboard/ /app/dashboard/

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]