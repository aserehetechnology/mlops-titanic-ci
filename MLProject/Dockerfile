FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir mlflow==2.19.0 scikit-learn pandas

COPY mlruns/ mlruns/

EXPOSE 5000

CMD ["sh", "-c", "RUN_ID=$(find mlruns -mindepth 2 -maxdepth 2 -type d -name '[a-f0-9]*' | head -1 | xargs basename) && EXP_ID=$(find mlruns -mindepth 2 -maxdepth 2 -type d -name '[a-f0-9]*' | head -1 | xargs dirname | xargs basename) && mlflow models serve -m mlruns/$EXP_ID/$RUN_ID/artifacts/model -h 0.0.0.0 -p 5000 --no-conda"]
