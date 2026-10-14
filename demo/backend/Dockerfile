FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/opt/accelerated_features:/app \
    PYTHONUNBUFFERED=1 \
    PIP_TIMEOUT=300 \
    PIP_RETRIES=10

# XFeat (verlab/accelerated_features) — репозиторий без setup.py, кладём на PYTHONPATH.
# Веса (~6 МБ) лежат в weights/ внутри репозитория.
ADD https://github.com/verlab/accelerated_features/archive/refs/heads/main.tar.gz /tmp/xfeat.tar.gz
RUN mkdir -p /opt/accelerated_features \
    && tar -xzf /tmp/xfeat.tar.gz -C /opt/accelerated_features --strip-components=1 \
    && rm /tmp/xfeat.tar.gz

# torch только для CPU (~200 МБ вместо ~2.5 ГБ с CUDA) — Mask2Former и XFeat считаются на процессоре
RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    torch

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
