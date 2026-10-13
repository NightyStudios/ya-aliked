FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PIP_TIMEOUT=300 \
    PIP_RETRIES=10

# torch только для CPU (~200 МБ вместо ~2.5 ГБ с CUDA) — Mask2Former считается на процессоре
RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    torch

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
