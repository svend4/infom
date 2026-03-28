FROM python:3.11-slim

WORKDIR /app

# Зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn

# Код
COPY . .

# Порт
EXPOSE 8000

# Запуск REST API
CMD ["python", "infom_api.py"]

# Сборка и запуск локально:
#   docker build -t infom-api .
#   docker run -p 8000:8000 \
#     -e INFOM_PROVIDER=groq \
#     -e GROQ_API_KEY=gsk-... \
#     -e JINA_API_KEY=... \
#     infom-api
