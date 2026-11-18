# 多阶段：前端构建 → python 运行
FROM node:20-slim AS fe
WORKDIR /app/web/frontend
COPY web/frontend/package*.json ./
RUN npm install
COPY web/frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY scripts/ ./scripts/
COPY web/backend/ ./web/backend/
COPY templates/ ./templates/
COPY demo/ ./demo/
COPY --from=fe /app/web/frontend/dist/ ./web/frontend/dist/
ENV HOST=0.0.0.0 PORT=8601
EXPOSE 8601
CMD ["python", "-m", "uvicorn", "web.backend.app:app", "--host", "0.0.0.0", "--port", "8601"]
