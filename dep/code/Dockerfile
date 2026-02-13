# 第一阶段：构建阶段
FROM python:3.12.12-slim as builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装依赖到虚拟环境
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# 第二阶段：运行阶段
FROM python:3.12.12-slim

WORKDIR /app 

# 安装运行时依赖（减少到最小）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 从构建阶段复制虚拟环境
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 复制应用代码
COPY . .

# 创建必要的目录
RUN mkdir -p data/vector_db models/cache static/img templates

# 创建非root用户（安全考虑）
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app   
USER appuser

# 环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "main.py"]