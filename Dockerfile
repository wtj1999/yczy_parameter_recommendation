# ===== Dockerfile =====
FROM python:3.11-slim

# 切回 root 并设置工作目录
USER root
WORKDIR /app

# 1) 使用国内 apt 镜像（直接覆盖 sources.list，适配 slim 镜像没有原文件的情况）
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian bookworm main contrib non-free" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian bookworm-updates main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main contrib non-free" >> /etc/apt/sources.list

# 2) 安装系统依赖（catboost 编译依赖）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    gcc \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# 4) 配置 pip 使用国内镜像并安装依赖
COPY requirements.txt /app/requirements.txt
RUN mkdir -p /etc && \
    printf '[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple\ntrusted-host = pypi.tuna.tsinghua.edu.cn\ntimeout = 120\n' > /etc/pip.conf
RUN pip install --no-cache-dir -r /app/requirements.txt --timeout 120 --retries 6

# 5) 复制应用代码
COPY . /app

# 6) 可选：把模型打包到镜像中（如果你想包含模型）
# COPY models/catboost_model.cbm /app/models/catboost_model.cbm

# 7) 端口
EXPOSE 8000

# 8) 非 root 运行（安全性）
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser

# 9) 启动命令
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "120"]
