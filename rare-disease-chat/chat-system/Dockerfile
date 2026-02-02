# Chat system standalone image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    UV_SYSTEM_PYTHON=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    DEBIAN_FRONTEND=noninteractive \
    TERM=xterm

# Configure apt mirror (Aliyun for faster downloads)
RUN if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's|http://deb.debian.org|http://mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources && \
        sed -i 's|https://deb.debian.org|http://mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources; \
    elif [ -f /etc/apt/sources.list ]; then \
        sed -i 's|http://deb.debian.org|http://mirrors.aliyun.com|g' /etc/apt/sources.list && \
        sed -i 's|https://deb.debian.org|http://mirrors.aliyun.com|g' /etc/apt/sources.list; \
    fi

# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install uv (Tsinghua mirror for faster downloads)
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple uv

# Copy project files
COPY . .

# Install project dependencies (Tsinghua mirror for faster downloads)
RUN uv pip install --system --no-cache --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    'langchain>=1.2.3' \
    'langchain-core>=1.2.7' \
    'requests>=2.32.5' \
    'transformers>=4.57.3' \
    'langchain-openai>=1.1.7' || \
    (echo "Warning: Failed to install dependencies, continuing..." && true)

# Set default environment variables
ENV PROJECT_ROOT=/app \
    MCP_ENDPOINT=http://mcp-server:3000/mcp/ \
    MCP_TIMEOUT=600

# Default: start chat system
CMD ["python", "phenotype_to_disease_controller_langchain_stream_api.py"]
