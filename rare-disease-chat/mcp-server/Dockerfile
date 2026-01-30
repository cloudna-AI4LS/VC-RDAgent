# MCP server standalone image
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


# Install CPU-only torch (no CUDA, smaller image)
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
        -f https://mirrors.aliyun.com/pytorch-wheels/cpu \
        'torch>=2.9.1' || \
    pip install --no-cache-dir \
        -f https://download.pytorch.org/whl/cpu \
        'torch>=2.9.1'

# Install MCP server dependencies (torch already installed above)
RUN uv pip install --system --no-cache --index-url https://pypi.tuna.tsinghua.edu.cn/simple -e . || \
    (echo "Error: Failed to install dependencies" && exit 1)
# Verify core dependencies
RUN python -c "import click; import mcp; print('Core dependencies installed')" && \
    (python -c "import sentence_transformers" 2>/dev/null || \
     uv pip install --system --no-cache --index-url https://pypi.tuna.tsinghua.edu.cn/simple 'sentence-transformers>=5.2.0')

# Set default environment variables
ENV PROJECT_ROOT=/app \
    MCP_SIMPLE_TOOL_DIR=/app/mcp_simple_tool \
    PORT=3000

# Expose port
EXPOSE 3000

# Default: start MCP server (run via python -m so modules are found)
CMD ["python", "-m", "mcp_simple_tool.server", "--port", "3000", "--json-response"]
