# Unified Dockerfile: MCP Server + Dashboard WebUI
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
    TERM=xterm \
    MCP_PORT=3000 \
    DASHBOARD_PORT=8080

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
    procps \
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

# Install MCP server dependencies
WORKDIR /app/mcp-server
RUN uv pip install --system --no-cache --index-url https://pypi.tuna.tsinghua.edu.cn/simple -e . || \
    (echo "Error: Failed to install MCP server dependencies" && exit 1)

# Verify MCP core dependencies
RUN python -c "import click; import mcp; print('MCP core dependencies installed')" && \
    (python -c "import sentence_transformers" 2>/dev/null || \
     uv pip install --system --no-cache --index-url https://pypi.tuna.tsinghua.edu.cn/simple 'sentence-transformers>=5.2.0')

# Install chat-system dependencies
WORKDIR /app/chat-system
RUN uv pip install --system --no-cache --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    'langchain>=1.2.3' \
    'langchain-core>=1.2.7' \
    'requests>=2.32.5' \
    'transformers>=4.57.3' \
    'langchain-openai>=1.1.7' \
    'fastapi>=0.128.0' \
    'uvicorn[standard]>=0.30.0' \
    'httpx[socks]>=0.27.0' \
    'xxhash>=3.6.0' || \
    (echo "Warning: Failed to install chat-system dependencies, continuing..." && true)

# Set default environment variables
ENV PROJECT_ROOT=/app \
    MCP_SIMPLE_TOOL_DIR=/app/mcp-server/mcp_simple_tool \
    MCP_ENDPOINT=http://localhost:3000/mcp/ \
    MCP_TIMEOUT=600

# Expose ports
EXPOSE 3000 8080

# Create startup script
WORKDIR /app
RUN cat > /app/start_all.sh << 'EOFSCRIPT'
#!/bin/bash

# Function to handle shutdown
cleanup() {
    echo ""
    echo "=== Shutting down services ==="
    [ -n "$MCP_PID" ] && kill $MCP_PID 2>/dev/null || true
    [ -n "$DASHBOARD_PID" ] && kill $DASHBOARD_PID 2>/dev/null || true
    sleep 2
    [ -n "$MCP_PID" ] && kill -9 $MCP_PID 2>/dev/null || true
    [ -n "$DASHBOARD_PID" ] && kill -9 $DASHBOARD_PID 2>/dev/null || true
    echo "All services stopped"
    exit 0
}

# Trap SIGTERM and SIGINT
trap cleanup SIGTERM SIGINT EXIT

# Set environment variables
export PROJECT_ROOT=/app
export MCP_SIMPLE_TOOL_DIR=/app/mcp-server/mcp_simple_tool
export MCP_ENDPOINT=http://localhost:${MCP_PORT:-3000}/mcp/
export PORT=${DASHBOARD_PORT:-8080}

# Start MCP server in background
echo "=== Starting MCP Server on port ${MCP_PORT:-3000} ==="
cd /app/mcp-server
python -m mcp_simple_tool.server --port ${MCP_PORT:-3000} --json-response > /tmp/mcp.log 2>&1 &
MCP_PID=$!
echo "MCP Server started (PID: $MCP_PID)"

# Wait for MCP server to be ready
echo "Waiting for MCP server to be ready..."
MCP_READY=0
for i in {1..30}; do
    if curl -s -X POST http://localhost:${MCP_PORT:-3000}/mcp/ \
        -H 'Content-Type: application/json' \
        -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' > /dev/null 2>&1; then
        echo "✓ MCP Server is ready"
        MCP_READY=1
        break
    fi
    if ! ps -p $MCP_PID > /dev/null 2>&1; then
        echo "✗ MCP Server process died (check logs: /tmp/mcp.log)"
        cat /tmp/mcp.log
        exit 1
    fi
    sleep 1
done

if [ $MCP_READY -eq 0 ]; then
    echo "✗ MCP Server failed to start within 30 seconds (check logs: /tmp/mcp.log)"
    cat /tmp/mcp.log
    exit 1
fi

# Start Dashboard WebUI
echo "=== Starting Dashboard WebUI on port ${DASHBOARD_PORT:-8080} ==="
cd /app/chat-system
python rdagent_dashboard_api.py > /tmp/dashboard.log 2>&1 &
DASHBOARD_PID=$!
echo "Dashboard started (PID: $DASHBOARD_PID)"

# Wait for dashboard to be ready
echo "Waiting for Dashboard to be ready..."
DASHBOARD_READY=0
for i in {1..30}; do
    if curl -s http://localhost:${DASHBOARD_PORT:-8080}/rdagent/ > /dev/null 2>&1; then
        echo "✓ Dashboard is ready"
        DASHBOARD_READY=1
        break
    fi
    if ! ps -p $DASHBOARD_PID > /dev/null 2>&1; then
        echo "✗ Dashboard process died (check logs: /tmp/dashboard.log)"
        cat /tmp/dashboard.log
        exit 1
    fi
    sleep 1
done

if [ $DASHBOARD_READY -eq 0 ]; then
    echo "✗ Dashboard failed to start within 30 seconds (check logs: /tmp/dashboard.log)"
    cat /tmp/dashboard.log
    exit 1
fi

# Services started successfully
echo ""
echo "=== All services started successfully ==="
echo "MCP Server: http://localhost:${MCP_PORT:-3000}"
echo "Dashboard: http://localhost:${DASHBOARD_PORT:-8080}/rdagent/"
echo ""
echo "MCP Server logs: tail -f /tmp/mcp.log"
echo "Dashboard logs: tail -f /tmp/dashboard.log"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for processes (keep container running)
while true; do
    if ! ps -p $MCP_PID > /dev/null 2>&1; then
        echo "MCP Server process died, shutting down..."
        cleanup
    fi
    if ! ps -p $DASHBOARD_PID > /dev/null 2>&1; then
        echo "Dashboard process died, shutting down..."
        cleanup
    fi
    sleep 5
done
EOFSCRIPT
RUN chmod +x /app/start_all.sh

# Default: start both services
CMD ["/app/start_all.sh"]
