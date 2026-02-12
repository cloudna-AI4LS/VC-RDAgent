#!/bin/bash
# One-click start: MCP server (background) + Web UI dashboard (foreground).
# Run from rare-disease-chat/ (this directory).
# Prerequisite: run ./set_config.sh once; install deps in mcp-server/ and chat-system/ (see README).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -z "${PROJECT_ROOT:-}" ]; then
    export PROJECT_ROOT="$SCRIPT_DIR"
fi
if [ -z "${MCP_SIMPLE_TOOL_DIR:-}" ]; then
    export MCP_SIMPLE_TOOL_DIR="$PROJECT_ROOT/mcp-server/mcp_simple_tool"
fi

echo "=== One-click start (MCP server + Web UI) ==="
echo ""

if [ ! -d "$SCRIPT_DIR/mcp-server/.venv" ] || [ ! -d "$SCRIPT_DIR/chat-system/.venv" ]; then
    echo "Error: .venv not found. Run Installation in README (uv venv + uv pip install -e . in mcp-server/ and chat-system/)."
    exit 1
fi

# 1. Stop MCP server if running
echo "Stopping MCP server if running..."
"$SCRIPT_DIR/mcp-server/stop_server.sh" || true
echo ""

# 2. Start MCP server in background (prepend mcp-server venv to PATH so python3 finds mcp)
echo "Starting MCP server (background)..."
PATH="$SCRIPT_DIR/mcp-server/.venv/bin:$PATH" "$SCRIPT_DIR/mcp-server/start_server.sh"
echo ""

# 3. Wait for MCP server to be ready
echo "Waiting for MCP server to be ready..."
sleep 3
echo ""

# 4. Start Web UI dashboard (foreground; prepend chat-system venv to PATH)
echo "Starting Web UI dashboard (foreground)..."
echo "Open: http://localhost:${PORT:-8080}/rdagent/"
echo "Press Ctrl+C to stop the dashboard (MCP server keeps running; stop with mcp-server/stop_server.sh)."
echo ""

PATH="$SCRIPT_DIR/chat-system/.venv/bin:$PATH" exec "$SCRIPT_DIR/chat-system/start_dashboard.sh"
