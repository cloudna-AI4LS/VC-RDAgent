#!/bin/bash
# -*- coding: utf-8 -*-
# Stop API server script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${PID_FILE:-api_server.pid}"
PORT="${PORT:-3000}"

# Function to check port usage
check_port() {
    local port=$1
    if command -v lsof &> /dev/null; then
        lsof -ti :$port 2>/dev/null
    elif command -v netstat &> /dev/null; then
        netstat -tuln 2>/dev/null | grep ":$port " | awk '{print $NF}' | cut -d'/' -f1 | head -1
    elif command -v ss &> /dev/null; then
        ss -tuln 2>/dev/null | grep ":$port " | awk '{print $NF}' | cut -d'/' -f1 | head -1
    fi
}

# Function to stop process
stop_process() {
    local pid=$1
    if [ -z "$pid" ] || ! ps -p "$pid" > /dev/null 2>&1; then
        return 1
    fi
    
    echo "Stopping process $pid..."
    kill "$pid" 2>/dev/null
    
    # Wait for process to exit
    for i in {1..10}; do
        if ! ps -p "$pid" > /dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    
    # If still running, force kill
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "Force stopping process $pid..."
        kill -9 "$pid" 2>/dev/null
        sleep 1
    fi
    
    return 0
}

# Method 1: Stop via PID file
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        stop_process "$PID"
        rm -f "$PID_FILE"
        echo "✓ Server stopped via PID file"
    else
        echo "Process $PID in PID file does not exist, cleaning PID file"
        rm -f "$PID_FILE"
    fi
fi

# Method 2: Stop via port check (if PID file method failed)
PORT_PID=$(check_port "$PORT")
if [ -n "$PORT_PID" ]; then
    PORT_CMD=$(ps -p "$PORT_PID" -o cmd= 2>/dev/null)
    if echo "$PORT_CMD" | grep -q "mcp_simple_tool/server.py"; then
        stop_process "$PORT_PID"
        echo "✓ Server stopped via port check (PID: $PORT_PID)"
    fi
fi

echo "Server stopped"
