#!/bin/bash
# -*- coding: utf-8 -*-
# Start API server script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Auto-set project root and MCP tool path (when unset, same as README env vars)
if [ -z "${PROJECT_ROOT:-}" ]; then
    export PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
if [ -z "${MCP_SIMPLE_TOOL_DIR:-}" ]; then
    export MCP_SIMPLE_TOOL_DIR="$PROJECT_ROOT/mcp-server/mcp_simple_tool"
fi

# Configuration
PORT="${PORT:-3000}"
LOG_FILE="${LOG_FILE:-api_server.log}"
PID_FILE="${PID_FILE:-api_server.pid}"

# Model config: values written into prompt_config_forKG.json before starting (edit below to change)
LLM_MODEL="Qwen/Qwen3-8B"
LLM_BASE_URL="http://192.168.0.127:8000/v1"
LLM_API_KEY="EMPTY"
LLM_MODEL_PROVIDER=""
LLM_TEMPERATURE="0.1"
LLM_TOP_P="0.95"
LLM_STREAMING="true"

CONFIG_JSON="$MCP_SIMPLE_TOOL_DIR/scripts/rare_disease_diagnose/prompt_config_forKG.json"
if [ -f "$CONFIG_JSON" ]; then
    python3 -c '
import json, sys
p = sys.argv[1]
model, provider, base_url, api_key, temp, top_p, stream = sys.argv[2:9]
with open(p, "r", encoding="utf-8") as f:
    c = json.load(f)
c.setdefault("model_config", {})["model"] = model
c["model_config"]["model_provider"] = provider
c["model_config"]["base_url"] = base_url
c["model_config"]["api_key"] = api_key
c["model_config"]["temperature"] = float(temp)
c["model_config"]["top_p"] = float(top_p)
c["model_config"]["streaming"] = stream.lower() in ("1", "true", "yes")
with open(p, "w", encoding="utf-8") as f:
    json.dump(c, f, indent=2, ensure_ascii=False)
' "$CONFIG_JSON" "$LLM_MODEL" "$LLM_MODEL_PROVIDER" "$LLM_BASE_URL" "$LLM_API_KEY" "$LLM_TEMPERATURE" "$LLM_TOP_P" "$LLM_STREAMING"
fi

# Check Python environment
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Check if port is in use
check_port() {
    local port=$1
    if command -v lsof &> /dev/null; then
        local pid=$(lsof -ti :$port 2>/dev/null)
        if [ -n "$pid" ]; then
            echo "$pid"
            return 0
        fi
    elif command -v netstat &> /dev/null; then
        local pid=$(netstat -tuln 2>/dev/null | grep ":$port " | awk '{print $NF}' | cut -d'/' -f1 | head -1)
        if [ -n "$pid" ]; then
            echo "$pid"
            return 0
        fi
    elif command -v ss &> /dev/null; then
        local pid=$(ss -tuln 2>/dev/null | grep ":$port " | awk '{print $NF}' | cut -d'/' -f1 | head -1)
        if [ -n "$pid" ]; then
            echo "$pid"
            return 0
        fi
    fi
    return 1
}

# Check if already running (via PID file)
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "Server already running (PID: $OLD_PID)"
        echo "To restart, run first: ./stop_server.sh"
        exit 1
    else
        # PID file exists but process is gone, remove PID file
        rm -f "$PID_FILE"
    fi
fi

# Check if port is in use
PORT_PID=$(check_port "$PORT")
if [ -n "$PORT_PID" ]; then
    echo "Warning: port $PORT is in use by process $PORT_PID"
    echo "Process info:"
    ps -p "$PORT_PID" -o pid,cmd,etime 2>/dev/null || echo "Cannot get process info"
    
    # Check if same server process
    PORT_CMD=$(ps -p "$PORT_PID" -o cmd= 2>/dev/null)
    if echo "$PORT_CMD" | grep -q "mcp_simple_tool/server.py"; then
        echo ""
        echo "Same server process detected, stopping and restarting..."
        kill "$PORT_PID" 2>/dev/null
        sleep 2
        if ps -p "$PORT_PID" > /dev/null 2>&1; then
            echo "Force stopping process..."
            kill -9 "$PORT_PID" 2>/dev/null
            sleep 1
        fi
        if ps -p "$PORT_PID" > /dev/null 2>&1; then
            echo "Error: cannot stop process $PORT_PID"
            exit 1
        else
            echo "✓ Old process stopped"
        fi
    else
        # Non-interactive: if AUTO_KILL_PORT is set, auto-stop
        if [ "${AUTO_KILL_PORT:-}" = "1" ]; then
            echo "Auto-stopping process using port..."
            kill "$PORT_PID" 2>/dev/null
            sleep 2
            if ps -p "$PORT_PID" > /dev/null 2>&1; then
                kill -9 "$PORT_PID" 2>/dev/null
                sleep 1
            fi
            if ! ps -p "$PORT_PID" > /dev/null 2>&1; then
                echo "✓ Process stopped"
            else
                echo "Error: cannot stop process $PORT_PID"
                exit 1
            fi
        else
            # Interactive mode
            echo ""
            echo "Please choose:"
            echo "1. Stop process using port and continue"
            echo "2. Use another port (set PORT env var)"
            echo "3. Exit"
            echo ""
            read -p "Enter choice (1/2/3): " choice
            
            case "$choice" in
                1)
                    echo "Stopping process $PORT_PID..."
                    kill "$PORT_PID" 2>/dev/null
                    sleep 2
                    if ps -p "$PORT_PID" > /dev/null 2>&1; then
                        echo "Force stopping process..."
                        kill -9 "$PORT_PID" 2>/dev/null
                        sleep 1
                    fi
                    if ps -p "$PORT_PID" > /dev/null 2>&1; then
                        echo "Error: cannot stop process $PORT_PID"
                        exit 1
                    else
                        echo "✓ Process stopped"
                    fi
                    ;;
                2)
                    read -p "Enter new port number: " new_port
                    PORT="$new_port"
                    echo "Using port: $PORT"
                    ;;
                3)
                    echo "Exiting"
                    exit 0
                    ;;
                *)
                    echo "Invalid choice, exiting"
                    exit 1
                    ;;
            esac
        fi
    fi
fi

# Start server
echo "Starting MCP API server..."
echo "Port: $PORT"
echo "Log file: $LOG_FILE"
echo "PID file: $PID_FILE"
echo "Server file: mcp_simple_tool/server.py"
echo ""

# Run in background with nohup
nohup python3 "$SCRIPT_DIR/mcp_simple_tool/server.py" \
    --port "$PORT" \
    --log-level INFO \
    --json-response \
    > "$LOG_FILE" 2>&1 &

# Save PID
SERVER_PID=$!
echo $SERVER_PID > "$PID_FILE"

# Wait and check if started successfully
sleep 2

if ps -p "$SERVER_PID" > /dev/null 2>&1; then
    echo "✓ Server started successfully!"
    echo "  PID: $SERVER_PID"
    echo "  Access: http://0.0.0.0:$PORT"
    echo "  View log: tail -f $LOG_FILE"
    echo "  Stop: ./stop_server.sh"
else
    echo "✗ Server failed to start, check log: $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi
