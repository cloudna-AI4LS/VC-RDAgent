#!/bin/bash
# Write LLM/model_config into MCP and/or chat-system config files.
# Edit the variables below (MCP and chat-system can use different values), then run: ./set_config.sh
# Run from rare-disease-chat/ (this directory).
# To only update one side: run with MCP_ONLY=1 or CHAT_ONLY=1 (e.g. MCP_ONLY=1 ./set_config.sh).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# -----------------------------------------------------------------------------
# MCP server config (prompt_config_forKG.json)
# -----------------------------------------------------------------------------
MCP_LLM_MODEL="Qwen/Qwen3-8B"
MCP_LLM_BASE_URL="http://192.168.0.127:8000/v1"
MCP_LLM_API_KEY="EMPTY"
MCP_LLM_MODEL_PROVIDER=""
MCP_LLM_TEMPERATURE="0.1"
MCP_LLM_TOP_P="0.95"
MCP_LLM_STREAMING="true"

# -----------------------------------------------------------------------------
# Chat-system config (inference_config.json)
# -----------------------------------------------------------------------------
CHAT_LLM_MODEL="Qwen/Qwen3-8B"
CHAT_LLM_BASE_URL="http://192.168.0.127:8000/v1"
CHAT_LLM_API_KEY="EMPTY"
CHAT_LLM_MODEL_PROVIDER=""
CHAT_LLM_TEMPERATURE="0.1"
CHAT_LLM_TOP_P="0.95"
CHAT_LLM_STREAMING="true"

# -----------------------------------------------------------------------------
# Paths to the two config files
# -----------------------------------------------------------------------------
MCP_CONFIG="$SCRIPT_DIR/mcp-server/mcp_simple_tool/scripts/rare_disease_diagnose/prompt_config_forKG.json"
CHAT_CONFIG="$SCRIPT_DIR/chat-system/inference_config.json"

_write_mcp_config() {
    python3 -c '
import json, sys
p = sys.argv[1]
model, provider, base_url, api_key, temp, top_p, stream, kg_url = sys.argv[2:10]
with open(p, "r", encoding="utf-8") as f:
    c = json.load(f)
c.setdefault("model_config", {})["model"] = model
c["model_config"]["model_provider"] = provider
c["model_config"]["base_url"] = base_url
c["model_config"]["api_key"] = api_key
c["model_config"]["temperature"] = float(temp) if temp else 0.1
c["model_config"]["top_p"] = float(top_p) if top_p else 0.95
c["model_config"]["streaming"] = stream.lower() in ("1", "true", "yes")
c["kg_api_url"] = kg_url
with open(p, "w", encoding="utf-8") as f:
    json.dump(c, f, indent=2, ensure_ascii=False)
' "$MCP_CONFIG" "$MCP_LLM_MODEL" "$MCP_LLM_MODEL_PROVIDER" "$MCP_LLM_BASE_URL" "$MCP_LLM_API_KEY" "$MCP_LLM_TEMPERATURE" "$MCP_LLM_TOP_P" "$MCP_LLM_STREAMING" "$MCP_KG_API_URL"
}

_write_chat_config() {
    python3 -c '
import json, sys
p = sys.argv[1]
model, provider, base_url, api_key, temp, top_p, stream = sys.argv[2:9]
try:
    with open(p, "r", encoding="utf-8") as f:
        c = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    c = {}
c.setdefault("model_config", {})["model"] = model
c["model_config"]["model_provider"] = provider
c["model_config"]["base_url"] = base_url
c["model_config"]["api_key"] = api_key
c["model_config"]["temperature"] = float(temp) if temp else 0.1
c["model_config"]["top_p"] = float(top_p) if top_p else 0.95
c["model_config"]["streaming"] = stream.lower() in ("1", "true", "yes")
with open(p, "w", encoding="utf-8") as f:
    json.dump(c, f, indent=2, ensure_ascii=False)
' "$CHAT_CONFIG" "$CHAT_LLM_MODEL" "$CHAT_LLM_MODEL_PROVIDER" "$CHAT_LLM_BASE_URL" "$CHAT_LLM_API_KEY" "$CHAT_LLM_TEMPERATURE" "$CHAT_LLM_TOP_P" "$CHAT_LLM_STREAMING"
}

echo "Writing config..."
if [ "${CHAT_ONLY:-0}" = "1" ]; then
    _write_chat_config
    echo "  OK: $CHAT_CONFIG"
elif [ "${MCP_ONLY:-0}" = "1" ]; then
    if [ -f "$MCP_CONFIG" ]; then
        _write_mcp_config
        echo "  OK: $MCP_CONFIG"
    else
        echo "  Skip (not found): $MCP_CONFIG"
    fi
else
    if [ -f "$MCP_CONFIG" ]; then
        _write_mcp_config
        echo "  OK: $MCP_CONFIG"
    else
        echo "  Skip (not found): $MCP_CONFIG"
    fi
    _write_chat_config
    echo "  OK: $CHAT_CONFIG"
fi
echo "Done. Run mcp-server/start_server.sh and/or chat-system/start_dashboard.sh next."
