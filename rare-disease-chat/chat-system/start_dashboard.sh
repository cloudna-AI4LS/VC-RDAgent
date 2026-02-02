#!/bin/bash
# Start the Professional Agent Dashboard server.
# Default port: 8080. Override with PORT=8082 ./start_dashboard.sh

echo "Starting VCAP-RDAgent Professional Dashboard..."
echo ""

# Model config: values written into inference_config.json before starting (edit below to change)
LLM_MODEL="Qwen/Qwen3-8B"
LLM_BASE_URL="http://192.168.0.127:8000/v1"
LLM_API_KEY="EMPTY"
LLM_MODEL_PROVIDER=""
LLM_TEMPERATURE="0.1"
LLM_TOP_P="0.95"
LLM_STREAMING="true"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_JSON="$SCRIPT_DIR/inference_config.json"
cd "$SCRIPT_DIR" || exit 1
if [ -n "$SCRIPT_DIR" ]; then
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
c["model_config"]["temperature"] = float(temp)
c["model_config"]["top_p"] = float(top_p)
c["model_config"]["streaming"] = stream.lower() in ("1", "true", "yes")
with open(p, "w", encoding="utf-8") as f:
    json.dump(c, f, indent=2, ensure_ascii=False)
' "$CONFIG_JSON" "$LLM_MODEL" "$LLM_MODEL_PROVIDER" "$LLM_BASE_URL" "$LLM_API_KEY" "$LLM_TEMPERATURE" "$LLM_TOP_P" "$LLM_STREAMING"
fi

echo "RDAgent will be available at: http://localhost:${PORT:-8080}/rdagent/"
echo ""

python3 rdagent_dashboard_api.py
