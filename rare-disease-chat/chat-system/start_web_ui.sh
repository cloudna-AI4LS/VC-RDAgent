#!/bin/bash
# Start Web UI service

echo "🚀 Starting Phenotype-to-Disease Web UI..."
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

# Check Python environment
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    exit 1
fi

# Check required Python packages (uses same Python as the shell that ran this script)
echo "📦 Checking dependencies..."
python3 -c "import fastapi" 2>/dev/null || {
    echo "❌ Error: fastapi not found in current environment. Install deps first, e.g. uv pip install -e ."
    exit 1
}

# Start service
echo ""
echo "🌐 Starting Web server..."
echo "📍 Local access: http://localhost:8080"
echo ""

python3 web_ui_api.py
