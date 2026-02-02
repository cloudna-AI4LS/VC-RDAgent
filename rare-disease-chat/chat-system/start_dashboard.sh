#!/bin/bash
# Start the Professional Agent Dashboard server.
# Default port: 8080. Override with PORT=8082 ./start_dashboard.sh
# Config: run set_config.sh in project root (rare-disease-chat/) to write inference_config.json

echo "Starting VCAP-RDAgent Professional Dashboard..."
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "RDAgent will be available at: http://localhost:${PORT:-8080}/rdagent/"
echo ""

python3 rdagent_dashboard_api.py
