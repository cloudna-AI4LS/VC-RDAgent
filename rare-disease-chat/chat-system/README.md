# Chat System

This directory contains the intelligent chat system module of VCAP-RDAgent (rare-disease-chat), providing two usage modes.

For **installing dependencies** (venv, `uv pip install -e .`), see the project root [README.md](../README.md) section **"Start the chat system"**.

---

## Option 1: Web UI (RDAgent Dashboard)

Interact with the system through the RDAgent Dashboard in your browser.

1. **Start the MCP server first** (see "Starting the Server" in the project root [README.md](../README.md)).
2. In this directory, run:
   ```bash
   ./start_dashboard.sh
   ```
   Or: `python3 rdagent_dashboard_api.py`
3. Open in your browser: **http://localhost:8080/rdagent/**

See [WEB_UI_README.md](./WEB_UI_README.md) for details.

---

## Option 2: Local CLI

Run interactively in the terminal.

1. **Start the MCP server first** (see "Starting the Server" in the project root [README.md](../README.md)).
2. In this directory, run:
   ```bash
   python phenotype_to_disease_controller_langchain_stream_api.py
   ```
3. Enter your query in the terminal; type `quit` to exit and `clear` to clear chat history.

For project overview, Docker usage, environment variables, etc., see the parent [README.md](../README.md).
