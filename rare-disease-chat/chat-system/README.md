# Chat System

This directory contains the intelligent chat system module of VCAP-RDAgent (rare-disease-chat), providing two usage modes.

For **installing dependencies** (venv, `uv pip install -e .`), see the project root [README.md](../README.md) section **"Start the chat system"**.

---

## Option 1: Web UI

Interact with the system through a web page in your browser.

1. **Start the MCP server first** (see "Starting the Server" in the project root [README.md](../README.md)).
2. In this directory, run:
   ```bash
   ./start_web_ui.sh
   ```
   Or: `python3 web_ui_api.py` (Chinese), `python3 web_ui_api_en.py` (English).
3. Open in your browser: **http://localhost:8080**

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
