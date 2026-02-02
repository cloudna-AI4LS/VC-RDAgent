# Phenotype-to-Disease Diagnosis System — Web UI Guide

## Overview

This is a web-based phenotype-to-disease diagnosis system with a graphical user interface and real-time streaming of diagnosis results. The recommended entry is:

- **RDAgent Dashboard** — Professional agent UI with pipeline steps and reasoning, served by `rdagent_dashboard_api.py`

## Features

- 🎨 **Modern UI**: Gradient layout and responsive design
- 🔄 **Real-time streaming**: Server-Sent Events (SSE) streaming
- 💬 **Chat history**: Session history and multi-turn conversation
- 📊 **Status indicators**: Live progress and pipeline stage information
- 🧠 **Token management**: Automatic history length handling to stay within limits
- 📋 **Dashboard**: Step-by-step pipeline display and reasoning view

## Dependencies

```bash
# Install FastAPI and Uvicorn
pip install fastapi uvicorn[standard]
```

## Starting the Service

### RDAgent Dashboard (recommended)

Default port: `8080` (override with env: `PORT=8082`).

```bash
# Startup script
./start_dashboard.sh

# Or run directly
python3 rdagent_dashboard_api.py
```

Access: **http://localhost:8080/rdagent/** (or the port you set).

## Accessing the UI

| Service            | URL                             |
|--------------------|---------------------------------|
| RDAgent Dashboard  | http://localhost:8080/rdagent/  |

From another machine on the same network, replace `localhost` with the host IP.

## Usage

1. **Enter query**: Type patient symptoms, phenotype descriptions, or HPO IDs in the input box
2. **Send**: Click “Send” or press Enter
3. **View results**: Progress and diagnosis results are streamed in real time
4. **Clear chat**: Click “Clear” to reset the current conversation

## Example Queries

- `Patient has intellectual disability, epilepsy, and facial dysmorphism`
- `HP:0001249, HP:0001250, HP:0000272`
- `Diagnose possible diseases`
- `What disease might this phenotype combination indicate?`

## API Endpoints (RDAgent Dashboard)

| Method | Path | Description |
|--------|------|-------------|
| GET | / | Redirect to /rdagent/ |
| GET | /rdagent, /rdagent/ | Serve `rdagent_dashboard.html` |
| GET | /api/config | Current model name (for dashboard badge) |
| POST | /api/chat/stream_dashboard | Streaming chat (SSE) |

Static assets are mounted under `/rdagent/` (e.g. `rdagent_dashboard.html`, CSS, JS).

## Architecture

```
┌─────────────┐
│  Web Browser │
└──────┬──────┘
       │ HTTP/SSE
       ▼
┌─────────────────────────────────────┐
│  rdagent_dashboard_api.py           │  ← RDAgent Dashboard (FastAPI)
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Controller Pipeline                 │
│  (phenotype_to_disease_controller_   │
│   langchain_stream_api)              │
│  - InfoExtractionAgent               │
│  - WorkflowAgent                     │
│  - Synthesizer                       │
│  - EvaluationAgent                   │
└─────────────────────────────────────┘
```

## File Structure

```
chat-system/
├── rdagent_dashboard_api.py   # RDAgent Dashboard backend
├── rdagent/
│   └── rdagent_dashboard.html # RDAgent Dashboard frontend
├── start_dashboard.sh         # Start Dashboard
├── WEB_UI_README.md           # This document
├── README.md                  # Directory overview
└── phenotype_to_disease_controller_langchain_stream_api.py   # Core controller
```

## Configuration

### Port

Default port `8080`. Override with env: `PORT=8082 ./start_dashboard.sh` or change at the end of `rdagent_dashboard_api.py`.

### CORS

Default allows all origins. For production, restrict `allow_origins` to your domain (e.g. `http://localhost:8080`).

### Task queue

Dashboard limits concurrent pipeline runs via `TASK_QUEUE_SIZE` (default `1`). Set in environment if needed: `TASK_QUEUE_SIZE=2 ./start_dashboard.sh`.

## Troubleshooting

### 1. Port in use

If port 8080 is in use, change the port or stop the process using it:

```bash
# Find process using the port
lsof -i :8080

# Or
netstat -tulpn | grep 8080
```

### 2. Static files not found

Ensure `rdagent/rdagent_dashboard.html` exists and the path is correct.

### 3. Missing dependencies

If you see import errors, install the required packages:

```bash
pip install fastapi uvicorn[standard] langchain langgraph
```

## Development

### Frontend

Edit `rdagent/rdagent_dashboard.html`; reload the browser to see changes (use uvicorn with `--reload` for auto-reload).

### Backend

Edit `rdagent_dashboard_api.py` and restart the service.

### Adding features

1. Add new stages in the controller stream API.
2. Use `yield` to send status and content.
3. Handle new message types in `rdagent_dashboard.html`.

## License

Same as the main project.
