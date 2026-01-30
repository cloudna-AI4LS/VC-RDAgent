# Phenotype-to-Disease Diagnosis System — Web UI Guide

## Overview

This is a web-based phenotype-to-disease diagnosis system with a graphical user interface and real-time streaming of diagnosis results.

## Features

- 🎨 **Modern UI**: Gradient layout and responsive design
- 🔄 **Real-time streaming**: Server-Sent Events (SSE) streaming
- 💬 **Chat history**: Session history and multi-turn conversation
- 📊 **Status indicators**: Live progress and stage information
- 🧠 **Token management**: Automatic history length handling to stay within limits

## Dependencies

```bash
# Install FastAPI and Uvicorn
pip install fastapi uvicorn[standard]
```

## Starting the Service

### Option 1: Startup script

```bash
./start_web_ui.sh
```

### Option 2: Run Python directly

```bash
# Chinese UI
python3 web_ui_api.py

# English UI
python3 web_ui_api_en.py
```

### Option 3: Uvicorn command

```bash
uvicorn web_ui_api:app --host 0.0.0.0 --port 8080 --reload
```

## Accessing the UI

After starting the service locally, open in your browser:

```
http://localhost:8080
```

From another machine on the same network, replace `localhost` with the host machine’s IP.

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

## API Endpoints

### POST /api/chat/stream
Streaming chat API; returns Server-Sent Events.

**Request body:**
```json
{
    "query": "Patient symptom description",
    "session_id": "Optional session ID"
}
```

**Response:** Server-Sent Events stream

### POST /api/chat
Non-streaming chat API (for compatibility).

**Request body:**
```json
{
    "query": "Patient symptom description",
    "session_id": "Optional session ID"
}
```

**Response:**
```json
{
    "session_id": "session_xxx",
    "response": "Full response text",
    "status": "success"
}
```

### DELETE /api/session/{session_id}
Clear the specified session.

### GET /api/sessions
List all sessions.

## Architecture

```
┌─────────────┐
│  Web Browser │
└──────┬──────┘
       │ HTTP/SSE
       ▼
┌─────────────┐
│  FastAPI Server  │
│  (web_ui_api.py) │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Controller Pipeline                 │
│  (phenotype_to_disease_controller)   │
│  - InfoExtractionAgent               │
│  - WorkflowAgent                     │
│  - Synthesizer                       │
│  - EvaluationAgent                   │
└─────────────────────────────────────┘
```

## File Structure

```
chat-system/
├── web_ui_api.py              # FastAPI backend (Chinese)
├── web_ui_api_en.py           # FastAPI backend (English)
├── rdagent/
│   ├── index.html             # Chinese frontend
│   └── index_en.html          # English frontend
├── start_web_ui.sh            # Startup script
├── WEB_UI_README.md           # This document
├── README.md                  # Directory overview
└── phenotype_to_disease_controller_langchain_stream_api.py   # Core controller
```

## Configuration

### Port

Default port is `8080`. Change it at the end of `web_ui_api.py`:

```python
uvicorn.run(app, host="0.0.0.0", port=8080)
```

### CORS

The default allows all origins. For production, restrict `allow_origins` to your domain (e.g. `http://localhost:8080`).

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

Ensure `rdagent/index.html` exists and the path is correct.

### 3. Missing dependencies

If you see import errors, install the required packages:

```bash
pip install fastapi uvicorn[standard] langchain langgraph
```

## Development

### Editing the frontend

Edit `rdagent/index.html` (Chinese) or `rdagent/index_en.html` (English). Reload the browser to see changes (with `--reload` enabled).

### Editing the backend

Edit `web_ui_api.py` and restart the service.

### Adding features

1. Add new stages in the `controller_pipeline_stream` function
2. Use `yield` to send status updates and content
3. Handle new message types in the frontend (`index.html`)

## License

Same as the main project.
