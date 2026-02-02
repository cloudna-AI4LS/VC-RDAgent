#!/usr/bin/env python3
"""
RDAgent Dashboard API — standalone backend for the Professional Agent UI.

This module provides a separate FastAPI app for the VCAP-RDAgent dashboard:
- POST /api/chat/stream_dashboard — streaming endpoint (no page guard)
- GET /rdagent, /rdagent/ — serve rdagent_dashboard.html
- Static files under /rdagent/

Run: python rdagent_dashboard_api.py  (default port 8080)
LLM config: start_dashboard.sh writes inference_config.json from script variables; if missing, defaults in _LLM_DEFAULT_MODEL_CONFIG are merged. Controller reads inference_config.json.
"""

import asyncio
import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import only from existing modules (no modification of those files)
from phenotype_to_disease_controller_langchain_stream_api import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    MAX_TOKENS,
    count_messages_tokens,
)
from web_ui_api_en import ChatRequest, controller_pipeline_stream_en

logger = logging.getLogger("rdagent.dashboard_api")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Phase display labels for richer pipeline UI
PHASE_LABELS: Dict[str, str] = {
    "info_extraction": "Info extraction",
    "workflow": "Diagnosis flow",
    "classification": "Task type",
    "evaluation": "Evaluation",
    "synthesis": "Report",
    "complete": "Complete",
    "error": "Error",
}


async def _enrich_stream(
    stream: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    """
    Wrap the controller stream and emit optional 'step' events for pipeline UI.
    For each 'status' event we also emit a 'step' event with phase and display label.
    tool_response (tool_name + actual content) is emitted by web_ui_api_en when tools return.
    """
    async for chunk in stream:
        yield chunk
        try:
            line = chunk.strip()
            if not line:
                continue
            data = json.loads(line)
            typ = data.get("type")
            phase = data.get("phase") or ""
            payload = data.get("data") or ""

            if typ == "status" and phase:
                display = PHASE_LABELS.get(phase, phase)
                is_done = isinstance(payload, str) and (
                    payload.startswith("[Completed]") or payload.startswith("[All done]")
                )
                step_event = json.dumps(
                    {
                        "type": "step",
                        "phase": phase,
                        "display": display,
                        "status": "completed" if is_done else "running",
                        "message": payload if isinstance(payload, str) else "",
                    },
                    ensure_ascii=False,
                )
                yield step_event + "\n"
        except Exception:
            continue


# ---------------------------------------------------------------------------
# App and session state
# ---------------------------------------------------------------------------

# Task queue: limit concurrent pipeline runs (1 = strict queue, one at a time)
TASK_QUEUE_SIZE = int(os.environ.get("TASK_QUEUE_SIZE", "1"))
_task_semaphore: Optional[asyncio.Semaphore] = None


def _get_task_semaphore() -> asyncio.Semaphore:
    global _task_semaphore
    if _task_semaphore is None:
        _task_semaphore = asyncio.Semaphore(TASK_QUEUE_SIZE)
    return _task_semaphore


app = FastAPI(
    title="VCAP-RDAgent Dashboard API",
    description="Professional agent dashboard: streaming, steps, reasoning.",
)

sessions_dashboard: Dict[str, Dict[str, Any]] = {}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RDAGENT_DIR = os.path.join(SCRIPT_DIR, "rdagent")
INFERENCE_CONFIG_PATH = os.path.join(SCRIPT_DIR, "inference_config.json")

# LLM model_config defaults (same as start_dashboard.sh). Merged into inference_config.json when file exists; edit in start_dashboard.sh or inference_config.json.
_LLM_DEFAULT_MODEL_CONFIG = {
    "model": "Qwen/Qwen3-8B",
    "model_provider": "",
    "base_url": "http://192.168.0.127:8000/v1",
    "api_key": "EMPTY",
    "temperature": 0.1,
    "top_p": 0.95,
    "streaming": True,
}


def _ensure_llm_config() -> None:
    """Ensure inference_config.json has model_config (merge defaults with existing). Controller reads this file."""
    try:
        with open(INFERENCE_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        config = {}
    config["model_config"] = {**_LLM_DEFAULT_MODEL_CONFIG, **(config.get("model_config") or {})}
    try:
        with open(INFERENCE_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except OSError as e:
        logger.warning("Could not write inference_config.json: %s", e)


_ensure_llm_config()


def _get_model_name_from_config() -> str:
    """Read current model name from inference_config.json."""
    if not os.path.isfile(INFERENCE_CONFIG_PATH):
        return "Agent Dashboard"
    try:
        with open(INFERENCE_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        model_config = config.get("model_config") or {}
        return model_config.get("model") or "Agent Dashboard"
    except (json.JSONDecodeError, OSError):
        return "Agent Dashboard"


@app.get("/api/config")
async def get_config():
    """Return current model name for dashboard badge."""
    return {"model": _get_model_name_from_config()}


@app.get("/rdagent", response_class=HTMLResponse)
@app.get("/rdagent/", response_class=HTMLResponse)
async def rdagent_root():
    """Serve the professional agent dashboard page."""
    path = os.path.join(RDAGENT_DIR, "rdagent_dashboard.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    from fastapi.responses import HTMLResponse as HR
    return HR(content="<h1>RDAgent not found</h1>", status_code=404)  # type: ignore[arg-type]


@app.get("/", response_class=RedirectResponse)
async def root_redirect():
    """Redirect root to rdagent."""
    return RedirectResponse(url="/rdagent/")


@app.post("/api/chat/stream_dashboard")
async def chat_stream_dashboard(payload: ChatRequest):
    """
    Streaming endpoint for the professional dashboard.
    Same logic as stream_en but without page guard; optionally enriches with step events.
    """
    session_id = payload.session_id or f"dashboard_session_{len(sessions_dashboard)}"
    if session_id not in sessions_dashboard:
        sessions_dashboard[session_id] = {
            "messages": [],
            "created_at": asyncio.get_event_loop().time(),
        }

    session = sessions_dashboard[session_id]
    user_msg = HumanMessage(content=payload.query)
    session["messages"].append(user_msg)
    conversation_messages = session["messages"][:-1]

    total_tokens = count_messages_tokens(conversation_messages)
    if total_tokens > MAX_TOKENS:
        while total_tokens > MAX_TOKENS and len(conversation_messages) > 0:
            conversation_messages.pop(0)
            total_tokens = count_messages_tokens(conversation_messages)

    async def generate():
        sem = _get_task_semaphore()
        # Notify client they are in queue (keeps connection alive while waiting)
        yield f"data: {json.dumps({'type': 'status', 'data': 'Waiting in queue...', 'phase': 'queue'}, ensure_ascii=False)}\n\n"
        async with sem:
            full_response = ""
            try:
                base_stream = controller_pipeline_stream_en(
                    payload.query, conversation_messages
                )
                async for chunk in _enrich_stream(base_stream):
                    yield f"data: {chunk}\n\n"
                    try:
                        chunk_data = json.loads(chunk)
                        if chunk_data.get("type") == "content":
                            full_response += chunk_data.get("data", "")
                    except Exception:
                        pass

                if full_response:
                    session["messages"].append(AIMessage(content=full_response))
            except asyncio.CancelledError:
                logger.info("stream_dashboard: client disconnected, request cancelled")
            except (ConnectionResetError, ConnectionError, BrokenPipeError) as e:
                logger.info("stream_dashboard: client disconnected (%s)", type(e).__name__)
            except Exception as e:
                logger.exception("stream_dashboard generate() failed")
                try:
                    error_chunk = json.dumps(
                        {"type": "error", "data": str(e) or "Unknown error", "phase": "error"},
                        ensure_ascii=False,
                    )
                    yield f"data: {error_chunk}\n\n"
                except Exception:
                    pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# Mount static files so /rdagent/rdagent_dashboard.html and assets work
if os.path.exists(RDAGENT_DIR):
    app.mount("/rdagent", StaticFiles(directory=RDAGENT_DIR), name="rdagent")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    logger.info("RDAgent server: http://0.0.0.0:%s/rdagent/", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
