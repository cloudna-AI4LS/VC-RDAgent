#!/usr/bin/env python3
"""
RDAgent Dashboard API — standalone backend for the Professional Agent UI.

This module provides a separate FastAPI app for the VCAP-RDAgent dashboard:
- POST /api/chat/stream_dashboard — streaming endpoint (SSE, no page guard)
- WebSocket /api/chat/ws — same pipeline over WebSocket (one message per connection or per client message)
- GET /rdagent, /rdagent/ — serve rdagent_dashboard.html
- Static files under /rdagent/

Run: python rdagent_dashboard_api.py  (default port 8080)
LLM config: start_dashboard.sh writes inference_config.json from script variables; if missing, defaults in _LLM_DEFAULT_MODEL_CONFIG are merged. Controller reads inference_config.json.
"""

import asyncio
import json
import logging
import multiprocessing
import os
import queue
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Deque, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

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
try:
    TASK_QUEUE_SIZE = max(1, int(os.environ.get("TASK_QUEUE_SIZE", "1")))
except (ValueError, TypeError):
    TASK_QUEUE_SIZE = 1
_task_semaphore: Optional[asyncio.Semaphore] = None


@dataclass
class WaitingEntry:
    request_id: str


_waiting_queue: Deque[WaitingEntry] = deque()


def _get_task_semaphore() -> asyncio.Semaphore:
    global _task_semaphore
    if _task_semaphore is None:
        _task_semaphore = asyncio.Semaphore(TASK_QUEUE_SIZE)
    return _task_semaphore


def _queue_add() -> str:
    """Add a request to the global waiting queue and return its ID."""
    req_id = str(uuid.uuid4())
    _waiting_queue.append(WaitingEntry(request_id=req_id))
    return req_id


def _queue_remove(req_id: str) -> None:
    """Remove a request from the global waiting queue by ID."""
    if not req_id:
        return
    for idx, entry in enumerate(_waiting_queue):
        if entry.request_id == req_id:
            del _waiting_queue[idx]
            break


def _queue_position(req_id: str) -> int:
    """Return 1-based position of the request in the waiting queue; 0 if not found."""
    if not req_id:
        return 0
    for idx, entry in enumerate(_waiting_queue):
        if entry.request_id == req_id:
            return idx + 1
    return 0


async def _consume_stream_into_queue(
    query: str, messages: List[BaseMessage], out_queue: queue.Queue
) -> None:
    try:
        base_stream = controller_pipeline_stream_en(query, messages)
        enriched_stream = _enrich_stream(base_stream)
        try:
            async for chunk in enriched_stream:
                out_queue.put(("chunk", chunk))
        except Exception as e:
            out_queue.put(("error", str(e)))
    except Exception as e:
        out_queue.put(("error", str(e)))
    finally:
        out_queue.put(("end", None))


def _run_pipeline_in_thread(
    query: str,
    messages: List[BaseMessage],
    out_queue: queue.Queue,
    loop_holder: Optional[List[asyncio.AbstractEventLoop]] = None,
    task_holder: Optional[List[asyncio.Task]] = None,
) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    if loop_holder is not None:
        loop_holder.append(loop)
    try:
        task = loop.create_task(_consume_stream_into_queue(query, messages, out_queue))
        if task_holder is not None:
            task_holder.append(task)
        loop.run_until_complete(task)
    except asyncio.CancelledError:
        # _consume_stream_into_queue's finally already put ("end", None); no need to put again
        pass
    except Exception as e:
        try:
            out_queue.put(("error", str(e)))
            out_queue.put(("end", None))
        except Exception:
            pass
    finally:
        if task_holder:
            try:
                task_holder.clear()
            except Exception:
                pass
        if loop_holder:
            try:
                loop_holder.clear()
            except Exception:
                pass
        loop.close()


def _serialize_messages_for_process(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """Serialize conversation messages to picklable dicts for the worker process."""
    out = []
    for m in messages:
        name = type(m).__name__
        content = getattr(m, "content", "")
        out.append({"type": name, "content": content})
    return out


def _run_pipeline_process(
    query: str,
    messages_data: List[Dict[str, Any]],
    out_queue: multiprocessing.Queue,
) -> None:
    """
    Run pipeline in a subprocess. When the client disconnects, the main process
    can terminate() this process to immediately stop MCP/tool execution.
    """
    import asyncio

    from phenotype_to_disease_controller_langchain_stream_api import AIMessage, BaseMessage, HumanMessage

    def _deserialize_messages(data: List[Dict[str, Any]]) -> List[BaseMessage]:
        result = []
        for d in data:
            content = d.get("content", "")
            if d.get("type") == "HumanMessage":
                result.append(HumanMessage(content=content))
            else:
                result.append(AIMessage(content=content))
        return result

    messages = _deserialize_messages(messages_data)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_consume_stream_into_queue(query, messages, out_queue))
    except Exception:
        pass
    finally:
        loop.close()


app = FastAPI(
    title="VC-RDAgent Dashboard API",
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


# Patch injected into served HTML: skip empty assistant messages on load so refresh
# does not show "Waiting..." for in-flight replies that were never completed.
_INDENT = "                        "  # 24 spaces, matches loadConversation for-loop in HTML
_HTML_LOAD_CONVERSATION_SKIP_EMPTY_ASSISTANT = (
    f"var item = list[i];\n{_INDENT}var bubble = addMessage(item.role, item.content, item.msgId);"
)
_HTML_LOAD_CONVERSATION_SKIP_EMPTY_ASSISTANT_PATCH = (
    f"var item = list[i];\n{_INDENT}if (item.role === 'assistant' && !item.content) continue;\n{_INDENT}var bubble = addMessage(item.role, item.content, item.msgId);"
)


@app.get("/rdagent", response_class=HTMLResponse)
@app.get("/rdagent/", response_class=HTMLResponse)
async def rdagent_root():
    """Serve the professional agent dashboard page."""
    path = os.path.join(RDAGENT_DIR, "rdagent_dashboard.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        # Avoid restoring empty assistant messages so refresh does not show "Waiting..."
        if _HTML_LOAD_CONVERSATION_SKIP_EMPTY_ASSISTANT in html:
            html = html.replace(
                _HTML_LOAD_CONVERSATION_SKIP_EMPTY_ASSISTANT,
                _HTML_LOAD_CONVERSATION_SKIP_EMPTY_ASSISTANT_PATCH,
                1,
            )
        return html
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
            "created_at": asyncio.get_running_loop().time(),
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
        """
        Streaming wrapper around controller_pipeline_stream_en with heartbeat.

        - Uses a semaphore to limit concurrent runs.
        - Forwards chunks from the controller (plus step events from _enrich_stream)
          to the SSE client.
        - If no chunk is produced for KEEPALIVE_INTERVAL seconds, sends a
          lightweight keepalive event so the frontend knows the backend is alive.
        """
        sem = _get_task_semaphore()

        # Send a lightweight queue status event before acquiring the semaphore,
        # so the frontend can show "waiting in queue" even while this request
        # is still pending for a running pipeline. Use a global waiting queue
        # so that the position can be updated dynamically if needed.
        req_id = ""
        try:
            req_id = _queue_add()
            position = _queue_position(req_id)
            queue_event = json.dumps(
                {
                    "type": "status",
                    "data": f"Waiting in queue (#{position})...",
                    "phase": "queue",
                    "position": position,
                },
                ensure_ascii=False,
            )
            yield f"data: {queue_event}\n\n"
        except Exception:
            # Queue hint is purely cosmetic; do not affect the main pipeline.
            _queue_remove(req_id)

        async with sem:
            _queue_remove(req_id)
            full_response = ""
            try:
                KEEPALIVE_INTERVAL = float(os.environ.get("DASHBOARD_KEEPALIVE_INTERVAL", "10"))
            except (ValueError, TypeError):
                KEEPALIVE_INTERVAL = 10.0
            chunk_queue: multiprocessing.Queue = multiprocessing.Queue()
            messages_data = _serialize_messages_for_process(conversation_messages)
            worker_process = multiprocessing.Process(
                target=_run_pipeline_process,
                args=(payload.query, messages_data, chunk_queue),
                daemon=True,
            )
            worker_process.start()

            def _terminate_worker() -> None:
                try:
                    if worker_process.is_alive():
                        worker_process.terminate()
                        worker_process.join(timeout=2.0)
                        if worker_process.is_alive():
                            worker_process.kill()
                            worker_process.join(timeout=1.0)
                except Exception as e:
                    logger.warning("_terminate_worker: %s", e)

            try:
                loop = asyncio.get_running_loop()
                while True:
                    try:
                        kind, value = await loop.run_in_executor(
                            None,
                            lambda q=chunk_queue, t=KEEPALIVE_INTERVAL: q.get(timeout=t),
                        )
                    except queue.Empty:
                        try:
                            keepalive = json.dumps(
                                {"type": "keepalive", "data": "", "phase": "keepalive"},
                                ensure_ascii=False,
                            )
                            yield f"data: {keepalive}\n\n"
                        except Exception:
                            pass
                        continue

                    if kind == "end":
                        break
                    if kind == "error":
                        try:
                            err = json.dumps(
                                {"type": "error", "data": value or "Unknown error", "phase": "error"},
                                ensure_ascii=False,
                            )
                            yield f"data: {err}\n\n"
                        except Exception:
                            pass
                        _terminate_worker()
                        break

                    chunk = value
                    yield f"data: {chunk}\n\n"
                    try:
                        chunk_data = json.loads(chunk)
                        if chunk_data.get("type") == "content":
                            full_response += chunk_data.get("data", "")
                    except Exception:
                        pass

                if full_response:
                    session["messages"].append(AIMessage(content=full_response))
                worker_process.join(timeout=5.0)
            except asyncio.CancelledError:
                logger.info("stream_dashboard: client disconnected, request cancelled")
                _terminate_worker()
            except (ConnectionResetError, ConnectionError, BrokenPipeError) as e:
                logger.info("stream_dashboard: client disconnected (%s)", type(e).__name__)
                _terminate_worker()
            except Exception as e:
                logger.exception("stream_dashboard generate() failed")
                _terminate_worker()
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


async def _run_pipeline_and_send_ws(
    websocket: WebSocket,
    query: str,
    conversation_messages: List[BaseMessage],
    session: Dict[str, Any],
    sem: asyncio.Semaphore,
) -> None:
    """
    Run pipeline in subprocess and send chunks to WebSocket. Same logic as SSE generate()
    but sends raw JSON text frames. Caller must hold sem and handle disconnect.
    """
    try:
        KEEPALIVE_INTERVAL = float(os.environ.get("DASHBOARD_KEEPALIVE_INTERVAL", "10"))
    except (ValueError, TypeError):
        KEEPALIVE_INTERVAL = 10.0
    # Shorter probe interval for WebSocket so we detect client disconnect (e.g. refresh) sooner
    try:
        WS_PROBE_INTERVAL = float(os.environ.get("DASHBOARD_WS_PROBE_INTERVAL", "1"))
    except (ValueError, TypeError):
        WS_PROBE_INTERVAL = 1.0
    queue_get_timeout = min(WS_PROBE_INTERVAL, KEEPALIVE_INTERVAL)
    chunk_queue: multiprocessing.Queue = multiprocessing.Queue()
    messages_data = _serialize_messages_for_process(conversation_messages)
    worker_process = multiprocessing.Process(
        target=_run_pipeline_process,
        args=(query, messages_data, chunk_queue),
        daemon=True,
    )
    worker_process.start()

    def _terminate_worker() -> None:
        try:
            if worker_process.is_alive():
                worker_process.terminate()
                worker_process.join(timeout=2.0)
                if worker_process.is_alive():
                    worker_process.kill()
                    worker_process.join(timeout=1.0)
        except Exception as e:
            logger.warning("_terminate_worker: %s", e)

    loop = asyncio.get_running_loop()
    full_response = ""
    # Queue status is sent by chat_ws before acquiring semaphore so client sees it while waiting
    try:
        while True:
            try:
                kind, value = await loop.run_in_executor(
                    None,
                    lambda q=chunk_queue, t=queue_get_timeout: q.get(timeout=t),
                )
            except queue.Empty:
                try:
                    keepalive = json.dumps(
                        {"type": "keepalive", "data": "", "phase": "keepalive"},
                        ensure_ascii=False,
                    )
                    await websocket.send_text(keepalive)
                except Exception:
                    _terminate_worker()
                    return
                continue

            if kind == "end":
                break
            if kind == "error":
                try:
                    err = json.dumps(
                        {"type": "error", "data": value or "Unknown error", "phase": "error"},
                        ensure_ascii=False,
                    )
                    await websocket.send_text(err)
                except Exception:
                    pass
                _terminate_worker()
                return

            chunk = value
            try:
                await websocket.send_text(chunk)
            except Exception:
                _terminate_worker()
                return
            try:
                chunk_data = json.loads(chunk)
                if chunk_data.get("type") == "content":
                    full_response += chunk_data.get("data", "")
            except Exception:
                pass

        if full_response:
            session["messages"].append(AIMessage(content=full_response))
        worker_process.join(timeout=5.0)
    except (WebSocketDisconnect, ConnectionResetError, ConnectionError, BrokenPipeError) as e:
        logger.info("chat_ws: client disconnected (%s)", type(e).__name__)
        _terminate_worker()
    except Exception as e:
        logger.exception("chat_ws: pipeline failed")
        _terminate_worker()
        try:
            error_chunk = json.dumps(
                {"type": "error", "data": str(e) or "Unknown error", "phase": "error"},
                ensure_ascii=False,
            )
            await websocket.send_text(error_chunk)
        except Exception:
            pass


@app.websocket("/api/chat/ws")
async def chat_ws(websocket: WebSocket):
    """
    WebSocket endpoint: same pipeline as POST /api/chat/stream_dashboard.

    Protocol:
      - Client sends JSON text frames: {"query": "...", "session_id": "..."}.
      - Server sends a sequence of JSON text frames (same shape as SSE data payload):
        {"type": "status"|"step"|"content"|"reasoning"|"tool_response"|"error"|"done"|"keepalive", "data": "...", "phase": "..."}
    One request per client message: after "done" or "error", client may send another message
    on the same connection for the next turn.
    """
    await websocket.accept()
    sem = _get_task_semaphore()

    while True:
        try:
            raw = await websocket.receive_text()
        except WebSocketDisconnect:
            return
        except RuntimeError as e:
            # Client closed (e.g. refresh): Starlette may raise "WebSocket is not connected"
            if "not connected" in str(e).lower() or "accept" in str(e).lower():
                return
            raise
        try:
            payload = json.loads(raw)
            query = (payload.get("query") or "").strip()
            session_id = payload.get("session_id") or f"dashboard_session_{len(sessions_dashboard)}"
        except Exception:
            try:
                await websocket.send_text(json.dumps(
                    {"type": "error", "data": "Invalid JSON or missing query", "phase": "error"},
                    ensure_ascii=False,
                ))
            except Exception:
                pass
            continue
        if not query:
            try:
                await websocket.send_text(json.dumps(
                    {"type": "error", "data": "Empty query", "phase": "error"},
                    ensure_ascii=False,
                ))
            except Exception:
                pass
            continue

        if session_id not in sessions_dashboard:
            sessions_dashboard[session_id] = {
                "messages": [],
                "created_at": asyncio.get_running_loop().time(),
            }
        session = sessions_dashboard[session_id]
        user_msg = HumanMessage(content=query)
        session["messages"].append(user_msg)
        conversation_messages = session["messages"][:-1]

        total_tokens = count_messages_tokens(conversation_messages)
        if total_tokens > MAX_TOKENS:
            while total_tokens > MAX_TOKENS and len(conversation_messages) > 0:
                conversation_messages.pop(0)
                total_tokens = count_messages_tokens(conversation_messages)

        # Send queue status before acquiring semaphore so client shows "Queue" while actually waiting.
        # While waiting, periodically send keepalive (phase=queue) to keep the connection alive.
        req_id = ""
        try:
            req_id = _queue_add()
            position = _queue_position(req_id)
            queue_event = json.dumps(
                {
                    "type": "status",
                    "data": f"Waiting in queue (#{position})...",
                    "phase": "queue",
                    "position": position,
                },
                ensure_ascii=False,
            )
            await websocket.send_text(queue_event)
        except Exception:
            continue

        # Wait for semaphore with heartbeat while in the queue
        try:
            try:
                KEEPALIVE_INTERVAL = float(os.environ.get("DASHBOARD_QUEUE_KEEPALIVE_INTERVAL", "10"))
            except (ValueError, TypeError):
                KEEPALIVE_INTERVAL = 10.0

            got_sem = False
            while True:
                try:
                    await asyncio.wait_for(sem.acquire(), timeout=KEEPALIVE_INTERVAL)
                    got_sem = True
                    break
                except asyncio.TimeoutError:
                    try:
                        position = _queue_position(req_id)
                        keepalive = json.dumps(
                            {
                                "type": "keepalive",
                                "data": "",
                                "phase": "queue",
                                "position": position,
                            },
                            ensure_ascii=False,
                        )
                        await websocket.send_text(keepalive)
                    except Exception:
                        # Client likely gone; give up this request
                        raise

            # Remove from waiting queue: this request is now running.
            _queue_remove(req_id)

            try:
                await _run_pipeline_and_send_ws(websocket, query, conversation_messages, session, sem)
            finally:
                if got_sem:
                    sem.release()
        except Exception:
            _queue_remove(req_id)
            # Let outer loop handle disconnect / errors appropriately
            return


# Mount static files so /rdagent/rdagent_dashboard.html and assets work
if os.path.exists(RDAGENT_DIR):
    app.mount("/rdagent", StaticFiles(directory=RDAGENT_DIR), name="rdagent")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    logger.info("RDAgent server: http://0.0.0.0:%s/rdagent/", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
