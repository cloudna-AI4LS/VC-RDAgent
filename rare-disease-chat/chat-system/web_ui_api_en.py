#!/usr/bin/env python3
"""
FastAPI Web UI (English) for Phenotype-to-Disease Controller.

This module provides English-only status/error messages for the streaming UI API.
It is intended to be called only by the English UI page served at `/rdagent/index_en.html`
(the file lives at `rdagent/index_en.html` on disk).
"""

import asyncio
import json
import re
import time
import logging
from typing import AsyncGenerator, List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, RedirectResponse
from pydantic import BaseModel

# Import the existing controller modules (core logic remains unchanged)
from phenotype_to_disease_controller_langchain_stream_api import (
    count_messages_tokens,
    MAX_TOKENS,
    HumanMessage,
    AIMessage,
    BaseMessage,
    InfoExtractionAgent,
    WorkflowAgent,
    EvaluationAgent,
    classify_task_type,
    synthesize_results,
)

logger = logging.getLogger("rdagent.web_ui_api_en")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


def _require_called_by_index_en(request: Request) -> None:
    """
    Best-effort guard: only allow requests coming from `index_en.html`.

    Note: This is not a security boundary (headers can be spoofed), but it prevents
    accidental use by other pages (e.g., the Chinese UI).
    """
    # Check all possible header name variations (HTTP headers should be case-insensitive, but check all)
    ui_page = (
        request.headers.get("x-ui-page", "") or 
        request.headers.get("X-UI-Page", "") or
        request.headers.get("X-Ui-Page", "") or
        request.headers.get("x-Ui-Page", "")
    )
    referer = (
        request.headers.get("referer", "") or 
        request.headers.get("Referer", "") or
        request.headers.get("REFERER", "")
    )
    
    # Log all headers for debugging
    all_headers = dict(request.headers)
    logger.debug(f"English UI request headers: {all_headers}")
    logger.debug(f"Extracted ui_page: '{ui_page}', referer: '{referer}'")

    # Primary check: if x-ui-page header is present, it must be index_en.html
    # If header is missing, we'll rely on referer check instead
    if ui_page and ui_page != "index_en.html":
        logger.warning(f"English UI endpoint rejected: x-ui-page header is '{ui_page}', expected 'index_en.html'. Referer: '{referer}'")
        raise HTTPException(
            status_code=403,
            detail=f"Forbidden: this endpoint is only available for index_en.html. Received x-ui-page: '{ui_page}'",
        )

    # Secondary check: reject if referer explicitly points to Chinese page (contains index.html but not index_en.html)
    # Allow requests from /rdagent/, /rdagent, /en, or if referer contains index_en.html
    if referer and "index.html" in referer and "index_en.html" not in referer:
        logger.warning(f"English UI endpoint rejected: referer '{referer}' points to Chinese page")
        raise HTTPException(
            status_code=403,
            detail="Forbidden: invalid referer for the English UI endpoint.",
        )
    
    # If we get here, either:
    # 1. x-ui-page header is correct (index_en.html), OR
    # 2. x-ui-page header is missing/empty but referer doesn't point to Chinese page
    # Allow the request in both cases


def _tool_result_to_str(obj: Any) -> str:
    """Serialize tool result for dashboard tool_response display."""
    if obj is None:
        return ""
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, ensure_ascii=False, indent=2)
    return str(obj)


def _should_yield_stream_chunk(phase: str, chunk_type: str) -> bool:
    """Only yield content for synthesis phase so only final step shows in reply window."""
    if chunk_type == "reasoning":
        return True
    if chunk_type == "content":
        return phase == "synthesis"
    return True


def translate_workflow_status_en(status_msg: str) -> str:
    """Convert WorkflowAgent status messages into user-friendly English UI text."""
    if status_msg.startswith("Executing "):
        tool_name = status_msg.replace("Executing ", "").replace("...", "")
        tool_name_mapping = {
            "disease_diagnosis_tool": "disease diagnosis analysis",
        }
        display_name = tool_name_mapping.get(tool_name, tool_name)
        return f"Running {display_name}..."

    if status_msg.startswith("Completed "):
        completed_item = status_msg.replace("Completed ", "")
        if completed_item == "evaluation":
            return "[Completed] Diagnostic needs assessment"
        tool_name_mapping = {
            "disease_diagnosis_tool": "Disease diagnosis analysis",
        }
        display_name = tool_name_mapping.get(completed_item, completed_item)
        return f"[Completed] {display_name}"

    if status_msg == "No workflow tool needed":
        return "[Completed] No disease diagnosis required for this query"

    return status_msg


class WorkflowStatusCallback:
    """WorkflowAgent status callback that records translated UI status messages."""

    def __init__(self, status_messages: List[str]):
        self.status_messages = status_messages

    def __call__(self, status_msg: str):
        self.status_messages.append(translate_workflow_status_en(status_msg))


async def controller_pipeline_stream_en(
    user_query: str,
    conversation_messages: List[BaseMessage] | None = None,
    disease_diagnosis_retry: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Streaming version of the controller pipeline for the English UI.

    Yields JSON lines (not SSE-wrapped):
      {"type": "status|content|reasoning|error|done", "data": "...", "phase": "..."}
    """
    try:
        start_time = time.time()

        completed_tools: List[str] = []
        tool_call_counts: Dict[str, int] = {}
        all_tool_results: Dict[str, Any] = {}

        # Phase 0: Task type first (same classify_task_type as before, input = history + current query)
        yield json.dumps(
            {
                "type": "status",
                "data": "Identifying task type...",
                "phase": "classification",
            }
        ) + "\n"
        # Build context = history + current user query for classify_task_type
        parts = []
        if conversation_messages:
            for msg in conversation_messages:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                content = getattr(msg, "content", None) or ""
                parts.append(f"{role}: {content}")
        parts.append(f"User (current): {user_query}")
        context_str = "\n\n".join(parts)
        task_type = await classify_task_type(user_query, context_str)
        task_type_names = {
            "general_inquiry": "General inquiry",
            "phenotype_extraction": "Phenotype extraction",
            "disease_diagnosis": "Disease diagnosis",
            "disease_case_extraction": "Disease case extraction",
            "disease_information_retrieval": "Disease information retrieval",
        }
        task_type_display = task_type_names.get(task_type, task_type)
        yield json.dumps(
            {
                "type": "status",
                "data": f"[Completed] Task type: {task_type_display}",
                "phase": "classification",
            }
        ) + "\n"

        # Reject general inquiry to avoid abuse of computing resources
        if task_type == "general_inquiry":
            reject_msg = (
                "This system only supports **rare disease diagnosis related** queries. "
                "Please ask questions about phenotype extraction, disease diagnosis, disease case extraction, or disease information retrieval. "
                "General or off-topic questions are not supported."
            )
            yield json.dumps(
                {"type": "content", "data": reject_msg, "phase": "classification"},
                ensure_ascii=False,
            ) + "\n"
            yield json.dumps(
                {
                    "type": "status",
                    "data": "[Completed] General inquiry rejected",
                    "phase": "complete",
                }
            ) + "\n"
            yield json.dumps(
                {"type": "done", "data": "", "phase": "complete"},
            ) + "\n"
            return

        # Phase 1: InfoExtractionAgent (stream reasoning/content to page via queue)
        yield json.dumps(
            {
                "type": "status",
                "data": "Extracting phenotypes, symptoms, or disease information...",
                "phase": "info_extraction",
            }
        ) + "\n"

        stream_queue: asyncio.Queue = asyncio.Queue()

        def on_info_stream(chunk_type: str, data: str) -> None:
            stream_queue.put_nowait(("info_extraction", chunk_type, data))

        info_extraction_agent = InfoExtractionAgent()
        info_task = asyncio.create_task(
            info_extraction_agent.run(
                user_query, conversation_messages, stream_callback=on_info_stream
            )
        )
        while True:
            try:
                phase, ct, data = await asyncio.wait_for(stream_queue.get(), timeout=0.05)
                if not _should_yield_stream_chunk(phase, ct):
                    continue
                yield json.dumps(
                    {"type": ct, "data": data, "phase": phase}, ensure_ascii=False
                ) + "\n"
            except asyncio.TimeoutError:
                if info_task.done():
                    while not stream_queue.empty():
                        try:
                            phase, ct, data = stream_queue.get_nowait()
                            if not _should_yield_stream_chunk(phase, ct):
                                continue
                            yield json.dumps(
                                {"type": ct, "data": data, "phase": phase},
                                ensure_ascii=False,
                            ) + "\n"
                        except asyncio.QueueEmpty:
                            break
                    break
        info_extraction_results = await info_task

        for tool_name in info_extraction_results.keys():
            completed_tools.append(tool_name)
            tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1

        all_tool_results.update(info_extraction_results)

        for _name, _result in info_extraction_results.items():
            yield json.dumps(
                {
                    "type": "tool_response",
                    "tool_name": _name,
                    "data": _tool_result_to_str(_result),
                },
                ensure_ascii=False,
            ) + "\n"

        yield json.dumps(
            {
                "type": "status",
                "data": "[Completed] Information extraction",
                "phase": "info_extraction",
            }
        ) + "\n"

        # Phase 2: WorkflowAgent
        yield json.dumps(
            {
                "type": "status",
                "data": "Assessing diagnostic needs...",
                "phase": "workflow",
            }
        ) + "\n"

        workflow_status_messages: List[str] = []
        workflow_status_callback = WorkflowStatusCallback(workflow_status_messages)
        yielded_workflow_status: set = set()  # avoid duplicate status lines
        workflow_stream_queue: asyncio.Queue = asyncio.Queue()

        def get_workflow_status_chunk(data: str) -> Optional[str]:
            key = ("workflow", data)
            if key in yielded_workflow_status:
                return None
            yielded_workflow_status.add(key)
            return json.dumps(
                {"type": "status", "data": data, "phase": "workflow"},
            ) + "\n"

        def on_workflow_stream(chunk_type: str, data: str) -> None:
            workflow_stream_queue.put_nowait(("workflow", chunk_type, data))

        workflow_agent = WorkflowAgent()
        workflow_task = asyncio.create_task(
            workflow_agent.run(
                user_query,
                tool_results=all_tool_results,
                conversation_messages=conversation_messages,
                status_callback=workflow_status_callback,
                stream_callback=on_workflow_stream,
            )
        )

        last_status_count = 0
        while not workflow_task.done():
            while not workflow_stream_queue.empty():
                try:
                    phase, ct, data = workflow_stream_queue.get_nowait()
                    if not _should_yield_stream_chunk(phase, ct):
                        continue
                    yield json.dumps(
                        {"type": ct, "data": data, "phase": phase},
                        ensure_ascii=False,
                    ) + "\n"
                except asyncio.QueueEmpty:
                    break
            if len(workflow_status_messages) > last_status_count:
                for i in range(last_status_count, len(workflow_status_messages)):
                    chunk = get_workflow_status_chunk(workflow_status_messages[i])
                    if chunk is not None:
                        yield chunk
                last_status_count = len(workflow_status_messages)

            await asyncio.sleep(0.01)

        while not workflow_stream_queue.empty():
            try:
                phase, ct, data = workflow_stream_queue.get_nowait()
                if not _should_yield_stream_chunk(phase, ct):
                    continue
                yield json.dumps(
                    {"type": ct, "data": data, "phase": phase}, ensure_ascii=False
                ) + "\n"
            except asyncio.QueueEmpty:
                break
        if len(workflow_status_messages) > last_status_count:
            for i in range(last_status_count, len(workflow_status_messages)):
                chunk = get_workflow_status_chunk(workflow_status_messages[i])
                if chunk is not None:
                    yield chunk

        workflow_info = await workflow_task

        # If tools are missing, retry information extraction
        if workflow_info.get("missing_tools"):
            missing_tools = workflow_info["missing_tools"]
            yield json.dumps(
                {
                    "type": "status",
                    "data": "More information is needed. Re-extracting required details...",
                    "phase": "info_extraction",
                }
            ) + "\n"

            retry_agent = InfoExtractionAgent(specified_tools=missing_tools)
            retry_task = asyncio.create_task(
                retry_agent.run(
                    user_query, conversation_messages, stream_callback=on_info_stream
                )
            )
            while True:
                try:
                    phase, ct, data = await asyncio.wait_for(stream_queue.get(), timeout=0.05)
                    if not _should_yield_stream_chunk(phase, ct):
                        continue
                    yield json.dumps(
                        {"type": ct, "data": data, "phase": phase}, ensure_ascii=False
                    ) + "\n"
                except asyncio.TimeoutError:
                    if retry_task.done():
                        while not stream_queue.empty():
                            try:
                                phase, ct, data = stream_queue.get_nowait()
                                if not _should_yield_stream_chunk(phase, ct):
                                    continue
                                yield json.dumps(
                                    {"type": ct, "data": data, "phase": phase},
                                    ensure_ascii=False,
                                ) + "\n"
                            except asyncio.QueueEmpty:
                                break
                        break
            retry_results = await retry_task

            for tool_name, result in retry_results.items():
                all_tool_results[tool_name] = result
                if tool_name not in completed_tools:
                    completed_tools.append(tool_name)
                tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
                yield json.dumps(
                    {
                        "type": "tool_response",
                        "tool_name": tool_name,
                        "data": _tool_result_to_str(result),
                    },
                    ensure_ascii=False,
                ) + "\n"

            # Mark information-extraction phase as completed again so the UI
            # can re-highlight the \"Info extraction\" step after the retry.
            yield json.dumps(
                {
                    "type": "status",
                    "data": "[Completed] Information extraction",
                    "phase": "info_extraction",
                }
            ) + "\n"

            yield json.dumps(
                {
                    "type": "status",
                    "data": "Re-assessing diagnostic needs...",
                    "phase": "workflow",
                }
            ) + "\n"

            workflow_status_messages.clear()
            last_status_count = 0
            yielded_workflow_status.clear()

            workflow_agent = WorkflowAgent()
            workflow_task = asyncio.create_task(
                workflow_agent.run(
                    user_query,
                    tool_results=all_tool_results,
                    conversation_messages=conversation_messages,
                    status_callback=workflow_status_callback,
                    stream_callback=on_workflow_stream,
                )
            )

            while not workflow_task.done():
                while not workflow_stream_queue.empty():
                    try:
                        phase, ct, data = workflow_stream_queue.get_nowait()
                        if not _should_yield_stream_chunk(phase, ct):
                            continue
                        yield json.dumps(
                            {"type": ct, "data": data, "phase": phase},
                            ensure_ascii=False,
                        ) + "\n"
                    except asyncio.QueueEmpty:
                        break
                if len(workflow_status_messages) > last_status_count:
                    for i in range(last_status_count, len(workflow_status_messages)):
                        chunk = get_workflow_status_chunk(workflow_status_messages[i])
                        if chunk is not None:
                            yield chunk
                    last_status_count = len(workflow_status_messages)

                await asyncio.sleep(0.05)

            while not workflow_stream_queue.empty():
                try:
                    phase, ct, data = workflow_stream_queue.get_nowait()
                    if not _should_yield_stream_chunk(phase, ct):
                        continue
                    yield json.dumps(
                        {"type": ct, "data": data, "phase": phase},
                        ensure_ascii=False,
                    ) + "\n"
                except asyncio.QueueEmpty:
                    break
            if len(workflow_status_messages) > last_status_count:
                for i in range(last_status_count, len(workflow_status_messages)):
                    chunk = get_workflow_status_chunk(workflow_status_messages[i])
                    if chunk is not None:
                        yield chunk

            workflow_info = await workflow_task

        if workflow_info.get("workflow") is not None:
            wf_name = workflow_info["workflow"]
            wf_result = workflow_info["result"]
            all_tool_results[wf_name] = wf_result
            completed_tools.append(wf_name)
            tool_call_counts[wf_name] = tool_call_counts.get(wf_name, 0) + 1
            yield json.dumps(
                {
                    "type": "tool_response",
                    "tool_name": wf_name,
                    "data": _tool_result_to_str(wf_result),
                },
                ensure_ascii=False,
            ) + "\n"
        # Completion status already sent via workflow_status_callback; avoid duplicate status log.

        # Override task_type when workflow actually ran disease_diagnosis_tool
        if workflow_info.get("workflow") == "disease_diagnosis_tool":
            task_type = "disease_diagnosis"

        if task_type == "disease_diagnosis" and workflow_info.get("workflow") is None:
            if disease_diagnosis_retry:
                # Already retried once with enhanced_query; return error as workflow tool result
                workflow_result_msg = "Disease diagnosis workflow was not executed. Please try rephrasing or call the diagnosis tool explicitly."
                workflow_info["workflow"] = "disease_diagnosis_tool"
                workflow_info["result"] = workflow_result_msg
                all_tool_results["disease_diagnosis_tool"] = workflow_result_msg
                completed_tools.append("disease_diagnosis_tool")
                tool_call_counts["disease_diagnosis_tool"] = tool_call_counts.get("disease_diagnosis_tool", 0) + 1
                yield json.dumps(
                    {
                        "type": "tool_response",
                        "tool_name": "disease_diagnosis_tool",
                        "data": workflow_result_msg,
                    },
                    ensure_ascii=False,
                ) + "\n"
                # Fall through to Phase 4/5 so this result is used in evaluation/synthesis
            else:
                yield json.dumps(
                    {
                        "type": "status",
                        "data": "Disease diagnosis is required but was not executed. Restarting diagnosis workflow...",
                        "phase": "classification",
                    }
                ) + "\n"

                enhanced_query = f"""{user_query}

**IMPORTANT: This query requires disease diagnosis. You MUST call the **disease_case_extractor_tool** to obtain disease cases, and MUST call the workflow tool **disease_diagnosis_tool** to perform the diagnosis analysis.**"""

                async for chunk in controller_pipeline_stream_en(enhanced_query, conversation_messages, disease_diagnosis_retry=True):
                    yield chunk
                return

        # Phase 4: Evaluation (moved right after workflow; final_response uses workflow result)
        evaluation_response_clean = ""
        if task_type == "disease_diagnosis":
            yield json.dumps(
                {
                    "type": "status",
                    "data": "Evaluating diagnostic results...",
                    "phase": "evaluation",
                }
            ) + "\n"

            eval_stream_queue: asyncio.Queue = asyncio.Queue()

            def on_eval_stream(chunk_type: str, data: str) -> None:
                eval_stream_queue.put_nowait(("evaluation", chunk_type, data))

            evaluation_agent = EvaluationAgent()
            workflow_result_for_eval = str(workflow_info.get("result", "") or "")
            eval_task = asyncio.create_task(
                evaluation_agent.run(
                    user_query=user_query,
                    all_tool_results=all_tool_results,
                    conversation_messages=conversation_messages,
                    final_response=workflow_result_for_eval,
                    task_type=task_type,
                    stream_callback=on_eval_stream,
                )
            )
            while True:
                try:
                    phase, ct, data = await asyncio.wait_for(
                        eval_stream_queue.get(), timeout=0.05
                    )
                    if not _should_yield_stream_chunk(phase, ct):
                        continue
                    yield json.dumps(
                        {"type": ct, "data": data, "phase": phase}, ensure_ascii=False
                    ) + "\n"
                except asyncio.TimeoutError:
                    if eval_task.done():
                        while not eval_stream_queue.empty():
                            try:
                                phase, ct, data = eval_stream_queue.get_nowait()
                                if not _should_yield_stream_chunk(phase, ct):
                                    continue
                                yield json.dumps(
                                    {"type": ct, "data": data, "phase": phase},
                                    ensure_ascii=False,
                                ) + "\n"
                            except asyncio.QueueEmpty:
                                break
                        break
            evaluation_results = await eval_task

            evaluation_tool_results = evaluation_results.get("evaluation_tool_results") or {}
            for _name, _result in evaluation_tool_results.items():
                yield json.dumps(
                    {
                        "type": "tool_response",
                        "tool_name": _name,
                        "data": _tool_result_to_str(_result),
                    },
                    ensure_ascii=False,
                ) + "\n"

            evaluation_response = evaluation_results.get("evaluation_response", "") or ""
            reasoning_matches = re.finditer(r"<think>(.*?)</think>", evaluation_response, re.DOTALL)

            for match in reasoning_matches:
                reasoning_text = match.group(1)
                if reasoning_text.strip():
                    yield json.dumps(
                        {
                            "type": "reasoning",
                            "data": reasoning_text,
                            "phase": "evaluation",
                        }
                    ) + "\n"

            evaluation_response_clean = re.sub(
                r"<think>.*?</think>", "", evaluation_response, flags=re.DOTALL
            )
            evaluation_response_clean = re.sub(
                r".*?</think>", "", evaluation_response_clean, flags=re.DOTALL
            ).strip()

            yield json.dumps(
                {
                    "type": "status",
                    "data": "[Completed] Diagnostic evaluation",
                    "phase": "evaluation",
                }
            ) + "\n"

            # Feed evaluation back into synthesis (do not output as separate content to avoid duplication)
            if evaluation_response_clean:
                all_tool_results["diagnostic_evaluation"] = evaluation_response_clean

        # Phase 5: Synthesis (moved to the very end; includes evaluation feedback)
        yield json.dumps(
            {
                "type": "status",
                "data": "Generating the analysis report...",
                "phase": "synthesis",
            }
        ) + "\n"

        synthesis_stream_queue: asyncio.Queue = asyncio.Queue()

        def on_synthesis_stream(chunk_type: str, data: str) -> None:
            synthesis_stream_queue.put_nowait(("synthesis", chunk_type, data))

        synthesis_task = asyncio.create_task(
            synthesize_results(
                user_query=user_query,
                tool_results=all_tool_results,
                conversation_messages=conversation_messages,
                stream_callback=on_synthesis_stream,
            )
        )
        while True:
            try:
                phase, ct, data = await asyncio.wait_for(
                    synthesis_stream_queue.get(), timeout=0.05
                )
                if not _should_yield_stream_chunk(phase, ct):
                    continue
                yield json.dumps(
                    {"type": ct, "data": data, "phase": phase}, ensure_ascii=False
                ) + "\n"
            except asyncio.TimeoutError:
                if synthesis_task.done():
                    while not synthesis_stream_queue.empty():
                        try:
                            phase, ct, data = synthesis_stream_queue.get_nowait()
                            if not _should_yield_stream_chunk(phase, ct):
                                continue
                            yield json.dumps(
                                {"type": ct, "data": data, "phase": phase},
                                ensure_ascii=False,
                            ) + "\n"
                        except asyncio.QueueEmpty:
                            break
                    break
        await synthesis_task

        yield json.dumps(
            {
                "type": "status",
                "data": "[Completed] Report generated",
                "phase": "synthesis",
            }
        ) + "\n"

        total_time = time.time() - start_time
        yield json.dumps(
            {
                "type": "status",
                "data": f"[All done] Analysis completed (elapsed: {total_time:.2f}s)",
                "phase": "complete",
            }
        ) + "\n"

        yield json.dumps(
            {
                "type": "done",
                "data": "",
                "phase": "complete",
            }
        ) + "\n"

    except Exception as e:
        # Log full traceback on server only (avoid leaking stack traces to client)
        logger.exception("controller_pipeline_stream_en failed")

        # Return only the exception message to the client (no traceback).
        error_msg = str(e) if e else "Unknown error"
        yield json.dumps(
            {
                "type": "error",
                "data": error_msg,
                "phase": "error",
            }
        ) + "\n"


def register_english_ui_routes(app: FastAPI, static_dir: str) -> None:
    """
    Register English UI routes onto an existing FastAPI app.

    - GET  /en                 -> redirects to `/rdagent/index_en.html`
    - POST /api/chat/stream_en -> English streaming endpoint (guarded)
    """

    sessions_en: Dict[str, Dict[str, Any]] = {}

    @app.get("/en")
    async def read_english_root():
        # Redirect to the actual static file so the Referer contains `index_en.html`
        return RedirectResponse(url="/rdagent/index_en.html")

    @app.post("/api/chat/stream_en")
    async def chat_stream_en(request: Request, payload: ChatRequest):
        _require_called_by_index_en(request)

        session_id = payload.session_id or f"en_session_{len(sessions_en)}"
        if session_id not in sessions_en:
            sessions_en[session_id] = {
                "messages": [],
                "created_at": asyncio.get_event_loop().time(),
            }

        session = sessions_en[session_id]

        user_msg = HumanMessage(content=payload.query)
        session["messages"].append(user_msg)

        conversation_messages = session["messages"][:-1]

        total_tokens = count_messages_tokens(conversation_messages)
        if total_tokens > MAX_TOKENS:
            while total_tokens > MAX_TOKENS and len(conversation_messages) > 0:
                conversation_messages.pop(0)
                total_tokens = count_messages_tokens(conversation_messages)

        async def generate():
            full_response = ""
            try:
                async for chunk in controller_pipeline_stream_en(payload.query, conversation_messages):
                    yield f"data: {chunk}\n\n"
                    try:
                        chunk_data = json.loads(chunk)
                        if chunk_data.get("type") == "content":
                            full_response += chunk_data.get("data", "")
                    except Exception:
                        pass

                if full_response:
                    session["messages"].append(AIMessage(content=full_response))
            except Exception as e:
                logger.exception("SSE generate() failed")

                # Return only the exception message to the client (no traceback).
                error_msg = str(e) if e else "Unknown error"
                error_chunk = json.dumps(
                    {"type": "error", "data": error_msg, "phase": "error"}
                )
                yield f"data: {error_chunk}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

