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
    get_model,
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

        # Phase 1: InfoExtractionAgent
        yield json.dumps(
            {
                "type": "status",
                "data": "Extracting phenotypes, symptoms, or disease information...",
                "phase": "info_extraction",
            }
        ) + "\n"

        info_extraction_agent = InfoExtractionAgent()
        info_extraction_results = await info_extraction_agent.run(
            user_query, conversation_messages
        )

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

        def get_workflow_status_chunk(data: str) -> Optional[str]:
            key = ("workflow", data)
            if key in yielded_workflow_status:
                return None
            yielded_workflow_status.add(key)
            return json.dumps(
                {"type": "status", "data": data, "phase": "workflow"},
            ) + "\n"

        workflow_agent = WorkflowAgent()
        workflow_task = asyncio.create_task(
            workflow_agent.run(
                user_query,
                tool_results=all_tool_results,
                conversation_messages=conversation_messages,
                status_callback=workflow_status_callback,
            )
        )

        last_status_count = 0
        while not workflow_task.done():
            if len(workflow_status_messages) > last_status_count:
                for i in range(last_status_count, len(workflow_status_messages)):
                    chunk = get_workflow_status_chunk(workflow_status_messages[i])
                    if chunk is not None:
                        yield chunk
                last_status_count = len(workflow_status_messages)

            await asyncio.sleep(0.01)

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
            retry_results = await retry_agent.run(user_query, conversation_messages)

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
                )
            )

            while not workflow_task.done():
                if len(workflow_status_messages) > last_status_count:
                    for i in range(last_status_count, len(workflow_status_messages)):
                        chunk = get_workflow_status_chunk(workflow_status_messages[i])
                        if chunk is not None:
                            yield chunk
                    last_status_count = len(workflow_status_messages)

                await asyncio.sleep(0.05)

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

            evaluation_agent = EvaluationAgent()
            workflow_result_for_eval = str(workflow_info.get("result", "") or "")
            evaluation_results = await evaluation_agent.run(
                user_query=user_query,
                all_tool_results=all_tool_results,
                conversation_messages=conversation_messages,
                final_response=workflow_result_for_eval,
                task_type=task_type,
            )

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

        model = await get_model("synthesizer")

        results_text = "\n\n".join(
            [
                f"**{tool_name.upper()} Results:**\n{result}"
                for tool_name, result in all_tool_results.items()
            ]
        )

        synthesis_prompt = f"""You are a Medical Report Synthesizer.

**Your Task:** Synthesize the results from multiple tools into a comprehensive, clear response.

**User Query:** {user_query}

**Tool Results:**
{results_text}

**Guidelines:**
1. Integrate information from all tool results
2. Present information clearly and logically
3. Use tables or lists when appropriate
4. Highlight key findings
5. If results are inconsistent, note the discrepancies
6. Provide a direct answer to the user's question
7. Be concise but comprehensive

**Generate a well-structured final response:**"""

        if "disease_diagnosis_tool" in all_tool_results:
            synthesis_prompt = f"""
You are a Medical Report Synthesizer specialized in disease diagnosis.

**Your Task:** Synthesize the results from multiple tools into a comprehensive, clear diagnostic report.

**User Query:** {user_query}

**Tool Results:**
{results_text}

**Required Output Format:**
You MUST structure your response exactly according to the following sections:

1. **Phenotype Analysis**
   - Summarize the key phenotypes, symptoms, and clinical features from the user query and phenotype extraction tool results
   - Present them in a clear, organized manner

2. **Related Disease Case Analysis**
   - Provide a highly summarized and synthesized overview of the disease cases from the disease case extraction tool results
   - Do NOT analyze each disease case individually
   - Instead, synthesize and summarize the key patterns, common features, and important insights across all extracted cases
   - Focus on overall trends, shared characteristics, and high-level findings rather than case-by-case details

3. **Top-5 Differential Diagnosis**
   - Present the final top 5 differential diagnoses from the 'disease_diagnosis_tool' and 'diagnostic_evaluation' tool results
    (in table markdown format):
    | Disease Name | Matched Patient Phenotypes | Unmatched Patient Phenotypes | the Key Symptoms in the Disease but Not in the Patient | Matching Degree |
    | --- | --- | --- | --- | --- |
    | Disease Name 1 | Matched Patient Phenotypes | Unmatched Patient Phenotypes | the Key Symptoms in the Disease but Not in the Patient | Matching Degree |
    | ... | ... | ... | ... | ... |
    | Disease Name 5 | Matched Patient Phenotypes | Unmatched Patient Phenotypes | the Key Symptoms in the Disease but Not in the Patient | Matching Degree |
   - Other important information for the top 5 differential diagnoses

4. Other Differential Diagnoses
   - Provide a list of other differential diagnoses from the 'disease_diagnosis_tool' results
   - For each diagnosis, include:
     - Diagnosis name
     - Key matching features
     - Other important information

5. Additional Information
   - Based on the information from the previous sections and your medical knowledge, provide additional relevant information, insights, or recommendations
   - This may include:
     - Additional diagnostic considerations or alternative perspectives
     - Important clinical considerations, warnings, or precautions
     - Recommended follow-up tests, examinations, or monitoring
     - Treatment considerations or management suggestions (if appropriate)
     - Limitations or uncertainties in the current diagnosis
     - Any other relevant medical information that would be helpful for the user

**Guidelines:**
- Use clear section headers (## for main sections, ### for subsections)
- Use tables or lists when appropriate for better readability
- Highlight key findings and important information
- Be concise but comprehensive
- If results are inconsistent, note the discrepancies
- Provide actionable insights when possible

**Generate a well-structured diagnostic report following the format above:**"""

        messages = [HumanMessage(content=synthesis_prompt)]
        if conversation_messages:
            messages = conversation_messages + messages

        final_response_chunks: List[str] = []
        accumulated_reasoning: List[str] = [] 
        reasoning_buffer = ""
        end_tag = "</think>"
        end_tag_len = len(end_tag)
        response_start_tag = "FINAL_RESPONSE"
        response_start_tag_len = len(response_start_tag)
        found_split_point = False

        async for chunk in model.astream(messages):
            if hasattr(chunk, "content") and chunk.content:
                content = chunk.content

                if not found_split_point:
                    temp_content = reasoning_buffer + content
                    reasoning_buffer = ""

                    end_pos = temp_content.find(end_tag)
                    if end_pos != -1:
                        if end_pos > 0:
                            reasoning_data = temp_content[:end_pos]
                            accumulated_reasoning.append(reasoning_data)
                            yield json.dumps(
                                {
                                    "type": "reasoning",
                                    "data": reasoning_data,
                                    "phase": "synthesis",
                                }
                            ) + "\n"

                        remaining = temp_content[end_pos + end_tag_len :]
                        remaining = remaining.replace(response_start_tag, "")
                        if remaining:
                            final_response_chunks.append(remaining)
                            yield json.dumps(
                                {
                                    "type": "content",
                                    "data": remaining,
                                    "phase": "synthesis",
                                }
                            ) + "\n"

                        found_split_point = True
                        continue

                    response_start_pos = temp_content.find(response_start_tag)
                    if response_start_pos != -1:
                        if response_start_pos > 0:
                            reasoning_data = temp_content[:response_start_pos]
                            accumulated_reasoning.append(reasoning_data)
                            yield json.dumps(
                                {
                                    "type": "reasoning",
                                    "data": reasoning_data,
                                    "phase": "synthesis",
                                }
                            ) + "\n"

                        remaining = temp_content[
                            response_start_pos + response_start_tag_len :
                        ]
                        remaining = remaining.lstrip("\n\r\t ")
                        remaining = remaining.replace(response_start_tag, "")
                        if remaining:
                            final_response_chunks.append(remaining)
                            yield json.dumps(
                                {
                                    "type": "content",
                                    "data": remaining,
                                    "phase": "synthesis",
                                }
                            ) + "\n"

                        found_split_point = True
                        continue

                    potential_tag = False
                    if len(temp_content) >= end_tag_len - 1:
                        for i in range(1, end_tag_len):
                            suffix = temp_content[-i:]
                            if end_tag.startswith(suffix):
                                potential_tag = True
                                reasoning_buffer = temp_content[-(end_tag_len - 1) :]
                                output_reasoning = temp_content[: -(end_tag_len - 1)]
                                if output_reasoning:
                                    accumulated_reasoning.append(output_reasoning)
                                    yield json.dumps(
                                        {
                                            "type": "reasoning",
                                            "data": output_reasoning,
                                            "phase": "synthesis",
                                        }
                                    ) + "\n"
                                break

                    if (
                        not potential_tag
                        and len(temp_content) >= response_start_tag_len - 1
                    ):
                        for i in range(1, response_start_tag_len):
                            suffix = temp_content[-i:]
                            if response_start_tag.startswith(suffix):
                                potential_tag = True
                                reasoning_buffer = temp_content[
                                    -(response_start_tag_len - 1) :
                                ]
                                output_reasoning = temp_content[
                                    : -(response_start_tag_len - 1)
                                ]
                                if output_reasoning:
                                    accumulated_reasoning.append(output_reasoning)
                                    yield json.dumps(
                                        {
                                            "type": "reasoning",
                                            "data": output_reasoning,
                                            "phase": "synthesis",
                                        }
                                    ) + "\n"
                                break

                    if not potential_tag and temp_content:
                        accumulated_reasoning.append(temp_content)
                        yield json.dumps(
                            {
                                "type": "reasoning",
                                "data": temp_content,
                                "phase": "synthesis",
                            }
                        ) + "\n"
                else:
                    filtered_content = content.replace(response_start_tag, "")
                    if filtered_content:
                        final_response_chunks.append(filtered_content)
                        yield json.dumps(
                            {
                                "type": "content",
                                "data": filtered_content,
                                "phase": "synthesis",
                            }
                        ) + "\n"

        if reasoning_buffer.strip() and not found_split_point:
            accumulated_reasoning.append(reasoning_buffer)
            yield json.dumps(
                {
                    "type": "reasoning",
                    "data": reasoning_buffer,
                    "phase": "synthesis",
                }
            ) + "\n"

        final_response = "".join(final_response_chunks)
        if not final_response.strip():
            full_reasoning = "".join(accumulated_reasoning)
            if full_reasoning.strip():
                yield json.dumps(
                    {
                        "type": "content",
                        "data": full_reasoning,
                        "phase": "synthesis",
                    },
                    ensure_ascii=False,
                ) + "\n"

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

