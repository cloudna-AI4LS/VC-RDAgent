#!/usr/bin/env python3
"""
Autonomous Multi-Agent System with LangGraph - DashScope API

Same functionality as phenotype_to_disease_controller_langchain_stream_api.py,
using LangChain/LangGraph (InfoExtractionAgent, WorkflowAgent, EvaluationAgent, etc.).
Model calls use Alibaba DashScope API with reasoning_content (thinking) and content (reply) separated.

Reference:
  from openai import OpenAI
  client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
  completion = client.chat.completions.create(model=config["model"], messages=..., extra_body={"enable_thinking": True}, stream=True)
  # reasoning_content -> thinking, content -> full reply
"""
import asyncio
import os
import time
import json
import re
from typing import TypedDict, Annotated, Sequence, Literal, Optional, Dict, Any, List, Union, Callable
import requests
from openai import AsyncOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.messages.utils import convert_to_openai_messages
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

from transformers import AutoTokenizer

_tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-8B",
    trust_remote_code=True
)

# Global variables
_models = {}

# Single path: OpenAI-compatible streaming (_stream_dashscope_*); model/base_url from inference_config.json.

# MCP HTTP endpoint
MCP_ENDPOINT = os.environ.get("MCP_ENDPOINT", "http://localhost:3000/mcp/")
MCP_TIMEOUT = int(os.environ.get("MCP_TIMEOUT", "600"))
generate_diagnosis_prompt = None


def _get_model_config_from_file() -> Optional[Dict[str, Any]]:
    """Load model_config from chat-system/inference_config.json."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference_config.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("model_config", {})
    except (json.JSONDecodeError, OSError):
        return None


def _get_delta_field(delta, name: str):
    """Extract field from delta, supports extended fields like reasoning_content."""
    if delta is None:
        return None
    val = getattr(delta, name, None)
    if val is not None:
        return val
    if hasattr(delta, "model_extra") and delta.model_extra and name in delta.model_extra:
        return delta.model_extra[name]
    return None


def _get_choice_field(choice, name: str):
    """Extract field from stream choice (delta or top-level), for completeness of reasoning/content."""
    if choice is None:
        return None
    val = getattr(choice, name, None)
    if val is not None:
        return val
    if hasattr(choice, "model_extra") and choice.model_extra and name in choice.model_extra:
        return choice.model_extra[name]
    return None


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


class _ThinkStreamParser:
    """
    流式解析 content：检测到 <think> 时，将后续输出作为推理部分记录，直到 </think>，
    之后全部作为 content 记录。支持标签被 chunk 截断的情况。
    """
    __slots__ = ("_state", "_buf")
    STATE_BEFORE = "before_think"
    STATE_IN_THINK = "in_think"
    STATE_AFTER = "after_think"

    def __init__(self):
        self._state = self.STATE_BEFORE
        self._buf = ""

    def feed(self, chunk: str) -> List[tuple]:
        """
        喂入一段 content 文本，返回 [(type, data), ...]，type 为 "reasoning" 或 "content"，data 为对应片段。
        """
        if not chunk:
            return []
        self._buf += chunk
        out = []
        while self._buf:
            if self._state == self.STATE_BEFORE:
                i = self._buf.find(THINK_OPEN)
                if i == -1:
                    # 保留可能成为 <think> 前缀的尾部，避免跨 chunk 被截断
                    keep = len(THINK_OPEN) - 1
                    if len(self._buf) > keep:
                        emit = self._buf[:-keep] if keep else self._buf
                        self._buf = self._buf[-keep:] if keep else self._buf
                        if emit:
                            out.append(("content", emit))
                    break
                if i > 0:
                    out.append(("content", self._buf[:i]))
                self._buf = self._buf[i + len(THINK_OPEN):]
                self._state = self.STATE_IN_THINK
            elif self._state == self.STATE_IN_THINK:
                j = self._buf.find(THINK_CLOSE)
                if j == -1:
                    keep = len(THINK_CLOSE) - 1
                    if len(self._buf) > keep:
                        emit = self._buf[:-keep] if keep else self._buf
                        self._buf = self._buf[-keep:] if keep else self._buf
                        if emit:
                            out.append(("reasoning", emit))
                    break
                out.append(("reasoning", self._buf[:j]))
                self._buf = self._buf[j + len(THINK_CLOSE):]
                self._state = self.STATE_AFTER
            else:  # STATE_AFTER
                out.append(("content", self._buf))
                self._buf = ""
                break
        return out

    def flush(self) -> List[tuple]:
        """将剩余 buffer 按当前状态输出（reasoning 或 content）。"""
        out = []
        if self._buf:
            if self._state == self.STATE_IN_THINK:
                out.append(("reasoning", self._buf))
            else:
                out.append(("content", self._buf))
            self._buf = ""
        self._state = self.STATE_AFTER
        return out


def _extract_think_from_content(content: str) -> tuple[str, str]:
    """当 reasoning_content 为空且 content 内含 <think>...</think> 时，提取 think 作为 reasoning，剩余作为 content。
    返回 (reasoning_part, content_without_think)。若无可提取的 think，返回 ("", content)。
    """
    if not content or not isinstance(content, str):
        return ("", content or "")
    m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if not m:
        return ("", content)
    think_part = m.group(1).strip()
    content_without = re.sub(r"<think>.*?</think>", "", content, count=1, flags=re.DOTALL).strip()
    return (think_part, content_without)


async def _stream_dashscope_with_tools(
    messages: Sequence[BaseMessage],
    tools: List,
    agent_label: str = "Agent",
    stream_callback: Optional[Callable[[str, str], None]] = None,
) -> AIMessage:
    """
    DashScope API streaming (OpenAI-compatible client): real-time reasoning_content + content output.
    Used by InfoExtractionAgent, EvaluationAgent, WorkflowAgent, PromptTemplateAgent.
    If stream_callback is set, call stream_callback(chunk_type, data) for each "reasoning" or "content" chunk.
    """
    cfg = _get_model_config_from_file()
    if not cfg:
        raise ValueError("inference_config.json: missing model_config; set model, base_url, etc.")
    model_name = cfg.get("model")
    if not model_name:
        raise ValueError("inference_config.json: model_config.model is required (model name must come from config only).")
    api_key = cfg.get("api_key") or os.environ.get("DASHSCOPE_API_KEY", "EMPTY")
    base_url = (cfg.get("base_url") or "").rstrip("/")
    if not base_url:
        raise ValueError("inference_config.json: model_config.base_url is required.")

    oai_messages = convert_to_openai_messages(messages)
    oai_tools = [convert_to_openai_tool(t) for t in tools]

    reasoning_parts = []
    content_parts = []
    tool_calls_acc = {}
    is_answering = False
    think_parser = _ThinkStreamParser()
    has_reasoning_content = False  # 仅当 API 未返回 reasoning_content 时才对 content 做 <think>/</think> 解析

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=300.0)
    stream = await client.chat.completions.create(
        model=model_name,
        messages=oai_messages,
        tools=oai_tools,
        stream=True,
        extra_body={"enable_thinking": True},
    )

    print("\n" + "=" * 60 + f" [{agent_label}] reasoning_content (stream) " + "=" * 60, flush=True)
    last_chunk = None
    async for chunk in stream:
        if not chunk.choices:
            continue
        last_chunk = chunk
        choice = chunk.choices[0]
        delta = getattr(choice, "delta", None)

        rc = _get_delta_field(delta, "reasoning_content")
        if not rc:
            rc = _get_choice_field(choice, "reasoning_content")
        if rc:
            has_reasoning_content = True
            reasoning_parts.append(rc)
            if stream_callback:
                stream_callback("reasoning", rc)
            print(rc, end="", flush=True)

        c = _get_delta_field(delta, "content") if delta else None
        if not c:
            c = _get_choice_field(choice, "content")
        if c:
            if not is_answering:
                print("\n" + "=" * 20 + "Full reply" + "=" * 20 + f"\n    {agent_label}: ", end="", flush=True)
                is_answering = True
            if has_reasoning_content:
                content_parts.append(c)
                if stream_callback:
                    stream_callback("content", c)
                print(c, end="", flush=True)
            else:
                for typ, data in think_parser.feed(c):
                    if typ == "reasoning":
                        reasoning_parts.append(data)
                        if stream_callback:
                            stream_callback("reasoning", data)
                        print(data, end="", flush=True)
                    else:
                        content_parts.append(data)
                        if stream_callback:
                            stream_callback("content", data)
                        print(data, end="", flush=True)

        for tc in (delta.tool_calls or []) if delta and hasattr(delta, "tool_calls") else []:
            idx = getattr(tc, "index", 0)
            if idx not in tool_calls_acc:
                tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
            acc = tool_calls_acc[idx]
            if getattr(tc, "id", None):
                acc["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn:
                if getattr(fn, "name", None):
                    acc["name"] = fn.name
                if getattr(fn, "arguments", None):
                    acc["arguments"] = acc.get("arguments", "") + fn.arguments

    # 仅在未收到 reasoning_content 时刷新 think 解析器
    if not has_reasoning_content:
        for typ, data in think_parser.flush():
            if typ == "reasoning":
                reasoning_parts.append(data)
                if stream_callback:
                    stream_callback("reasoning", data)
                print(data, end="", flush=True)
            else:
                content_parts.append(data)
                if stream_callback:
                    stream_callback("content", data)
                print(data, end="", flush=True)

    # Last chunk may carry reasoning/content in message or other fields (ensure no tail lost)
    if last_chunk and last_chunk.choices:
        msg = getattr(last_chunk.choices[0], "message", None)
        if msg is not None:
            tail_rc = _get_choice_field(msg, "reasoning_content")
            if tail_rc and isinstance(tail_rc, str) and tail_rc.strip():
                reasoning_parts.append(tail_rc)
                if stream_callback:
                    stream_callback("reasoning", tail_rc)
                print(tail_rc, end="", flush=True)
            elif not has_reasoning_content and not content_parts:
                tail_content = getattr(msg, "content", None)
                if tail_content and isinstance(tail_content, str) and tail_content.strip():
                    for typ, data in think_parser.feed(tail_content):
                        if typ == "reasoning":
                            reasoning_parts.append(data)
                            if stream_callback:
                                stream_callback("reasoning", data)
                            print(data, end="", flush=True)
                        else:
                            content_parts.append(data)
                            if stream_callback:
                                stream_callback("content", data)
                            print(data, end="", flush=True)
                    for typ, data in think_parser.flush():
                        if typ == "reasoning":
                            reasoning_parts.append(data)
                            if stream_callback:
                                stream_callback("reasoning", data)
                            print(data, end="", flush=True)
                        else:
                            content_parts.append(data)
                            if stream_callback:
                                stream_callback("content", data)
                            print(data, end="", flush=True)

    if reasoning_parts or content_parts:
        print(flush=True)
    print("=" * 60 + " [End] " + "=" * 60 + "\n", flush=True)

    reasoning_full = "".join(reasoning_parts)
    content_full = "".join(content_parts)
    # Fallback: 当 reasoning_content 为空但 content 内含 <think>...</think> 时，从 content 提取为 reasoning
    if not reasoning_full and content_full:
        think_part, content_without = _extract_think_from_content(content_full)
        if think_part:
            reasoning_full = think_part
            content_full = content_without
    tool_calls = []
    for idx in sorted(tool_calls_acc.keys()):
        acc = tool_calls_acc[idx]
        args_str = acc.get("arguments") or "{}"
        try:
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            args = {}
        tool_calls.append({
            "id": acc.get("id", ""),
            "name": acc.get("name", ""),
            "args": args,
            "type": "tool_call",
        })
    return AIMessage(
        content=content_full,
        tool_calls=tool_calls,
        additional_kwargs={"reasoning_content": reasoning_full} if reasoning_full else {},
    )


async def _stream_dashscope_no_tools(
    messages: Sequence[BaseMessage],
    agent_label: str = "Synthesis",
    stream_callback: Optional[Callable[[str, str], None]] = None,
) -> str:
    """
    DashScope API streaming (OpenAI-compatible client, no tools): real-time reasoning_content + content.
    Used by synthesize_results. Returns concatenated content string.
    If stream_callback is set, call stream_callback(chunk_type, data) for each "reasoning" or "content" chunk.
    """
    cfg = _get_model_config_from_file()
    if not cfg:
        raise ValueError("inference_config.json: missing model_config; set model, base_url, etc.")
    model_name = cfg.get("model")
    if not model_name:
        raise ValueError("inference_config.json: model_config.model is required (model name must come from config only).")
    api_key = cfg.get("api_key") or os.environ.get("DASHSCOPE_API_KEY", "EMPTY")
    base_url = (cfg.get("base_url") or "").rstrip("/")
    if not base_url:
        raise ValueError("inference_config.json: model_config.base_url is required.")

    oai_messages = convert_to_openai_messages(messages)

    reasoning_parts = []
    content_parts = []
    is_answering = False
    think_parser = _ThinkStreamParser()
    has_reasoning_content = False  # 仅当 API 未返回 reasoning_content 时才对 content 做 <think>/</think> 解析

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=300.0)
    stream = await client.chat.completions.create(
        model=model_name,
        messages=oai_messages,
        stream=True,
        extra_body={"enable_thinking": True},
    )

    print("\n" + "=" * 60 + f" [{agent_label}] reasoning_content (stream) " + "=" * 60, flush=True)
    last_chunk = None
    async for chunk in stream:
        if not chunk.choices:
            continue
        last_chunk = chunk
        choice = chunk.choices[0]
        delta = getattr(choice, "delta", None)

        rc = _get_delta_field(delta, "reasoning_content")
        if not rc:
            rc = _get_choice_field(choice, "reasoning_content")
        if rc:
            has_reasoning_content = True
            reasoning_parts.append(rc)
            if stream_callback:
                stream_callback("reasoning", rc)
            print(rc, end="", flush=True)

        c = _get_delta_field(delta, "content") if delta else None
        if not c:
            c = _get_choice_field(choice, "content")
        if c:
            if not is_answering:
                print("\n" + "=" * 20 + "Full reply" + "=" * 20 + f"\n    {agent_label}: ", end="", flush=True)
                is_answering = True
            if has_reasoning_content:
                content_parts.append(c)
                if stream_callback:
                    stream_callback("content", c)
                print(c, end="", flush=True)
            else:
                for typ, data in think_parser.feed(c):
                    if typ == "reasoning":
                        reasoning_parts.append(data)
                        if stream_callback:
                            stream_callback("reasoning", data)
                        print(data, end="", flush=True)
                    else:
                        content_parts.append(data)
                        if stream_callback:
                            stream_callback("content", data)
                        print(data, end="", flush=True)

    if not has_reasoning_content:
        for typ, data in think_parser.flush():
            if typ == "reasoning":
                reasoning_parts.append(data)
                if stream_callback:
                    stream_callback("reasoning", data)
                print(data, end="", flush=True)
            else:
                content_parts.append(data)
                if stream_callback:
                    stream_callback("content", data)
                print(data, end="", flush=True)

    if last_chunk and last_chunk.choices:
        msg = getattr(last_chunk.choices[0], "message", None)
        if msg is not None:
            tail_rc = _get_choice_field(msg, "reasoning_content")
            if tail_rc and isinstance(tail_rc, str) and tail_rc.strip():
                reasoning_parts.append(tail_rc)
                if stream_callback:
                    stream_callback("reasoning", tail_rc)
                print(tail_rc, end="", flush=True)
            elif not has_reasoning_content and not content_parts:
                tail_content = getattr(msg, "content", None)
                if tail_content and isinstance(tail_content, str) and tail_content.strip():
                    for typ, data in think_parser.feed(tail_content):
                        if typ == "reasoning":
                            reasoning_parts.append(data)
                            if stream_callback:
                                stream_callback("reasoning", data)
                            print(data, end="", flush=True)
                        else:
                            content_parts.append(data)
                            if stream_callback:
                                stream_callback("content", data)
                            print(data, end="", flush=True)
                    for typ, data in think_parser.flush():
                        if typ == "reasoning":
                            reasoning_parts.append(data)
                            if stream_callback:
                                stream_callback("reasoning", data)
                            print(data, end="", flush=True)
                        else:
                            content_parts.append(data)
                            if stream_callback:
                                stream_callback("content", data)
                            print(data, end="", flush=True)

    reasoning_full = "".join(reasoning_parts)
    content_full = "".join(content_parts)
    # Fallback: 当 reasoning_content 为空但 content 内含 <think>...</think> 时，从 content 提取；返回不含 think 的 content
    if not reasoning_full and content_full:
        think_part, content_without = _extract_think_from_content(content_full)
        if think_part:
            content_full = content_without
    if reasoning_full or content_full:
        print(flush=True)
    print("=" * 60 + " [End] " + "=" * 60 + "\n", flush=True)
    return content_full


# Task type constants
TASK_TYPE_GENERAL_INQUIRY = "general_inquiry"
TASK_TYPE_PHENOTYPE_EXTRACTION = "phenotype_extraction"
TASK_TYPE_DISEASE_DIAGNOSIS = "disease_diagnosis"
TASK_TYPE_DISEASE_CASE_EXTRACTION = "disease_case_extraction"
TASK_TYPE_DISEASE_INFO_RETRIEVAL = "disease_information_retrieval"

# Task type definitions
TASK_TYPES = [
    {"task_type": TASK_TYPE_GENERAL_INQUIRY, "description": "General questions and inquiries"},
    {"task_type": TASK_TYPE_PHENOTYPE_EXTRACTION, "description": "Extracting phenotype, symptom, HPO IDs (e.g., HP:0000123, HP:0000124) from any input and mapping them to specific phenotypes and phenotype descriptions."},
    {"task_type": TASK_TYPE_DISEASE_DIAGNOSIS, "description": "Diagnose diseases, determine possible diseases based on symptoms."},
    {"task_type": TASK_TYPE_DISEASE_CASE_EXTRACTION, "description": "Extract phenotypes from text and generate disease cases using ensemble methods."},
    {"task_type": TASK_TYPE_DISEASE_INFO_RETRIEVAL, "description": "Extract disease names from input, normalize to standard disease names, and enrich with detailed information."}
]

# Conversation history token limit
MAX_TOKENS = 15000


def _call_mcp_tool(name: str, arguments: Dict[str, Any]) -> str:
    """Call MCP tool via HTTP API.
    
    Supports both JSON and SSE (Server-Sent Events) response formats.
    """
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": name,
            "arguments": arguments
        },
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    try:
        resp = requests.post(MCP_ENDPOINT, json=payload, headers=headers, timeout=MCP_TIMEOUT)
        resp.raise_for_status()
        
        # Parse response: support both JSON and SSE formats
        data = None
        text = resp.text.strip()
        
        # Method 1: Try direct JSON parse (--json-response mode)
        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError):
            # Method 2: Parse SSE format (default mode)
            # SSE format: event: message\ndata: {...}
            for line in text.split('\n'):
                if line.startswith('data: '):
                    json_str = line[6:]  # strip 'data: ' prefix
                    try:
                        data = json.loads(json_str)
                        break  # take first valid JSON
                    except json.JSONDecodeError:
                        continue
        
        if data is None:
            return f"Error calling MCP tool {name}: Could not parse response (neither JSON nor SSE format)"
        
        if "error" in data:
            return json.dumps(data["error"], ensure_ascii=False, indent=2)
        result = data.get("result")
        if isinstance(result, dict) and "content" in result:
            content = result["content"]
            if isinstance(content, list) and content and isinstance(content[0], dict):
                text = content[0].get("text")
                if text is not None:
                    return text
            return json.dumps(result, ensure_ascii=False, indent=2)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error calling MCP tool {name}: {e}"


# ============================================================================
# ATOMIC TOOLS (Direct MCP Server Calls)
# ============================================================================

@tool
def phenotype_extractor_tool(query: Union[str, list, dict]) -> str:
    """Phenotype Extractor Tool: Support extracting phenotype, symptom or HPO IDs (e.g., HP:0000123, HP:0000124) from user query and mapping them to specific phenotypes and phenotype descriptions.
    The input query must describe phenotypes belonging to a single, independent entity (e.g., one patient). Phenotype descriptions for different entities should not be combined in a single input.
    """
    
    try:
        if isinstance(query, list):
            query = " ".join(str(item) for item in query)
        
        # Convert input to string if it's a dictionary
        if isinstance(query, dict):
            # Convert all list values to strings
            for key, value in query.items():
                if isinstance(value, list):
                    query[key] = " ".join(str(item) for item in value)
            query = json.dumps(query, ensure_ascii=False)
        
        return _call_mcp_tool("phenotype-extractor", {"query": query})
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def disease_diagnosis_tool(query: Union[str, list, dict], extracted_phenotypes: dict=None, extracted_disease_cases: dict=None) -> str:
    """
    Disease Diagnosis Workflow: Diagnose diseases, determine possible diseases based on symptoms.
    This function automatically identify phenotypes information from query and performs step-by-step disease diagnosis.
    This tool should only be invoked when there is a clear request for disease diagnosis.
    """
    try:
        if isinstance(query, list):
            query = " ".join(str(item) for item in query)
        # Convert input to string if it's a dictionary
        if isinstance(query, dict):
            # Convert all list values to strings
            for key, value in query.items():
                if isinstance(value, list):
                    query[key] = " ".join(str(item) for item in value)
            query = json.dumps(query, ensure_ascii=False)

        return _call_mcp_tool(
            "disease-diagnosis",
            {
                "original_query": query,
                "extracted_phenotypes": extracted_phenotypes,
                "disease_cases": extracted_disease_cases,
            },
        )
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def disease_case_extractor_tool(query: Union[str, list, dict]) -> str:
    """
    Disease Case Extractor Tool: Extracts a single, coherent phenotype set from the query, where the phenotype set represents one independent clinical entity (e.g., a single patient), and generates disease cases based exclusively on this entity-level phenotype set.
    """
    try:
        if isinstance(query, list):
            query = " ".join(str(item) for item in query)
        
        # Convert input to string if it's a dictionary
        if isinstance(query, dict):
            # Convert all list values to strings
            for key, value in query.items():
                if isinstance(value, list):
                    query[key] = " ".join(str(item) for item in value)
            query = json.dumps(query, ensure_ascii=False)

        return _call_mcp_tool(
            "disease-case-extractor",
            {"query": query},
        )
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def disease_information_retrieval_tool(query: Union[str, list, dict]) -> str:
    """
    Disease Information Retrieval Tool: extract disease names from query, normalize to standard disease names,
    and enrich with detailed information.
    """
    try:
        if isinstance(query, list):
            query = " ".join(str(item) for item in query)
        
        # Convert input to string if it's a dictionary
        if isinstance(query, dict):
            # Convert all list values to strings
            for key, value in query.items():
                if isinstance(value, list):
                    query[key] = " ".join(str(item) for item in value)
            query = json.dumps(query, ensure_ascii=False)

        return _call_mcp_tool(
            "disease-information-retrieval",
            {"query": query},
        )
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# PROMPT TEMPLATE TOOLS: Tools that return prompts and required tools
# ============================================================================

@tool
def call_diagnosis_prompt_template(query: str) -> str:
    """
    Diagnosis Prompt Template: Returns a diagnostic prompt template and the list of required tools for disease diagnosis.
    Use this when the user wants to diagnose diseases based on symptoms or phenotypes.
    Returns JSON with 'prompt' and 'required_tools' fields.
    """
    try:
        if generate_diagnosis_prompt:
            result = generate_diagnosis_prompt()
            # Parse the result to add required_tools
            result_dict = json.loads(result)
            return json.dumps(result_dict, ensure_ascii=False, indent=2)
        return json.dumps({
            "error": "generate_diagnosis_prompt not available",
            "required_tools": []
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "error": f"Error: {str(e)}",
            "required_tools": []
        }, ensure_ascii=False)


async def get_model(
    name: str = "default",
    model: str = None,
    model_provider: str = None,
    base_url: str = None,
    api_key: str = None,
    temperature: float = None,
    top_p: float = None,
    streaming: bool = None,
    reasoning_effort: str = None,
    **kwargs
):
    """
    Get model instance. When using DashScope API:
    - base_url = https://dashscope.aliyuncs.com/compatible-mode/v1
    - model_kwargs = {"extra_body": {"enable_thinking": True}}
    - reasoning_content = thinking, content = full reply
    """
    global _models
    if name not in _models:
        file_cfg = _get_model_config_from_file() or {}
        model_kwargs = {
            "model": model or file_cfg.get("model"),
            "base_url": (base_url or file_cfg.get("base_url") or "").rstrip("/"),
            "api_key": api_key or file_cfg.get("api_key", "EMPTY"),
            "temperature": temperature if temperature is not None else file_cfg.get("temperature", 0.1),
            "top_p": top_p if top_p is not None else file_cfg.get("top_p", 0.95),
            "streaming": streaming if streaming is not None else file_cfg.get("streaming", True),
            "model_provider": "openai",
            "extra_body": {"enable_thinking": True},
        }
        if reasoning_effort is not None:
            model_kwargs["reasoning_effort"] = reasoning_effort
        model_kwargs.update(kwargs)
        _models[name] = init_chat_model(**model_kwargs)
    return _models[name]


def count_tokens(text: str) -> int:
    """
    Count tokens using the loaded Hugging Face tokenizer (no model inference).

    Notes:
    - Uses tokenizer vocab/encoding only; does not load or run the model.
    - Tokenizer identity should match the chat model (model_config.model in inference_config.json).
    - Deterministic: add_special_tokens=False for raw token count.
    """
    if not text:
        return 0
    return len(_tokenizer.encode(text, add_special_tokens=False))


def count_messages_tokens(messages: List[BaseMessage]) -> int:
    """
    Count total tokens in a list of messages
    
    Args:
        messages: List of BaseMessage objects
        
    Returns:
        int: Total number of tokens across all messages
    """
    total_tokens = 0
    for message in messages:
        # Get message content
        if hasattr(message, 'content'):
            content = message.content
            if content:
                total_tokens += count_tokens(str(content))
    return total_tokens

# ============================================================================
# TASK CLASSIFICATION: Determine task type from user input
# ============================================================================

async def classify_task_type(user_query: str, final_response: str, task_types: list = None) -> str:
    """Determine task type from user query and final response. Model must output valid JSON."""
    if task_types is None:
        task_types = TASK_TYPES
    
    classification_model = await get_model("task_classification")
    task_descriptions = json.dumps(task_types, ensure_ascii=False)
    classification_prompt = f"""/no_think
You are a task classification expert. Analyze the user query and final response to determine the most appropriate task type.

User Query: {user_query}

Final Response: {final_response}

Available task types:
{task_descriptions}

You MUST respond with a valid JSON object in exactly this format, no other text:
{{"task_type": "<one of the task_type values from the list above>"}}

Example: {{"task_type": "disease_diagnosis"}}"""
    
    try:
        response_chunks = []
        async for chunk in classification_model.astream([HumanMessage(content=classification_prompt)]):
            if hasattr(chunk, 'content') and chunk.content:
                response_chunks.append(chunk.content)
        result_text = "".join(response_chunks).strip()
        
        if not result_text:
            return task_types[0]["task_type"]
        
        # Extract JSON: support markdown code block wrapper
        json_str = result_text
        if "```" in result_text:
            for block in result_text.split("```"):
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                if block.startswith("{"):
                    json_str = block
                    break
        elif "{" in result_text and "}" in result_text:
            start = result_text.find("{")
            end = result_text.rfind("}") + 1
            json_str = result_text[start:end]
        
        data = json.loads(json_str)
        classified_type = data.get("task_type", "")
        if not isinstance(classified_type, str):
            classified_type = str(classified_type)
        
        available_types = [t["task_type"] for t in task_types]
        if classified_type in available_types:
            return classified_type
        # If parsed value not in list, try fuzzy match
        ct_lower = classified_type.lower()
        for tt in available_types:
            if tt in ct_lower or ct_lower == tt:
                return tt
        return task_types[0]["task_type"]
            
    except Exception as e:
        print(f"Task classification error: {e}")
        return task_types[0]["task_type"]


# ============================================================================
# INFO EXTRACTION AGENT: Autonomous agent that automatically calls tools
# ============================================================================

class InfoExtractionAgent:
    """
    Autonomous agent using LangGraph that automatically calls appropriate tools
    based on user input without manual task classification
    """
    def __init__(self, specified_tools: List[str] = None):
        """
        Initialize InfoExtractionAgent
        
        Args:
            specified_tools: Optional list of tool names to use. If provided, only these tools
                           will be available and must be called. If None, uses all available tools.
        """
        self.model = None
        
        # Default tools to use when not specified
        default_tool_names = [
            'phenotype_extractor_tool',
            'disease_case_extractor_tool',
            'disease_information_retrieval_tool'
        ]
        
        # Set tools based on specified_tools parameter
        if specified_tools:
            # Get tool functions from global scope by name
            self.tools = [globals()[tool_name] for tool_name in specified_tools 
                         if tool_name in globals()]
            self.specified_tools = self.tools
            if not self.tools:
                raise ValueError(f"No valid tools found in specified_tools: {specified_tools}")
        else:
            # Use default tools
            self.tools = [globals()[tool_name] for tool_name in default_tool_names]
            self.specified_tools = None
        
        self.graph = None
        
    async def _get_model(self):
        """Get or create model instance"""
        if self.model is None:
            self.model = await get_model(
                name="info_extraction_agent",
                reasoning_effort="low"  # optional: "low", "medium", "high"
            )
        return self.model
    
    async def _call_model(self, state: MessagesState):
        """Call the model with tools (with streaming)"""
        messages = state["messages"]
        # TODO: strip thinking from AIMessage in messages
        for i in range(len(messages)):
            if isinstance(messages[i], AIMessage):
                messages[i].content = re.sub(r'<think>.*?</think>', '', messages[i].content, flags=re.DOTALL)
                messages[i].content = re.sub(r'.*?</think>', '', messages[i].content, flags=re.DOTALL)
        # print(f"DEBUG: Messages: {messages}")
        
        # Different prompts based on whether tools are specified
        if self.specified_tools:
            # Must call specified tools
            task_instruction = f"""
Task target: Your task is to call the {', '.join(tool.name for tool in self.specified_tools)} tools to parse the user query or tool response and extract the relevant information, rather than directly answering the user's question. If no tools need to be called, please respond directly with "No tools needed", without any other reasoning or explanation.
"""
            messages += [HumanMessage(content=task_instruction)]
            
            system_prompt = SystemMessage(content=f"""\n
Task target: My task is to call the {', '.join(tool.name for tool in self.specified_tools)} tools to parse the user query or tool response and extract the relevant information, rather than directly answering the user's question. If no tools need to be called, I should reply directly with "No tools needed", without any other reasoning or explanation.

**Important:**
- When necessary, combine conversation history to reconstruct and understand the user's complete query context.
- Call ALL specified tools: {', '.join(tool.name for tool in self.specified_tools)}.
""")
        else:
            # Flexible tool selection
            task_instruction = """
Task target: Your task is to call the appropriate tools to parse the user query or tool response and extract the relevant information, rather than directly answering the user's question. If no tools need to be called, please respond directly with "No tools needed", without any other reasoning or explanation.
"""
            messages += [HumanMessage(content=task_instruction)]
            
            system_prompt = SystemMessage(content="""\n
Task target: My task is to call the appropriate tools to parse the user query or tool response and extract the relevant information, rather than directly answering the user's question. If no tools need to be called, I should reply directly with "No tools needed", without any other reasoning or explanation.

**Important:** 
- When necessary, combine conversation history to reconstruct and understand the user's complete query context.
- Call MULTIPLE tools when relevant, don't limit to just one.
""")
        
        # Add system message if it's the first call
        if len(messages) == 1 or not any(hasattr(m, 'type') and m.type == 'system' for m in messages):
            messages = [system_prompt] + list(messages)
        
        # print(f"DEBUG: Messages: {messages}")
        
        response = await _stream_dashscope_with_tools(
            messages, self.tools, agent_label="InfoExtractionAgent",
            stream_callback=getattr(self, "stream_callback", None),
        )
        return {"messages": [response]}
    
    def _should_continue(self, state: MessagesState) -> str:
        """Determine if should continue to tools or end"""
        messages = state["messages"]
        last_message = messages[-1]

        # If the last message is from AI and has tool calls, go to tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            print(f"  🤖 Last message is a tool call: {last_message.tool_calls}")
            return "tools"
        # If the last message is a tool result, go back to call_model to check if more tools needed
        elif hasattr(last_message, 'type') and last_message.type == 'tool':
            # print(f"DEBUG: Last message is a tool result: {last_message.content}")
            return "call_model"
        # Otherwise end (AI message without tool calls)
        return "__end__"
    
    async def _build_graph(self):
        """Build LangGraph workflow - multi-turn tool calling"""
        if self.graph is None:
            tool_node = ToolNode(self.tools)
            
            builder = StateGraph(MessagesState)
            builder.add_node("call_model", self._call_model)
            builder.add_node("tools", tool_node)
            
            builder.add_edge(START, "call_model")
            builder.add_conditional_edges("call_model", self._should_continue)
            builder.add_conditional_edges("tools", self._should_continue)
            
            self.graph = builder.compile()
        
        return self.graph
    
    async def run(self, user_query: str, conversation_messages: List[BaseMessage] = None, stream_callback: Optional[Callable[[str, str], None]] = None) -> Dict[str, str]:
        """
        Run the info extraction agent to automatically select and execute tools
        
        Args:
            user_query: User's query
            conversation_messages: Conversation history
            stream_callback: Optional callback(chunk_type, data) for real-time reasoning/content chunks.
            
        Returns:
            Dict mapping tool names to their results
        """
        self.stream_callback = stream_callback
        print("\n" + "="*80)
        print("🤖 INFO EXTRACTION AGENT: Autonomous Tool Selection & Execution")
        print("="*80)
        
        if self.specified_tools:
            print(f"⚙️  Mode: SPECIFIED TOOLS (must call all)")
            print(f"   Required Tools: {', '.join([tool.name for tool in self.specified_tools])}")
        else:
            print(f"⚙️  Mode: AUTO SELECTION (flexible)")
            available_tools = [tool.name for tool in self.tools]
            print(f"   Available Tools: {', '.join(available_tools)}")
        
        # Build graph
        graph = await self._build_graph()
        
        # Prepare messages
        messages = []
        if conversation_messages:
            messages.extend(conversation_messages)
        
        # user_query = f"User query: {user_query}\nTask target: Call the appropriate tools to parse the user query and extract the relevant information."
        messages.append(HumanMessage(content=user_query))
        
        # Run graph with streaming
        start_time = time.time()
        
        all_messages = []
        round_num = 0
        tool_call_count = 0
        
        async for chunk in graph.astream({"messages": messages}):
            # chunk is a dictionary with node names as keys
            for node_name, node_output in chunk.items():
                if node_name == "call_model":
                    # Check if this is a new round (AI message with tool calls)
                    if node_output and "messages" in node_output:
                        last_msg = node_output["messages"][-1]
                        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                            round_num += 1
                            tool_call_count += len(last_msg.tool_calls)
                            print(f"  🤖 Round {round_num}: Model selecting tools... ({len(last_msg.tool_calls)} tool(s))")
                        else:
                            print(f"  ✅ Round {round_num + 1}: No more tools needed - stopping")
                elif node_name == "tools":
                    print(f"  🔧 Round {round_num}: Executing tools...")
                # Accumulate all messages from each node output
                if node_output and "messages" in node_output:
                    all_messages.extend(node_output["messages"])
        
        elapsed = time.time() - start_time
        print(f"✓ Info extraction agent completed in {elapsed:.2f}s ({round_num} round(s), {tool_call_count} tool call(s))")
        
        # Extract tool results from all accumulated messages
        tool_results = {}
        for message in all_messages:
            if hasattr(message, 'type') and message.type == 'tool':
                tool_name = message.name
                
                # Accumulate results (keep all calls, use set to avoid duplicates)
                if tool_name not in tool_results:
                    tool_results[tool_name] = set()
                tool_results[tool_name].add(message.content)
                print(f"  ✓ {tool_name} (call {len(tool_results[tool_name])}): {message.content[:200]}...")
        
        # Convert set to list and merge results automatically
        for tool_name in tool_results:
            result_list = list(tool_results[tool_name])
            if len(result_list) == 1:
                tool_results[tool_name] = result_list[0]
            else:
                # Try to parse and merge all JSON results
                parsed_results = []
                all_valid_json = True
                
                for result_str in result_list:
                    try:
                        result_json = json.loads(result_str)
                        if isinstance(result_json, dict):
                            parsed_results.append(result_json)
                        else:
                            all_valid_json = False
                            break
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        all_valid_json = False
                        break
                
                # If all results are valid JSON dictionaries, merge them
                if all_valid_json and parsed_results:
                    merged_result = {}
                    
                    # Get all unique top-level keys from all results
                    all_top_keys = set()
                    for result_json in parsed_results:
                        all_top_keys.update(result_json.keys())
                    
                    # Merge each top-level key
                    for top_key in all_top_keys:
                        # Check if all results have this key and it's a dict
                        all_have_key = all(top_key in r for r in parsed_results)
                        all_are_dict = all(isinstance(r.get(top_key), dict) for r in parsed_results if top_key in r)
                        
                        if all_have_key and all_are_dict:
                            # Merge dictionaries: combine all key-value pairs
                            # If same sub_key exists, keep the first one (no similarity comparison)
                            merged_dict = {}
                            for result_json in parsed_results:
                                if top_key in result_json:
                                    for sub_key, sub_value in result_json[top_key].items():
                                        # Only add if sub_key doesn't exist yet (keep first occurrence)
                                        if sub_key not in merged_dict:
                                            merged_dict[sub_key] = sub_value
                            merged_result[top_key] = merged_dict
                        else:
                            # For non-dict values or keys not in all results, use the first occurrence
                            for result_json in parsed_results:
                                if top_key in result_json:
                                    merged_result[top_key] = result_json[top_key]
                                    break
                    
                    tool_results[tool_name] = json.dumps(merged_result, ensure_ascii=False, indent=2)
                else:
                    # Otherwise, keep as list
                    tool_results[tool_name] = result_list
        
        return tool_results


# ============================================================================
# EVALUATION AGENT: Evaluates final response quality using tools
# ============================================================================

class EvaluationAgent:
    """
    Evaluation agent that uses the same tools as InfoExtractionAgent to evaluate
    the quality and completeness of the final response
    """
    def __init__(self):
        """
        Initialize EvaluationAgent with the same tools as InfoExtractionAgent
        """
        # Use the same default tools as InfoExtractionAgent
        tool_names = [
            'phenotype_extractor_tool',
            'disease_case_extractor_tool',
            'disease_information_retrieval_tool'
        ]
        
        self.tools = [globals()[tool_name] for tool_name in tool_names]
        self.graph = None
    
    async def _call_model(self, state: MessagesState):
        """Call the model with tools for evaluation (with streaming)"""
        messages = state["messages"]

        # Check if first call: last message type
        last_message = messages[-1] if messages else None
        is_first_call = isinstance(last_message, HumanMessage)
        sys_msg_content = "/no_think" if is_first_call else "/think"
        messages = [SystemMessage(content=sys_msg_content)] + messages

        response = await _stream_dashscope_with_tools(
            messages, self.tools, agent_label="EvaluationAgent",
            stream_callback=getattr(self, "stream_callback", None),
        )
        return {"messages": [response]}
    
    def _should_continue(self, state: MessagesState) -> str:
        """Determine if should continue to tools or end"""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the last message is from AI and has tool calls, go to tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        # If the last message is a tool result, go back to call_model to check if more tools needed
        elif hasattr(last_message, 'type') and last_message.type == 'tool':
            return "call_model"
        # Otherwise end (AI message without tool calls)
        return "__end__"
    
    async def _build_graph(self):
        """Build LangGraph workflow - multi-turn tool calling"""
        if self.graph is None:
            tool_node = ToolNode(self.tools)
            
            builder = StateGraph(MessagesState)
            builder.add_node("call_model", self._call_model)
            builder.add_node("tools", tool_node)
            
            builder.add_edge(START, "call_model")
            builder.add_conditional_edges("call_model", self._should_continue)
            builder.add_conditional_edges("tools", self._should_continue)
            
            self.graph = builder.compile()
        
        return self.graph
    
    def _get_evaluation_template(self, task_type: str, user_query: str, 
                                  tool_results_summary: str, final_response: str) -> str:
        """
        Get evaluation template based on task type
        
        Args:
            task_type: Type of task (disease_diagnosis, phenotype_extraction, etc.)
            user_query: Original user query
            tool_results_summary: Summary of tool results
            final_response: Final response to evaluate
            
        Returns:
            str: Evaluation query template
        """
        query_template = ""
        # Disease Diagnosis Template (original detailed template)
        if task_type == TASK_TYPE_DISEASE_DIAGNOSIS:
            query_template = f"""Evaluate the quality and completeness of the following final response:

**Original User Query:**
{user_query}

**Available Knowledge:**
{tool_results_summary}

**Final Response to Evaluate:**
{final_response}

**Task Type:** Disease Diagnosis

**Evaluation Instructions:**
1. Call **disease_information_retrieval_tool** tool to get the detailed phenotypes information of the top [N, N<=5] candidate diseases in the **Final Response**.
2. Analyze the key phenotypes (frequently occurring phenotypes) of each candidate disease and match them with patient symptoms.
3. Analyze the following information for each candidate disease: **matched patient phenotypes**, **unmatched patient phenotypes**, **the key phenotypes in the disease but Not in the patient**.
4. Based on the above analysis, reorder and score (range 0~10, 10 is the best match) the candidate diseases by the matching degree to the patient.
5. A higher score indicates: (a) the matched patient phenotypes are more, and (b) the key phenotypes in the disease but not in the patient are fewer.
6. Evaluation Output Format:

    Matching Degree Analysis (in table markdown format):
    | Disease Name | Matched Patient Phenotypes | Unmatched Patient Phenotypes | the Key Symptoms in the Disease but Not in the Patient | Matching Degree |
    | --- | --- | --- | --- | --- |
    | Disease Name 1 | Matched Patient Phenotypes | Unmatched Patient Phenotypes | the Key Symptoms in the Disease but Not in the Patient | Matching Degree |
    | ... | ... | ... | ... | ... |
    | Disease Name N | Matched Patient Phenotypes | Unmatched Patient Phenotypes | the Key Symptoms in the Disease but Not in the Patient | Matching Degree |

    Suggestions:
    1. Symptoms that need further examination to aid differential diagnosis

"""

        return query_template
    
    async def run(self, user_query: str, all_tool_results: Dict[str, str], 
                  conversation_messages: List[BaseMessage], final_response: str,
                  task_type: str = TASK_TYPE_GENERAL_INQUIRY, stream_callback: Optional[Callable[[str, str], None]] = None) -> Dict[str, Any]:
        """
        Run the evaluation agent to assess final response quality
        
        Args:
            user_query: Original user query
            all_tool_results: Results from all previously executed tools
            conversation_messages: Conversation history
            final_response: The final synthesized response to evaluate
            task_type: Type of task for selecting appropriate evaluation template
            stream_callback: Optional callback(chunk_type, data) for real-time reasoning/content chunks.
            
        Returns:
            Dict with evaluation results including:
            - evaluation_tool_results: Results from evaluation tools (for reference data)
            - evaluation_response: Model's final evaluation output (the actual evaluation)
        """
        self.stream_callback = stream_callback
        print("\n" + "="*80)
        print(f"🔍 EVALUATION AGENT: Assessing Response Quality (Task: {task_type})")
        print("="*80)
        
        # Build graph
        graph = await self._build_graph()
        
        # Prepare evaluation prompt
        tool_results_summary = "\n".join([
            f"{tool_name}:\n{result}"
            for tool_name, result in all_tool_results.items()
        ])
        
        # Get task-specific evaluation template
        evaluation_query = self._get_evaluation_template(
            task_type=task_type,
            user_query=user_query,
            tool_results_summary=tool_results_summary,
            final_response=final_response
        )
        
        if not evaluation_query:
            return {
                "evaluation_tool_results": {},  # Tool outputs for reference
                "evaluation_response": "",  # Model's final evaluation output
            }
        
        # Prepare messages
        messages = []
        if conversation_messages:
            messages.extend(conversation_messages)
        messages.append(HumanMessage(content=evaluation_query))
        
        # Run graph with streaming
        print("🔄 Running evaluation agent...")
        start_time = time.time()
        
        all_messages = []
        round_num = 0
        tool_call_count = 0
        
        async for chunk in graph.astream({"messages": messages}):
            # chunk is a dictionary with node names as keys
            for node_name, node_output in chunk.items():
                if node_name == "call_model":
                    # Check if this is a new round (AI message with tool calls)
                    if node_output and "messages" in node_output:
                        last_msg = node_output["messages"][-1]
                        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                            round_num += 1
                            tool_call_count += len(last_msg.tool_calls)
                            print(f"  🔍 Evaluation Round {round_num}: Calling verification tools... ({len(last_msg.tool_calls)} tool(s))")
                        else:
                            print(f"  ✅ Evaluation Round {round_num + 1}: No more verification needed - stopping")
                elif node_name == "tools":
                    print(f"  🔧 Evaluation Round {round_num}: Executing verification tools...")
                # Accumulate all messages from each node output
                if node_output and "messages" in node_output:
                    all_messages.extend(node_output["messages"])
        
        elapsed = time.time() - start_time

        # print(f"DEBUG: evaluation_query: {evaluation_query}")
        print(f"✓ Evaluation agent completed in {elapsed:.2f}s ({round_num} round(s), {tool_call_count} tool call(s))")
        
        # Extract tool results from all accumulated messages (for reference data)
        evaluation_tool_results = {}
        for message in all_messages:
            if hasattr(message, 'type') and message.type == 'tool':
                tool_name = message.name
                
                # Accumulate results
                if tool_name not in evaluation_tool_results:
                    evaluation_tool_results[tool_name] = set()
                evaluation_tool_results[tool_name].add(message.content)
                print(f"  ✓ Verification: {tool_name} (call {len(evaluation_tool_results[tool_name])})")
        
        # Convert set to list and flatten single-call results
        for tool_name in evaluation_tool_results:
            result_list = list(evaluation_tool_results[tool_name])
            if len(result_list) == 1:
                evaluation_tool_results[tool_name] = result_list[0]
            else:
                evaluation_tool_results[tool_name] = result_list
        
        # Extract final evaluation response from model (the last AI message without tool calls)
        evaluation_response = ""
        for message in reversed(all_messages):
            # Find the last AI message that doesn't have tool calls (final evaluation output)
            if (hasattr(message, 'type') and message.type == 'ai') or isinstance(message, AIMessage):
                if not (hasattr(message, 'tool_calls') and message.tool_calls):
                    evaluation_response = message.content if hasattr(message, 'content') else str(message)
                    break
        
        # If no evaluation response found, use empty string
        if not evaluation_response:
            evaluation_response = "No evaluation response generated"
        
        # print(f"\n📋 Final Evaluation Response:")
        # print("-" * 80)
        # print(evaluation_response)
        # print("-" * 80)
        
        return {
            "evaluation_tool_results": evaluation_tool_results,  # Tool outputs for reference
            "evaluation_response": evaluation_response,  # Model's final evaluation output
        }

# ============================================================================
# WORKFLOW AGENT: LLM decides whether to execute workflow
# ============================================================================

class WorkflowAgent:
    """
    Simple workflow agent: LLM decides whether to call workflow, execute with injected params
    """
    def __init__(self):
        self.workflow_tools = [disease_diagnosis_tool]
        self.tool_results = {}
        self.user_query = ""
        self.model_with_tools = None
        self.status_callback = None  # status callback
    
    async def _call_model(self, state: MessagesState):
        """LLM decides whether to call tool (with streaming)"""
        messages = state["messages"]
        status_sent = False

        response = await _stream_dashscope_with_tools(
            messages, self.workflow_tools, agent_label="WorkflowAgent",
            stream_callback=getattr(self, "stream_callback", None),
        )

        # Fallback: send status_callback when response has tool_calls (e.g. not sent during stream)
        if not status_sent and self.status_callback and hasattr(response, 'tool_calls') and response.tool_calls:
            # Send "Completed evaluation"
            self.status_callback("Completed evaluation")
            # Then send "Executing XXX"
            tool_name = response.tool_calls[0].get('name', 'workflow tool')
            self.status_callback(f"Executing {tool_name}...")
        
        return {"messages": [response]}
    
    def _route(self, state: MessagesState) -> Literal["execute", END]:
        """Determine if should execute tool or end"""
        last = state["messages"][-1]
        if hasattr(last, 'tool_calls') and last.tool_calls:
            print(f"    🔧 Routing to execute: {len(last.tool_calls)} tool call(s) detected")
            # Fallback if _call_model did not send status (tool_calls may appear only after merge)
            if self.status_callback and last.tool_calls:
                # Send "Completed evaluation"
                self.status_callback("Completed evaluation")
                # Then send "Executing XXX"
                tool_name = last.tool_calls[0].get('name', 'workflow tool')
                self.status_callback(f"Executing {tool_name}...")
            return "execute"
        else:
            return END
    
    async def _execute_tool(self, state: MessagesState):
        """Execute workflow tool with injected tool results"""
        
        outputs = []
        last_msg = state["messages"][-1]
        
        for tool_call in last_msg.tool_calls:
            tool_name = tool_call['name']
            args = tool_call['args']
            
            # Inject tool results into args
            if tool_name == 'disease_diagnosis_tool':
                # Convert JSON strings to dictionaries
                # phenotype_result = self.tool_results.get('phenotype_extractor_tool', {})
                case_result = self.tool_results.get('disease_case_extractor_tool', {})
                
                # # Parse JSON strings if needed
                # if isinstance(phenotype_result, str):
                #     try:
                #         phenotype_result = json.loads(phenotype_result)
                #         if 'extracted_phenotypes' in phenotype_result:
                #             phenotype_result = phenotype_result['extracted_phenotypes']
                #         else:
                #             phenotype_result = {}
                #     except json.JSONDecodeError:
                #         phenotype_result = {}
                
                if isinstance(case_result, str):
                    try:
                        case_result = json.loads(case_result)
                        if 'extracted_disease_cases' in case_result:
                            case_result = case_result['extracted_disease_cases']
                        else:
                            case_result = {}
                    except json.JSONDecodeError:
                        case_result = {}
                
                # If required params are empty: return prompt with tool names, end agent.
                # Controller gets feedback, calls InfoExtractionAgent for missing tools, then retries.
                # if not phenotype_result or not case_result:
                if not case_result:
                    # Determine missing tools
                    missing_tools = []
                    # if not phenotype_result:
                    #     missing_tools.append('phenotype_extractor_tool')
                    if not case_result:
                        missing_tools.append('disease_case_extractor_tool')
                    
                    # Return error with missing tool info
                    error_message = json.dumps({
                        "status": "MISSING_REQUIRED_TOOLS",
                        "missing_tools": missing_tools,
                    }, ensure_ascii=False)
                    
                    outputs.append(ToolMessage(
                        content=error_message, 
                        name=tool_name, 
                        tool_call_id=tool_call['id']
                    ))
                    break
                
                # Find common phenotype id keys in phenotype_result and case_result (dicts keyed by id)
                # symptom_sets_in_extracted_phenotypes = set(phenotype_result.keys()) if isinstance(phenotype_result, dict) else set()
                symptom_sets_in_extracted_cases = set(case_result.keys()) if isinstance(case_result, dict) else set()
                # common_ids = symptom_sets_in_extracted_phenotypes & symptom_sets_in_extracted_cases
                common_ids = symptom_sets_in_extracted_cases
                
                if not common_ids:
                    # If no common phenotype set, return error
                    error_message = json.dumps({
                        "status": "NO_COMMON_PHENOTYPE_IDS",
                        "message": "No common symptom sets found between phenotype_result and case_result",
                    }, ensure_ascii=False)
                    outputs.append(ToolMessage(
                        content=error_message,
                        name=tool_name,
                        tool_call_id=tool_call['id']
                    ))
                    break
                
                # For each common id, extract values and run workflow
                tool_func = next((t for t in self.workflow_tools if t.name == tool_name), None)
                
                if tool_func:
                    # Dict to store result per id_key
                    results_by_id = {}
                    
                    for id_key in common_ids:
                        # Independent args copy per id
                        id_args = args.copy()
                        # id_args['extracted_phenotypes'] = phenotype_result[id_key]
                        id_args['extracted_phenotypes'] = case_result[id_key]['extracted_phenotypes']
                        id_args['extracted_disease_cases'] = case_result[id_key]['disease_cases']
                        # Inject user query if not present
                        if 'query' not in id_args:
                            id_args['query'] = self.user_query
                        
                        try:
                            result = tool_func.invoke(id_args)
                            # Store result in dict by id_key
                            results_by_id[id_key] = result
                            print(f"  ✓ Executed: {tool_name} for phenotype ID: {id_key}")
                        except Exception as e:
                            # Store error in dict too
                            results_by_id[id_key] = {"error": str(e)}
                            print(f"  ❌ Error for phenotype ID {id_key}: {e}")
                    
                    # Add results organized by id_key to outputs
                    outputs.append(ToolMessage(
                        content=json.dumps(results_by_id, ensure_ascii=False, default=str),
                        name=tool_name,
                        tool_call_id=tool_call['id']
                    ))
                else:
                    outputs.append(ToolMessage(
                        content=f"Tool {tool_name} not found",
                        name=tool_name,
                        tool_call_id=tool_call['id']
                    ))
            else:
                # For non-disease_diagnosis_tool, keep original logic
                # Find and execute tool
                tool_func = next((t for t in self.workflow_tools if t.name == tool_name), None)
                
                if tool_func:
                    try:
                        result = tool_func.invoke(args)
                        outputs.append(ToolMessage(content=str(result), name=tool_name, tool_call_id=tool_call['id']))
                        print(f"  ✓ Executed: {tool_name}")
                    except Exception as e:
                        outputs.append(ToolMessage(content=f"Error: {e}", name=tool_name, tool_call_id=tool_call['id']))
                        print(f"  ❌ Error: {e}")
                else:
                    outputs.append(ToolMessage(content=f"Tool {tool_name} not found", name=tool_name, tool_call_id=tool_call['id']))
        
        return {"messages": outputs}
    
    async def run(self, user_query: str, tool_results: Dict[str, Any] = None, conversation_messages: List[BaseMessage] = None, status_callback: Optional[Callable[[str], None]] = None, stream_callback: Optional[Callable[[str, str], None]] = None) -> Dict[str, Any]:
        """
        LLM decides whether to call workflow, execute if needed
        
        Args:
            user_query: User's query
            tool_results: Results from previous tool calls
            conversation_messages: Conversation history
            status_callback: Optional callback function(status_message: str) to notify status changes
            stream_callback: Optional callback(chunk_type, data) for real-time reasoning/content chunks.
        """
        print("\n" + "="*80)
        print("🔄 WORKFLOW AGENT")
        print("="*80)
        
        self.tool_results = tool_results or {}
        self.user_query = user_query
        self.status_callback = status_callback
        self.stream_callback = stream_callback
        # Tool results from Info Extraction Agent, injected when running disease_diagnosis_tool
        print(f"📊 Tool results from previous phase (for injection): {list(self.tool_results.keys())}")
        
        # Get model with tools
        model = await get_model("workflow_agent")
        self.model_with_tools = model.bind_tools(self.workflow_tools)
        
        # Build graph
        graph = StateGraph(MessagesState)
        graph.add_node("agent", self._call_model)
        graph.add_node("execute", self._execute_tool)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", self._route)
        graph.add_edge("execute", END)
        app = graph.compile()
        
        # Build workflow descriptions
        workflow_descriptions = []
        for tool in self.workflow_tools:
            desc = tool.description if hasattr(tool, 'description') else "No description available"
            workflow_descriptions.append(f"- {tool.name}: {desc}")
        workflow_info = "\n".join(workflow_descriptions)
        
        task_target = f"""
Decide whether to call workflow tool based on user query.

User Query: 
{user_query}

Available workflows:
{workflow_info}

"""
        
        messages = []
        if conversation_messages:
            messages.extend(conversation_messages)
        messages.append(HumanMessage(content=task_target))
        
        # Run
        start = time.time()
        
        workflow_name = None
        workflow_result = None
        missing_tools = None
        tool_call_detected = False
        
        async for chunk in app.astream({"messages": messages}):
            for node, output in chunk.items():
                # Detect tool call (status callback in _call_model and _route)
                if node == "agent" and output and "messages" in output:
                    for msg in output["messages"]:
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            tool_call_detected = True
                            break
                
                # Detect tool execution complete
                if output and "messages" in output:
                    for msg in output["messages"]:
                        if hasattr(msg, 'type') and msg.type == 'tool':
                            workflow_name = msg.name
                            workflow_result = msg.content
                            
                            # Tool done, send status
                            if self.status_callback and workflow_name:
                                self.status_callback(f"Completed {workflow_name}")
                            
                            # Check for missing tools
                            try:
                                result_data = json.loads(workflow_result)
                                if isinstance(result_data, dict) and result_data.get('status') == 'MISSING_REQUIRED_TOOLS':
                                    missing_tools = result_data.get('missing_tools', [])
                            except (json.JSONDecodeError, TypeError):
                                pass  # not JSON, skip
        
        elapsed = time.time() - start
        
        # If no tool call, notify via callback
        if not tool_call_detected and self.status_callback:
            self.status_callback("No workflow tool needed")
        
        if workflow_name:
            print(f"✓ Executed: {workflow_name} ({elapsed:.2f}s)")
        else:
            print(f"✓ No workflow needed ({elapsed:.2f}s)")
        
        return {
            "workflow": workflow_name,
            "result": workflow_result,
            "missing_tools": missing_tools,
        }


# ============================================================================
# PROMPT TEMPLATE AGENT: Selects appropriate prompt template
# ============================================================================

class PromptTemplateAgent:
    """
    Autonomous agent using LangGraph that automatically selects appropriate prompt template
    based on user input. Returns prompt and required tools list.
    """
    def __init__(self):
        # Define available prompt template tools
        self.prompt_tools = [
            call_diagnosis_prompt_template,
        ]
    
    async def run(self, user_query: str) -> Dict[str, Any]:
        """
        Run the prompt template agent to automatically select appropriate template
        
        Args:
            user_query: User's input query
            
        Returns:
            Dict with 'prompt', 'required_tools', and 'template_name' fields
        """
        print("\n" + "="*80)
        print("📋 PROMPT TEMPLATE AGENT: Selecting Template")
        print("="*80)
        
        # Get model
        model = await get_model("prompt_template_selector")
        
        # Bind tools to model
        model_with_tools = model.bind_tools(self.prompt_tools)
        
        # Define workflow state
        class State(TypedDict):
            messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
        
        # Define the function that determines whether to continue or not
        def should_continue(state: State) -> Literal["tools", END]:
            messages = state['messages']
            last_message = messages[-1]
            # If the model makes a tool call, route to "tools" node
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            # Otherwise we finish (END)
            return END
        
        # Define the function that calls the model (with streaming)
        async def call_model(state: State):
            messages = state['messages']
            response = await _stream_dashscope_with_tools(
                list(messages), self.prompt_tools, agent_label="PromptTemplateAgent",
            )
            return {"messages": [response]}
        
        # Build the graph
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self.prompt_tools))
        
        # Set entry point
        workflow.add_edge(START, "agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            should_continue,
        )
        
        # Add edge from tools back to agent
        workflow.add_edge("tools", "agent")
        
        # Compile graph
        app = workflow.compile()
        
        # Prepare initial message
        initial_message = HumanMessage(content=f"""Based on the following user query, select the most appropriate prompt template.

User Query: {user_query}

Select the template that best matches the user's needs and call the corresponding tool to get the prompt template.""")
        
        # Run graph with streaming
        print("🔄 Selecting prompt template...")
        start_time = time.time()
        
        all_messages = []
        template_result = None
        
        async for output in app.astream({"messages": [initial_message]}, stream_mode="updates"):
            for node_name, node_output in output.items():
                # Accumulate all messages
                if node_output and "messages" in node_output:
                    all_messages.extend(node_output["messages"])
        
        elapsed = time.time() - start_time
        
        # Extract template result from tool messages
        for message in all_messages:
            if hasattr(message, 'type') and message.type == 'tool':
                try:
                    template_result = json.loads(message.content)
                    template_name = message.name
                    print(f"✓ Template selected: {template_name} in {elapsed:.2f}s")
                    break
                except json.JSONDecodeError:
                    continue
        
        if not template_result:
            # Fallback: if no template selected, return default
            print("⚠️  No template selected, using default diagnosis template")
            template_result = {
                "prompt_template": "",
                "required_tools": []
            }
            template_name = "default"
        
        return {
            "template_name": template_name if template_result else "default",
            "prompt": template_result.get("prompt_template", "") if template_result else "",
            "required_tools": template_result.get("required_tools", []) if template_result else []
        }


# ============================================================================
# SYNTHESIZER: Combine results and generate response
# ============================================================================

async def synthesize_results(
    user_query: str,
    tool_results: Dict[str, str],
    conversation_messages: List[BaseMessage] = None,
    stream_callback: Optional[Callable[[str, str], None]] = None,
) -> str:
    """
    Phase 3: Synthesize tool results into final response.
    If stream_callback is set, call stream_callback(chunk_type, data) for each "reasoning" or "content" chunk.
    """
    print("\n" + "="*80)
    print("🎨 PHASE 3: SYNTHESIS")
    print("="*80)
    
    # print(f"Tool Results: {tool_results}")

    # Build context from tool results
    results_text = "\n\n".join([
        f"**{tool_name.upper()} Results:**\n{result}"
        for tool_name, result in tool_results.items()
    ])
    
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

    if "disease_diagnosis_tool" in tool_results:
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

    print("\n📋 Final Response:")
    print("-" * 80)
    final_response = await _stream_dashscope_no_tools(
        messages, agent_label="Synthesis", stream_callback=stream_callback
    )
    print("\n" + "-" * 80)
    return final_response


# ============================================================================
# CONTROLLER: Main orchestrator
# ============================================================================

async def controller_pipeline(user_query: str, conversation_messages: List[BaseMessage] = None) -> str:
    """
    Main controller pipeline; logic aligned with controller_pipeline_stream_en (web_ui_api_en).
    1. InfoExtractionAgent
    2. WorkflowAgent (with missing_tools retry)
    3. Task classification (before evaluation/synthesis)
    4. If disease_diagnosis but workflow not run → recursive call with enhanced query
    5. Evaluation (only for disease_diagnosis, using workflow result)
    6. Synthesis via synthesize_results() module call
    """
    if conversation_messages is None:
        conversation_messages = []

    try:
        completed_tools: List[str] = []
        tool_call_counts: Dict[str, int] = {}
        all_tool_results: Dict[str, Any] = {}

        # Phase 1: InfoExtractionAgent
        info_extraction_agent = InfoExtractionAgent()
        info_extraction_results = await info_extraction_agent.run(user_query, conversation_messages)

        for tool_name in info_extraction_results.keys():
            completed_tools.append(tool_name)
            tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
        all_tool_results.update(info_extraction_results)

        # Phase 2: WorkflowAgent
        workflow_agent = WorkflowAgent()
        workflow_info = await workflow_agent.run(
            user_query,
            tool_results=all_tool_results,
            conversation_messages=conversation_messages,
            status_callback=None,
        )

        if workflow_info.get("missing_tools"):
            missing_tools = workflow_info["missing_tools"]
            retry_agent = InfoExtractionAgent(specified_tools=missing_tools)
            retry_results = await retry_agent.run(user_query, conversation_messages)

            for tool_name, result in retry_results.items():
                all_tool_results[tool_name] = result
                if tool_name not in completed_tools:
                    completed_tools.append(tool_name)
                tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1

            workflow_agent = WorkflowAgent()
            workflow_info = await workflow_agent.run(
                user_query,
                tool_results=all_tool_results,
                conversation_messages=conversation_messages,
                status_callback=None,
            )

        if workflow_info.get("workflow") is not None:
            all_tool_results[workflow_info["workflow"]] = workflow_info["result"]
            completed_tools.append(workflow_info["workflow"])
            tool_call_counts[workflow_info["workflow"]] = (
                tool_call_counts.get(workflow_info["workflow"], 0) + 1
            )

        # Phase 3: Task classification (before evaluation/synthesis)
        if workflow_info.get("workflow") == "disease_diagnosis_tool":
            task_type = "disease_diagnosis"
        else:
            workflow_result_text = str(workflow_info.get("result", "") or "")
            task_type = await classify_task_type(user_query, workflow_result_text)

        if task_type == "disease_diagnosis" and workflow_info.get("workflow") is None:
            enhanced_query = f"""{user_query}

**IMPORTANT: This query requires disease diagnosis. You MUST call the disease_case_extractor_tool to obtain disease cases, and MUST call the disease_diagnosis_tool to perform the diagnosis analysis.**"""
            return await controller_pipeline(enhanced_query, conversation_messages)

        # Phase 4: Evaluation (only for disease_diagnosis; use workflow result as final_response for eval)
        evaluation_response_clean = ""
        if task_type == "disease_diagnosis":
            evaluation_agent = EvaluationAgent()
            workflow_result_for_eval = str(workflow_info.get("result", "") or "")
            evaluation_results = await evaluation_agent.run(
                user_query=user_query,
                all_tool_results=all_tool_results,
                conversation_messages=conversation_messages,
                final_response=workflow_result_for_eval,
                task_type=task_type,
            )
            evaluation_response = evaluation_results.get("evaluation_response", "") or ""
            evaluation_response_clean = re.sub(
                r"<think>.*?</think>", "", evaluation_response, flags=re.DOTALL
            )
            evaluation_response_clean = re.sub(
                r".*?</think>", "", evaluation_response_clean, flags=re.DOTALL
            ).strip()
            if evaluation_response_clean:
                all_tool_results["diagnostic_evaluation"] = evaluation_response_clean

        # Phase 5: Synthesis (call module; do not inline)
        final_response = await synthesize_results(
            user_query, all_tool_results, conversation_messages
        )
        final_response = re.sub(r"<think>.*?</think>", "", final_response, flags=re.DOTALL)
        final_response = re.sub(r".*?</think>", "", final_response, flags=re.DOTALL)

        return final_response

    except Exception as e:
        print(f"\n❌ Pipeline Error: {e}")
        import traceback
        traceback.print_exc()
        return f"An error occurred: {str(e)}"


# ============================================================================
# MAIN INTERACTION LOOP
# ============================================================================

async def main():
    """Main interactive loop"""
    
    print("=" * 80)
    print("🤖 Autonomous Multi-Agent System with LangGraph (DashScope)")
    print("=" * 80)
    print("Architecture:")
    print("  1. 🤖 Info Extraction Agent: Autonomous agent that automatically selects and calls tools")
    print("  2. 🎨 Synthesizer: Combines all results into comprehensive final response")
    print("  3. 🔍 Evaluation Agent: Uses tools to verify and evaluate final response quality")
    print("  4. 📊 Result Analyzer: Determines if additional information is needed")
    print()
    print("Key Features:")
    print("  ✓ Autonomous tool selection (LangGraph-based)")
    print("  ✓ Multi-turn tool calling with state management")
    print("  ✓ DashScope API with enable_thinking (reasoning_content + content)")
    print("  ✓ Tool-based response evaluation and verification")
    print()
    print("Commands:")
    print("  'quit' - exit")
    print("  'clear' - clear conversation history")
    print("=" * 80)
    print()
    
    # Use LangChain messages for conversation history
    conversation_messages = []
    
    while True:
        user_input = input("\nUser: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'clear':
            conversation_messages = []
            print("✓ Conversation history cleared!")
            continue
        elif not user_input:
            continue
        
        # Add user message to history
        conversation_messages.append(HumanMessage(content=user_input))
        
        # Run controller pipeline with conversation history
        response = await controller_pipeline(user_input, conversation_messages)

        # print(f"DEBUG: response: {response}")

        # Add assistant response to history
        conversation_messages.append(AIMessage(content=response))
        
        # Trim conversation history if total tokens exceed limit
        total_tokens = count_messages_tokens(conversation_messages)
        print(f"DEBUG: total_tokens: {total_tokens}")
        # print(f"DEBUG: conversation_messages: {conversation_messages}")
        if total_tokens > MAX_TOKENS:
            print(f"⚠️  Total tokens ({total_tokens}) exceed limit ({MAX_TOKENS}), trimming conversation history...")
            removed_count = 0
            while total_tokens > MAX_TOKENS and len(conversation_messages) > 0:
                # Remove the first message (oldest)
                removed_message = conversation_messages.pop(0)
                removed_count += 1
                # Recalculate total tokens
                total_tokens = count_messages_tokens(conversation_messages)
            print(f"✓ Removed {removed_count} message(s), current total tokens: {total_tokens}")
        
        print()


def interactive_chat():
    """Synchronous wrapper"""
    asyncio.run(main())


if __name__ == "__main__":
    interactive_chat()
