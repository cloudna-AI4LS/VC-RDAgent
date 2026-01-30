#!/usr/bin/env python3
"""
Autonomous Multi-Agent System with LangGraph
自主多智能体系统：工具Agent → 综合结果 → 评估响应质量

Architecture:
1. 🤖 Info Extraction Agent: Autonomous agent using LangGraph for tool orchestration
   - Automatic tool selection based on user input
   - Multi-turn tool calling with state management
   - Self-directed workflow
2. 🎨 Synthesizer: Combines tool results into comprehensive final response
3. 🔍 Evaluation Agent: Uses tools to verify and evaluate final response quality
   - Same tools as Info Extraction Agent
   - Validates completeness and accuracy
   - Tool-based verification of key facts
4. 📊 Result Analyzer: Determines if additional information is needed

Key Features:
- LangGraph-based autonomous tool calling
- State-managed multi-turn tool execution
- Automatic tool routing
- Tool-based response evaluation
- Post-synthesis quality analysis
- Conversation memory support
"""
import asyncio
import os
import time
import json
import re
from typing import TypedDict, Annotated, Sequence, Literal, Optional, Dict, Any, List, Union, Callable
import requests
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage, SystemMessage
from pydantic import BaseModel, Field
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

# MCP HTTP endpoint
MCP_ENDPOINT = os.environ.get("MCP_ENDPOINT", "http://localhost:3000/mcp/")
# MCP request timeout in seconds (default: 600 seconds = 10 minutes)
MCP_TIMEOUT = int(os.environ.get("MCP_TIMEOUT", "600"))
# Placeholder for optional prompt generator (not available via MCP)
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
        
        # 尝试解析响应：支持 JSON 和 SSE 两种格式
        data = None
        text = resp.text.strip()
        
        # 方法1: 尝试直接解析为 JSON（--json-response 模式）
        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError):
            # 方法2: 解析 SSE 格式（默认模式）
            # SSE 格式: event: message\ndata: {...}
            for line in text.split('\n'):
                if line.startswith('data: '):
                    json_str = line[6:]  # 去掉 'data: ' 前缀
                    try:
                        data = json.loads(json_str)
                        break  # 取第一个有效的 JSON
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
    Get model instance. Parameters default to inference_config.json model_config.
    """
    global _models
    if name not in _models:
        file_cfg = _get_model_config_from_file() or {}
        model_provider_val = (model_provider if model_provider is not None else file_cfg.get("model_provider", "") or "").strip()
        model_kwargs = {
            "model": model if model is not None else file_cfg.get("model", "Qwen/Qwen3-8B"),
            "base_url": base_url if base_url is not None else file_cfg.get("base_url", "http://192.168.0.127:8000/v1"),
            "api_key": api_key if api_key is not None else file_cfg.get("api_key", "EMPTY"),
            "temperature": temperature if temperature is not None else file_cfg.get("temperature", 0.1),
            "top_p": top_p if top_p is not None else file_cfg.get("top_p", 0.95),
            "streaming": streaming if streaming is not None else file_cfg.get("streaming", True),
        }
        if model_provider_val:
            model_kwargs["model_provider"] = model_provider_val
        else:
            model_kwargs["model_provider"] = "openai"

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
    - Based on the same tokenizer as the chat model (Qwen/Qwen3-8B).
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

class TaskClassification(BaseModel):
    """Task classification result."""
    task_type: str = Field(description="The value of task type, without any index or description")


async def classify_task_type(user_query: str, final_response: str, task_types: list = None) -> str:
    """
    Generic task type classifier: Determine task type from user query and final response using structured output
    
    Args:
        user_query: User input query text
        final_response: Final synthesized response
        task_types: List of available task types with descriptions (defaults to TASK_TYPES global)
        
    Returns:
        str: Classified task type
    """
    if task_types is None:
        task_types = TASK_TYPES
    
    # Initialize model for task classification
    classification_model = await get_model("task_classification")
    
    # Use structured output to avoid JSON parsing issues
    structured_model = classification_model.with_structured_output(TaskClassification)
    
    # Build task types description
    task_descriptions = json.dumps(task_types, ensure_ascii=False)
    
    # Build classification prompt
    classification_prompt = f"""/no_think\n
You are a task classification expert. Please analyze the user query and final response to determine the most appropriate task type.

User Query: {user_query}

Final Response: {final_response}

Available task types:
{task_descriptions}

Please classify the task type based on the user query and the content of the final response.
"""
    
    try:
        # Call model for classification with structured output
        # For structured output, we need to accumulate all chunks first
        full_result = None
        async for chunk in structured_model.astream([HumanMessage(content=classification_prompt)]):
            full_result = chunk
        
        if full_result is None:
            return task_types[0]["task_type"]
        
        # Extract task_type from structured result
        classified_type = full_result.task_type if hasattr(full_result, 'task_type') else task_types[0]["task_type"]
        
        # Validate that the classified type is in our available task types
        available_types = [task['task_type'] for task in task_types]
        if classified_type not in available_types:
            print(f"Warning: Classified task type '{classified_type}' not in available types, using default")
            return task_types[0]["task_type"]
            
        return classified_type
            
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
                reasoning_effort="low"  # 可选: "low", "medium", "high"
            )
        return self.model
    
    async def _call_model(self, state: MessagesState):
        """Call the model with tools (with streaming)"""
        model = await self._get_model()
        model_with_tools = model.bind_tools(self.tools)
        
        messages = state["messages"]
        # TODO: 去除messages中AIMessage的思考部分
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
        
        # Stream the response and accumulate chunks
        response_chunks = []
        has_content = False
        
        async for chunk in model_with_tools.astream(messages):
            response_chunks.append(chunk)
            # If it's a content chunk, print it in real-time
            if hasattr(chunk, 'content') and chunk.content:
                if not has_content:
                    print("    Agent: ", end='', flush=True)
                    has_content = True
                print(chunk.content, end='', flush=True)
        
        # Print newline after streaming completes if we had any content
        if has_content:
            print()
        
        # Reconstruct the complete response message by merging chunks
        if response_chunks:
            response = response_chunks[0]
            for chunk in response_chunks[1:]:
                response = response + chunk
        else:
            response = AIMessage(content="")
        
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
    
    async def run(self, user_query: str, conversation_messages: List[BaseMessage] = None) -> Dict[str, str]:
        """
        Run the info extraction agent to automatically select and execute tools
        
        Args:
            user_query: User's query
            conversation_messages: Conversation history
            
        Returns:
            Dict mapping tool names to their results
        """
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
        model = await get_model("evaluation_agent")
        model_with_tools = model.bind_tools(self.tools)
        
        messages = state["messages"]

        # 判断是否是第一次调用：检查最后一条消息的类型
        # 第一次：最后一条是 HumanMessage
        # 后续：最后一条是 ToolMessage（从 tools 节点返回）
        last_message = messages[-1] if messages else None
        is_first_call = isinstance(last_message, HumanMessage)
        
        # 第一次用 /no_think，后续用 /think
        sys_msg_content = "/no_think" if is_first_call else "/think"
        messages = [SystemMessage(content=sys_msg_content)] + messages

        # Stream the response and accumulate chunks
        response_chunks = []
        has_content = False
        
        async for chunk in model_with_tools.astream(messages):
            response_chunks.append(chunk)
            # If it's a content chunk, print it in real-time
            if hasattr(chunk, 'content') and chunk.content:
                if not has_content:
                    print("    Evaluation Agent: ", end='', flush=True)
                    has_content = True
                print(chunk.content, end='', flush=True)
        
        # Print newline after streaming completes if we had any content
        if has_content:
            print()
        
        # Reconstruct the complete response message by merging chunks
        if response_chunks:
            response = response_chunks[0]
            for chunk in response_chunks[1:]:
                response = response + chunk
        else:
            response = AIMessage(content="")
        
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
                  task_type: str = TASK_TYPE_GENERAL_INQUIRY) -> Dict[str, Any]:
        """
        Run the evaluation agent to assess final response quality
        
        Args:
            user_query: Original user query
            all_tool_results: Results from all previously executed tools
            conversation_messages: Conversation history
            final_response: The final synthesized response to evaluate
            task_type: Type of task for selecting appropriate evaluation template
            
        Returns:
            Dict with evaluation results including:
            - evaluation_tool_results: Results from evaluation tools (for reference data)
            - evaluation_response: Model's final evaluation output (the actual evaluation)
        """
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
        self.status_callback = None  # 存储状态回调函数
    
    async def _call_model(self, state: MessagesState):
        """LLM decides whether to call tool (with streaming)"""
        # Stream the response and accumulate chunks
        response_chunks = []
        has_content = False
        status_sent = False  # 标记是否已发送状态
        
        async for chunk in self.model_with_tools.astream(state["messages"]):
            response_chunks.append(chunk)
            
            # 在流式过程中检查是否有 tool_calls，一旦检测到立即发送状态
            if not status_sent and hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                status_sent = True
                if self.status_callback:
                    # 先发送"已完成评估诊断需求"
                    self.status_callback("Completed evaluation")
                    # 然后发送"正在进行XXX"
                    tool_name = chunk.tool_calls[0].get('name', 'workflow tool')
                    self.status_callback(f"Executing {tool_name}...")
            
            # If it's a content chunk, print it in real-time
            if hasattr(chunk, 'content') and chunk.content:
                if not has_content:
                    print("    Workflow Agent: ", end='', flush=True)
                    has_content = True
                print(chunk.content, end='', flush=True)
        
        # Print newline after streaming completes if we had any content
        if has_content:
            print()
        
        # Reconstruct the complete response message by merging chunks
        if response_chunks:
            response = response_chunks[0]
            for chunk in response_chunks[1:]:
                response = response + chunk
        else:
            response = AIMessage(content="")
        
        # 如果流式过程中没有检测到 tool_calls，在合并后再次检查（作为备用）
        if not status_sent and self.status_callback and hasattr(response, 'tool_calls') and response.tool_calls:
            # 先发送"已完成评估诊断需求"
            self.status_callback("Completed evaluation")
            # 然后发送"正在进行XXX"
            tool_name = response.tool_calls[0].get('name', 'workflow tool')
            self.status_callback(f"Executing {tool_name}...")
        
        return {"messages": [response]}
    
    def _route(self, state: MessagesState) -> Literal["execute", END]:
        """Determine if should execute tool or end"""
        last = state["messages"][-1]
        if hasattr(last, 'tool_calls') and last.tool_calls:
            print(f"    🔧 Routing to execute: {len(last.tool_calls)} tool call(s) detected")
            # 如果 _call_model 中没有发送状态（可能 tool_calls 在合并后才出现），这里作为备用
            if self.status_callback and last.tool_calls:
                # 先发送"已完成评估诊断需求"
                self.status_callback("Completed evaluation")
                # 然后发送"正在进行XXX"
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
                
                # 实现下面的功能：如果上述必须的参数结果是空，直接返回提示信息,要能从中获取工具名称，结束该agent，
                # 然后在控制中心获取反馈，然后，在控制中心调用指定工具的InfoExtractionAgent获取对应的信息，
                # 然后，再重新调用该agent执行
                # if not phenotype_result or not case_result:
                if not case_result:
                    # 确定缺失的工具
                    missing_tools = []
                    # if not phenotype_result:
                    #     missing_tools.append('phenotype_extractor_tool')
                    if not case_result:
                        missing_tools.append('disease_case_extractor_tool')
                    
                    # 返回特殊标记的错误消息，包含缺失的工具信息
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
                
                # 找到phenotype_result和case_result中共同的表型id键
                # phenotype_result和case_result都是以表型id字符串为键的字典
                # symptom_sets_in_extracted_phenotypes = set(phenotype_result.keys()) if isinstance(phenotype_result, dict) else set()
                symptom_sets_in_extracted_cases = set(case_result.keys()) if isinstance(case_result, dict) else set()
                # common_ids = symptom_sets_in_extracted_phenotypes & symptom_sets_in_extracted_cases
                common_ids = symptom_sets_in_extracted_cases
                
                if not common_ids:
                    # 如果没有共同的症状集，返回错误信息
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
                
                # 对每个共同的id，提取对应的值并执行workflow
                tool_func = next((t for t in self.workflow_tools if t.name == tool_name), None)
                
                if tool_func:
                    # 创建一个字典来存储每个id_key对应的result
                    results_by_id = {}
                    
                    for id_key in common_ids:
                        # 为每个id创建独立的args副本
                        id_args = args.copy()
                        # id_args['extracted_phenotypes'] = phenotype_result[id_key]
                        id_args['extracted_phenotypes'] = case_result[id_key]['extracted_phenotypes']
                        id_args['extracted_disease_cases'] = case_result[id_key]['disease_cases']
                        # Inject user query if not present
                        if 'query' not in id_args:
                            id_args['query'] = self.user_query
                        
                        try:
                            result = tool_func.invoke(id_args)
                            # 将result存储到字典中，键为id_key
                            results_by_id[id_key] = result
                            print(f"  ✓ Executed: {tool_name} for phenotype ID: {id_key}")
                        except Exception as e:
                            # 错误信息也存储到字典中
                            results_by_id[id_key] = {"error": str(e)}
                            print(f"  ❌ Error for phenotype ID {id_key}: {e}")
                    
                    # 将按id_key组织的结果添加到outputs
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
                # 对于非disease_diagnosis_tool的工具，保持原有逻辑
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
    
    async def run(self, user_query: str, tool_results: Dict[str, Any] = None, conversation_messages: List[BaseMessage] = None, status_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """
        LLM decides whether to call workflow, execute if needed
        
        Args:
            user_query: User's query
            tool_results: Results from previous tool calls
            conversation_messages: Conversation history
            status_callback: Optional callback function(status_message: str) to notify status changes
        """
        print("\n" + "="*80)
        print("🔄 WORKFLOW AGENT")
        print("="*80)
        
        self.tool_results = tool_results or {}
        self.user_query = user_query
        self.status_callback = status_callback  # 存储回调函数
        print(f"📊 Available tool results: {list(self.tool_results.keys())}")
        
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

Call workflow tool ONLY if query needs workflow execution. Otherwise respond "No workflow tool needed".
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
        tool_call_detected = False  # 标记是否检测到工具调用
        
        async for chunk in app.astream({"messages": messages}):
            for node, output in chunk.items():
                # 检测工具调用（状态回调已在 _call_model 和 _route 中处理）
                if node == "agent" and output and "messages" in output:
                    for msg in output["messages"]:
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            tool_call_detected = True
                            break
                
                # 检测工具执行完成
                if output and "messages" in output:
                    for msg in output["messages"]:
                        if hasattr(msg, 'type') and msg.type == 'tool':
                            workflow_name = msg.name
                            workflow_result = msg.content
                            
                            # 工具执行完成，发送完成状态
                            if self.status_callback and workflow_name:
                                self.status_callback(f"Completed {workflow_name}")
                            
                            # 检测是否有缺失的工具
                            try:
                                result_data = json.loads(workflow_result)
                                if isinstance(result_data, dict) and result_data.get('status') == 'MISSING_REQUIRED_TOOLS':
                                    missing_tools = result_data.get('missing_tools', [])
                            except (json.JSONDecodeError, TypeError):
                                pass  # 不是JSON格式，正常处理
        
        elapsed = time.time() - start
        
        # 如果没有检测到工具调用，通过回调通知外部
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
            
            # Stream the response and accumulate chunks
            response_chunks = []
            has_content = False
            
            async for chunk in model_with_tools.astream(messages):
                response_chunks.append(chunk)
                # If it's a content chunk, print it in real-time
                if hasattr(chunk, 'content') and chunk.content:
                    if not has_content:
                        print("    Template Agent: ", end='', flush=True)
                        has_content = True
                    print(chunk.content, end='', flush=True)
            
            # Print newline after streaming completes if we had any content
            if has_content:
                print()
            
            # Reconstruct the complete response message by merging chunks
            if response_chunks:
                response = response_chunks[0]
                for chunk in response_chunks[1:]:
                    response = response + chunk
            else:
                response = AIMessage(content="")
            
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

async def synthesize_results(user_query: str, tool_results: Dict[str, str], conversation_messages: List[BaseMessage] = None) -> str:
    """
    Phase 3: Synthesize tool results into final response
    """
    print("\n" + "="*80)
    print("🎨 PHASE 3: SYNTHESIS")
    print("="*80)
    
    # print(f"Tool Results: {tool_results}")

    model = await get_model("synthesizer")
    
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
    
    # Stream the synthesis
    response_chunks = []
    print("\n📋 Final Response:")
    print("-" * 80)
    
    async for chunk in model.astream(messages):
        if hasattr(chunk, 'content') and chunk.content:
            print(chunk.content, end='', flush=True)
            response_chunks.append(chunk.content)
    
    print("\n" + "-" * 80)
    
    final_response = ''.join(response_chunks)
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

**IMPORTANT: This query requires disease diagnosis. You MUST call the disease_diagnosis_tool to perform the diagnosis analysis.**"""
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
    print("🤖 Autonomous Multi-Agent System with LangGraph")
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
    print("  ✓ Automatic tool routing based on model's understanding")
    print("  ✓ Tool-based response evaluation and verification")
    print("  ✓ Post-synthesis quality analysis")
    print("  ✓ 🧠 Conversation memory - remembers previous interactions")
    print()
    print("Available Tools:")
    print("  • phenotype_extractor_tool: Extract phenotype, symptom, HPO IDs from input")
    print("  • disease_case_extractor_tool: Extract phenotypes and generate disease cases")
    print("  • disease_information_retrieval_tool: Extract and normalize disease names with details")
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

