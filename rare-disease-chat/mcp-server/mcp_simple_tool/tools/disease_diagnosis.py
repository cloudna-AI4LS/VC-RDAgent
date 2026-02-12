#!/usr/bin/env python3
"""
MCP server for disease diagnosis functionality using step-by-step reasoning with LangGraph memory
"""
import os
import sys
import json
import re
from typing import List, Dict, Optional, TypedDict, Annotated, Any
from mcp import types
# from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.messages.utils import convert_to_openai_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from openai import OpenAI

# Add parent directory to path for importing other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import load_config function
try:
    from scripts.rare_disease_diagnose.query_kg import load_config
except ImportError as e:
    print(f"Warning: Could not import load_config: {e}")
    def load_config(config_file: str = "config.json") -> dict:
        """Fallback load_config function"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            # Replace {base_path} parameter in all string values
            if 'base_path' in config:
                base_path = config['base_path']
                for key, value in config.items():
                    if isinstance(value, str) and '{base_path}' in value:
                        config[key] = value.format(base_path=base_path)
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, str) and '{base_path}' in subvalue:
                                config[key][subkey] = subvalue.format(base_path=base_path)
            return config
        except Exception:
            return {}

# Import phenotype_extractor and disease_case_extractor
try:
    from tools.phenotype_extractor import phenotype_extractor
except ImportError as e:
    print(f"Warning: Could not import phenotype_extractor: {e}")
    phenotype_extractor = None

try:
    from tools.disease_case_extractor import extract_disease_cases
except ImportError as e:
    print(f"Warning: Could not import disease_case_extractor: {e}")
    extract_disease_cases = None

try:
    from tools.disease_information_retrieval import disease_information_retrieval
except ImportError as e:
    print(f"Warning: Could not import disease_information_retrieval: {e}")
    disease_information_retrieval = None

# Define state type
class DiagnosisState(TypedDict):
    messages: Annotated[List[BaseMessage], "conversation history"]
    step_outputs: Annotated[List[str], "output results for each step"]
    current_step: Annotated[int, "current step"]
    phenotype_data: Annotated[Dict, "phenotype data"]
    step_prompts: Annotated[List[str], "step prompts"]
    final_result: Annotated[Dict, "final result"]
    case_analysis_result: Annotated[Dict, "case analysis result"]
    is_paused: Annotated[bool, "whether the process is paused"]
    pause_step: Annotated[int, "step where the process was paused"]


# Load configuration (same pattern as phenotype_to_disease_controller_dashscope_api)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_file = os.path.join(BASE_DIR, 'scripts/rare_disease_diagnose/prompt_config_forKG.json')
config = load_config(config_file)
model_config = config.get('model_config', {})

USE_STREAMING = model_config.get("streaming", True)


def _get_delta_field(delta, name: str):
    """Extract field from delta (e.g. reasoning_content, content)."""
    if delta is None:
        return None
    val = getattr(delta, name, None)
    if val is not None:
        return val
    if hasattr(delta, "model_extra") and delta.model_extra and name in delta.model_extra:
        return delta.model_extra[name]
    return None


def _get_choice_field(choice, name: str):
    """Extract field from stream choice (delta or message)."""
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
    之后全部作为 content 记录。与 phenotype_to_disease_controller_dashscope_api 完全一致。
    """
    __slots__ = ("_state", "_buf")
    STATE_BEFORE = "before_think"
    STATE_IN_THINK = "in_think"
    STATE_AFTER = "after_think"

    def __init__(self):
        self._state = self.STATE_BEFORE
        self._buf = ""

    def feed(self, chunk: str) -> List:
        """喂入一段 content 文本，返回 [(type, data), ...]，type 为 "reasoning" 或 "content"。"""
        if not chunk:
            return []
        self._buf += chunk
        out = []
        while self._buf:
            if self._state == self.STATE_BEFORE:
                i = self._buf.find(THINK_OPEN)
                if i == -1:
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
            else:
                out.append(("content", self._buf))
                self._buf = ""
                break
        return out

    def flush(self) -> List:
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


def _extract_think_from_content(content: str) -> tuple:
    """当 reasoning_content 为空且 content 内含 <think>...</think> 时，提取 think 作为 reasoning，剩余作为 content。
    返回 (reasoning_part, content_without_think)。若无可提取的 think，返回 ("", content)。与参考脚本一致。
    """
    if not content or not isinstance(content, str):
        return ("", content or "")
    m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if not m:
        return ("", content)
    think_part = m.group(1).strip()
    content_without = re.sub(r"<think>.*?</think>", "", content, count=1, flags=re.DOTALL).strip()
    return (think_part, content_without)


def _call_chat_sync(
    messages: List[BaseMessage],
    stream: bool,
    on_chunk=None,
) -> str:
    """
    Call model via OpenAI-compatible client (same as phenotype_to_disease_controller_dashscope_api):
    client.chat.completions.create(..., extra_body={"enable_thinking": True}).
    Returns the reply text (content only; <think> stripped when reasoning_content was not returned).
    If stream=True and on_chunk is set, calls on_chunk(text) for each reasoning/content chunk.
    """
    model_name = model_config.get("model")
    if not model_name:
        raise ValueError("model_config.model is required.")
    base_url = (model_config.get("base_url") or "").rstrip("/")
    if not base_url:
        raise ValueError("model_config.base_url is required.")
    api_key = model_config.get("api_key", "EMPTY")
    oai_messages = convert_to_openai_messages(messages)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=300.0)

    if stream:
        stream_obj = client.chat.completions.create(
            model=model_name,
            messages=oai_messages,
            stream=True,
            extra_body={"enable_thinking": True},
        )
        reasoning_parts = []
        content_parts = []
        has_reasoning_content = False
        think_parser = _ThinkStreamParser()
        for chunk in stream_obj:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            rc = _get_delta_field(delta, "reasoning_content") or _get_choice_field(choice, "reasoning_content")
            if rc:
                has_reasoning_content = True
                reasoning_parts.append(rc)
                if on_chunk:
                    on_chunk(rc)
            c = _get_delta_field(delta, "content") if delta else None
            if not c:
                c = _get_choice_field(choice, "content")
            if c:
                if has_reasoning_content:
                    content_parts.append(c)
                    if on_chunk:
                        on_chunk(c)
                else:
                    for typ, data in think_parser.feed(c):
                        if typ == "reasoning":
                            reasoning_parts.append(data)
                            if on_chunk:
                                on_chunk(data)
                        else:
                            content_parts.append(data)
                            if on_chunk:
                                on_chunk(data)
        if not has_reasoning_content:
            for typ, data in think_parser.flush():
                if typ == "reasoning":
                    reasoning_parts.append(data)
                    if on_chunk:
                        on_chunk(data)
                else:
                    content_parts.append(data)
                    if on_chunk:
                        on_chunk(data)
        content_full = "".join(content_parts)
        if not has_reasoning_content and content_full:
            _, content_full = _extract_think_from_content(content_full)
        return content_full

    response = client.chat.completions.create(
        model=model_name,
        messages=oai_messages,
        stream=False,
        extra_body={"enable_thinking": True},
    )
    msg = response.choices[0].message if response.choices else None
    if not msg:
        return ""
    content = getattr(msg, "content", None) or ""
    rc = _get_choice_field(msg, "reasoning_content")
    if not rc and content:
        _, content = _extract_think_from_content(content)
    return content

def step_node(state: DiagnosisState) -> DiagnosisState:
    """Execute a single diagnosis step node"""
    current_step = state["current_step"]
    step_prompts = state["step_prompts"]
    step_outputs = state["step_outputs"]
    
    if current_step >= len(step_prompts):
        return state
    
    print(f"\n{'='*60}")
    print(f"🔄 Executing Step {current_step+1}/{len(step_prompts)}...")
    print(f"{'='*60}")
    
    try:
        # Build current step prompt
        current_prompt = step_prompts[current_step]
        
        # Execute current step with streaming output
        current_step_output = ""
        print(f"\n📝 Step {current_step+1} Response:")
        print("-" * 40)

        # print(f"DEBUG: current_prompt: {current_prompt}")

        # Create messages - directly use LangGraph memory mechanism
        # LangGraph automatically maintains messages history, no need for manual concatenation
        messages = state["messages"] + [HumanMessage(content=current_prompt)]
        
        # Call model via OpenAI client (same as phenotype_to_disease_controller_dashscope_api)
        def _print_chunk(t):
            print(t, end="", flush=True)
        current_step_output = _call_chat_sync(messages, USE_STREAMING, on_chunk=_print_chunk if USE_STREAMING else None)
        if not USE_STREAMING:
            print(current_step_output, end="", flush=True)
        
        print(f"\n{'-' * 40}")
        print(f"✅ Step {current_step+1} completed. Output length: {len(current_step_output)}")
        
        current_step_output_cleaned = re.sub(r'<think>.*?</think>', '', current_step_output, flags=re.DOTALL)
        current_step_output_cleaned = re.sub(r'.*?</think>', '', current_step_output_cleaned, flags=re.DOTALL)
        # Extract Step line from current_prompt
        step_match = re.search(r'^.*?(\*\*)?Step (\d+): [^\n]+.*$', current_prompt, re.MULTILINE)
        if step_match:
            step_label = step_match.group(0)
            step_index = step_match.group(2)
        else:
            step_label = f"Step {current_step + 1}"
            step_index = -1

        current_step_output_cleaned = f"{step_label}\n{current_step_output_cleaned}"
        current_step_output_cleaned = current_step_output_cleaned.strip()+"\n"
        
        # Update state
        new_step_outputs = step_outputs + [current_step_output_cleaned]
        # Remove <--> wrapped parts from HumanMessage in messages
        cleaned_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                # Remove <-XXX-> wrapped content from HumanMessage
                cleaned_content = re.sub(r'<-.*?->', '', msg.content, flags=re.DOTALL)
                cleaned_messages.append(HumanMessage(content=cleaned_content))
            else:
                cleaned_messages.append(msg)
        
        new_messages = cleaned_messages + [AIMessage(content=current_step_output_cleaned)]
        # Update steps_prompts
        if "Insert Step" in current_prompt:
            placeholder = f'<INSERT_STEP_{step_index}_OUTPUT_PLACEHOLDER>'
            for i, prompt in enumerate(state["step_prompts"]):
                if placeholder in prompt:
                    formatted_current_step_output_cleaned = format_no_case_prediction(current_step_output_cleaned)
                    state["step_prompts"][i] = prompt.replace(placeholder, formatted_current_step_output_cleaned)

        return {
            "messages": new_messages,
            "step_outputs": new_step_outputs,
            "current_step": current_step + 1,
            "phenotype_data": state["phenotype_data"],
            "step_prompts": state["step_prompts"],
            "final_result": state["final_result"],
            "case_analysis_result": state.get("case_analysis_result", {}),
            "is_paused": state.get("is_paused", False),
            "pause_step": state.get("pause_step", -1),
        }
        
    except Exception as e:
        print(f"\n❌ Error in Step {current_step+1}: {str(e)}")
        error_output = f"Error in Step {current_step+1}: {str(e)}"
        new_step_outputs = step_outputs + [error_output]
        
        # Also need to clean <--> wrapped content in messages
        cleaned_messages = []
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                # Remove <-XXX-> wrapped content from HumanMessage
                cleaned_content = re.sub(r'<-.*?->', '', msg.content, flags=re.DOTALL)
                cleaned_messages.append(HumanMessage(content=cleaned_content))
            else:
                cleaned_messages.append(msg)
        
        new_messages = cleaned_messages + [HumanMessage(content=step_prompts[current_step]), AIMessage(content=error_output)]
        
        return {
            "messages": new_messages,
            "step_outputs": new_step_outputs,
            "current_step": current_step + 1,
            "phenotype_data": state["phenotype_data"],
            "step_prompts": state["step_prompts"],
            "final_result": state["final_result"],
            "case_analysis_result": state.get("case_analysis_result", {}),
            "is_paused": state.get("is_paused", False),
            "pause_step": state.get("pause_step", -1),
        }

def should_continue(state: DiagnosisState) -> str:
    """Determine whether to continue to the next step"""
    current_step = state["current_step"]
    step_prompts = state["step_prompts"]
    
    # Check if we need to pause after no_case_prediction (step index 5, since step_node increments current_step)
    # no_case_prediction is the 5th step (0-indexed), so after execution current_step becomes 5
    if current_step == 5:  # After no_case_prediction execution
        return "pause_for_case_analysis"
    elif current_step < len(step_prompts):
        return "continue"
    else:
        return "end"

def pause_for_case_analysis(state: DiagnosisState) -> DiagnosisState:
    """Pause the diagnosis process and perform case analysis"""
    print(f"\n{'='*60}")
    print(f"⏸️  Pausing diagnosis process for case analysis...")
    print(f"{'='*60}")
    
    # Mark as paused
    return {
        "messages": state["messages"],
        "step_outputs": state["step_outputs"],
        "current_step": state["current_step"],
        "phenotype_data": state["phenotype_data"],
        "step_prompts": state["step_prompts"],
        "final_result": state["final_result"],
        "case_analysis_result": state.get("case_analysis_result", {}),
        "is_paused": True,
        "pause_step": state["current_step"],
    }

# Define a simple case analysis state
class CaseAnalysisState(TypedDict):
    messages: Annotated[List[BaseMessage], "case analysis conversation history"]
    analysis_output: Annotated[str, "case analysis output"]

def case_analysis_step(case_state: CaseAnalysisState) -> CaseAnalysisState:
    """Execute case analysis step"""
    messages = case_state["messages"]

    # print(f"DEBUG: case_analysis_step: messages: {messages}")
    
    # Call model via OpenAI client (same as phenotype_to_disease_controller_dashscope_api)
    print(f"\n📝 Case Analysis Response (New Dialog):")
    print("-" * 40)

    def _print_chunk(t):
        print(t, end="", flush=True)
    analysis_output = _call_chat_sync(messages, USE_STREAMING, on_chunk=_print_chunk if USE_STREAMING else None)
    if not USE_STREAMING:
        print(analysis_output, end="", flush=True)

    print(f"\n{'-' * 40}")
    print(f"✅ Case analysis completed. Output length: {len(analysis_output)}")
    
    return {
        "messages": messages + [AIMessage(content=analysis_output)],
        "analysis_output": analysis_output
    }

def format_no_case_prediction(no_case_prediction: str) -> str:
    """
    Format no_case_prediction output by extracting disease names and enriching with detailed information.
    
    Args:
        no_case_prediction: Raw prediction output string containing disease information
        
    Returns:
        Formatted no_case_prediction string in JSON format (without outer braces)
    """
    try:
        # Clean and locate JSON block
        cleaned = re.sub(r'<think>.*?</think>', '', no_case_prediction, flags=re.DOTALL).strip()
        cleaned = re.sub(r'.*?</think>', '', cleaned, flags=re.DOTALL)
        json_start = cleaned.find('{')
        json_end = cleaned.rfind('}')
        diseases_list: List[str] = []
        if json_start != -1 and json_end != -1 and json_end > json_start:
            payload = cleaned[json_start:json_end+1]
            data = json.loads(payload)
            # Parse strictly according to agreed structure: {"FINAL ANSWER": {"[INDEX. Name]": {...}}}
            if isinstance(data, dict) and isinstance(data.get("FINAL ANSWER"), dict):
                final_answer = data["FINAL ANSWER"]
                parsed: List[tuple] = []
                for k in final_answer.keys():
                    if not isinstance(k, str):
                        continue
                    m = re.match(r"\s*\[?\s*(\d+)\s*\.\s*(.+?)\s*\]?\s*$", k)
                    if not m:
                        continue
                    idx = int(m.group(1)) if m.group(1).isdigit() else -1
                    name = m.group(2).strip()
                    parsed.append((idx, name))
                parsed.sort(key=lambda x: x[0])
                diseases_list = [name for _, name in parsed]
        else:
            # Alternative way to extract diseases: directly extract numbered list format disease names from answer text
            # Support formats: "1. DISEASE_NAME", "**1. DISEASE_NAME**", "1. DISEASE_NAME - description", etc.
            lines = cleaned.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Match lines starting with numbers: 1., **1., "1., etc.
                if re.match(r'^\d+\.', line) or re.match(r'^\*?\*?\d+\.', line) or re.match(r'^\"\d+\.', line):
                    # Remove numbers, markdown format and quotes
                    disease_part = re.sub(r'^\"?\*?\*?\d+\.\s*', '', line)
                    disease_part = disease_part.replace('\"', '')
                    
                    # Remove description part (after - or :)
                    if ' - ' in disease_part:
                        disease_part = disease_part.split(' - ')[0].strip()
                    elif ' – ' in disease_part:
                        disease_part = disease_part.split(' – ')[0].strip()
                    elif ':' in disease_part:
                        disease_part = disease_part.split(':')[0].strip()
                    
                    # Remove parentheses and their content
                    disease_part = re.sub(r'\s*\([^)]*\)', '', disease_part)
                    
                    # Clean format symbols
                    disease_part = disease_part.replace('**', '').replace('*', '').replace("'s", '').replace('\\"', '').strip()
                    
                    # Only keep valid disease names (length > 2)
                    if disease_part and len(disease_part) > 2:
                        diseases_list.append(disease_part)
        
        # Use disease_information_retrieval to enrich with detailed information
        enriched_cases = {}
        if diseases_list:
            detailed_items: Dict[str, Dict] = {}
            if disease_information_retrieval is not None:
                try:
                    dir_result = disease_information_retrieval(
                        query="candidates provided",
                        use_model=False,
                        candidates=diseases_list,
                    )
                    dir_data = json.loads(dir_result)
                    detailed_items = dir_data.get("extracted_diseases", {}) if isinstance(dir_data, dict) else {}
                except Exception as _e:
                    detailed_items = {}
            # To build structure consistent with ranked_diseases_text
            # Build fast index of name -> detailed information (case-insensitive)
            # extracted_diseases is a dictionary, keys are input disease names, values are detailed information
            by_name = {}
            for input_name, item in detailed_items.items():
                if isinstance(input_name, str) and isinstance(item, dict):
                    by_name.setdefault(input_name.lower(), item)
            # Fill Case i according to no_case_prediction sort order
            for i, name in enumerate(diseases_list, 51):
                info = by_name.get(name.lower(), {})
                disease_id = info.get("disease_id", "UNKNOWN")
                if disease_id == "UNKNOWN":
                    continue
                disease_category = info.get("disease_category", "")
                disease_description = info.get("disease_description", "")
                if disease_description == "":
                    disease_description = "[Information is missing; please infer based on your memory.]"
                enriched_cases[f"Case {i}"] = {
                    "Disease name": info.get("standard_name", name),
                    "Disease id": disease_id,
                    "Disease category": disease_category,
                    "Disease description": disease_description,
                }
        
        if enriched_cases:
            # Convert no_case_prediction result to JSON text with same structure as disease_cases
            formatted_output = json.dumps(enriched_cases, ensure_ascii=False, indent=2)
            # Trim whitespace first, then remove outer braces if present
            _trimmed = formatted_output.strip()
            if _trimmed.startswith("{") and _trimmed.endswith("}"):
                return _trimmed[1:-1].strip()
            else:
                return _trimmed
        else:
            # If no enriched information, return original input
            return no_case_prediction
            
    except Exception as _parse_e:
        print(f"DEBUG: Error parsing no_case_prediction: {_parse_e}")
        # Keep original input, don't interrupt flow
        return no_case_prediction

def case_analysis_node(state: DiagnosisState) -> DiagnosisState:
    """Perform case analysis using a new thread_id"""
    print(f"\n{'='*60}")
    print(f"🔍 Starting Case Analysis...")
    print(f"{'='*60}")
    
    try:
        # Get the ranked diseases text from phenotype_data
        ranked_diseases_text = state["phenotype_data"].get("ranked_diseases_text", "")
        
        # Get the no_case_prediction output (now it's the 5th step output, 0-indexed)
        step_outputs = state["step_outputs"]
        no_case_prediction = ""
        if len(step_outputs) >= 5:  # no_case_prediction is the 5th step
            no_case_prediction = step_outputs[4]  # 0-indexed
            # Format no_case_prediction
            no_case_prediction = format_no_case_prediction(no_case_prediction)
        
        if ranked_diseases_text and no_case_prediction:
            
            # Extract the top 10 diagnoses from the output
            # This is a simplified extraction - you might need more sophisticated parsing
            case_analysis_prompt = f"""
**Insert Step 3: Case Analysis**
You are provided with a set of Cases, each exhibiting clinical manifestations characteristic of a specific disease. Detailed information for each patient is as follows:
{ranked_diseases_text}\n{no_case_prediction}

Based on both the disease cases data and the previous diagnosis results, summarize the key symptoms or combinations of phenotypes that are essential and specific for the diagnosis of each disease, and provide your output in the following JSON format (provide only valid JSON, not markdown format):
Please use standard phenotype names (Not HPO IDs) to describe the key symptoms of each disease.

**Insert Step 3 output format in json (must use json format, not markdown):**
{{
    "[Case INDEX. Disease Name]": "[List the key symptoms or combinations of phenotypes that are essential and specific for the diagnosis of this disease. Please use standard HPO phenotype names.]"
}}
""" 
            # Create a truly independent case analysis conversation using LangGraph
            case_checkpointer = InMemorySaver()
            
            # Build case analysis graph
            case_builder = StateGraph(CaseAnalysisState)
            case_builder.add_node("analyze", case_analysis_step)
            case_builder.add_edge(START, "analyze")
            case_builder.add_edge("analyze", END)
            case_graph = case_builder.compile(checkpointer=case_checkpointer)
            
            # Execute case analysis with completely new thread_id
            case_config = {"configurable": {"thread_id": f"case_analysis_{hash(ranked_diseases_text)}"}}
            case_initial_state = {
                "messages": [HumanMessage(content=case_analysis_prompt)],
                "analysis_output": ""
            }
            
            # Stream the case analysis execution
            case_analysis_output = ""
            for chunk in case_graph.stream(case_initial_state, case_config, stream_mode="values"):
                if chunk.get("analysis_output"):
                    case_analysis_output = chunk["analysis_output"]
            
            case_analysis_result = {
                "analysis_output": case_analysis_output,
                "status": "completed"
            }
            
        else:
            case_analysis_result = {
                "analysis_output": "No sufficient data available for case analysis (missing ranked_diseases_text or no_case_prediction output)",
                "status": "error"
            }
        
        # Resume the original diagnosis process
        print(f"\n{'='*60}")
        print(f"🔄 Resuming diagnosis process...")
        print(f"{'='*60}")

        case_analysis_output_cleaned = re.sub(r'<think>.*?<\/think>', '', case_analysis_output, flags=re.DOTALL)
        # Add case analysis result to step outputs for the next step to use
        # case_analysis_output_cleaned = f"**Case Analysis Results:**\n{case_analysis_output_cleaned}\n"
        
        # Update the prompt containing the placeholder to replace it with actual case analysis result
        updated_step_prompts = state["step_prompts"].copy()
        for i, prompt in enumerate(updated_step_prompts):
            if "<CASE_ANALYSIS_RESULT_PLACEHOLDER>" in prompt:
                updated_step_prompts[i] = prompt.replace(
                    "<CASE_ANALYSIS_RESULT_PLACEHOLDER>", 
                    case_analysis_output_cleaned
                )
        
        # Update state["messages"] and state["step_outputs"], remove messages and step_outputs containing insert step from corresponding lists
        # Filter out messages containing "Insert Step"
        # If HumanMessage contains "Insert Step", remove that HumanMessage and its following AIMessage
        filtered_messages = []
        i = 0
        while i < len(state["messages"]):
            msg = state["messages"][i]
            if isinstance(msg, HumanMessage) and "Insert Step" in msg.content:
                # Skip HumanMessage containing "Insert Step"
                i += 1
                # If next message is AIMessage, also skip it
                if i < len(state["messages"]) and isinstance(state["messages"][i], AIMessage):
                    i += 1
            else:
                # Keep other messages
                filtered_messages.append(msg)
                i += 1
        
        # Filter out step_outputs containing "Insert Step"
        filtered_step_outputs = []
        for output in state["step_outputs"]:
            if "Insert Step" not in output:
                filtered_step_outputs.append(output)
        
        return {
            "messages": filtered_messages,
            "step_outputs": filtered_step_outputs,
            "current_step": state["current_step"],
            "phenotype_data": state["phenotype_data"],
            "step_prompts": updated_step_prompts,
            "final_result": state["final_result"],
            "case_analysis_result": case_analysis_result,
            "is_paused": False,
            "pause_step": state.get("pause_step", -1),
        }
        
    except Exception as e:
        print(f"\n❌ Error in case analysis: {str(e)}")
        case_analysis_result = {
            "analysis_output": f"Error in case analysis: {str(e)}",
            "status": "error"
        }
        
        # Add error case analysis result to step outputs
        # error_case_analysis_step_output = f"**Case Analysis Results (Error):**\n{case_analysis_result['analysis_output']}\n"
        # new_step_outputs = state["step_outputs"] + [error_case_analysis_step_output]
        
        # Update the prompt containing the placeholder to replace it with error message
        updated_step_prompts = state["step_prompts"].copy()
        for i, prompt in enumerate(updated_step_prompts):
            if "<CASE_ANALYSIS_RESULT_PLACEHOLDER>" in prompt:
                updated_step_prompts[i] = prompt.replace(
                    "<CASE_ANALYSIS_RESULT_PLACEHOLDER>", 
                    case_analysis_result.get("analysis_output", "Error in case analysis")
                )
        
        # Update state["messages"] and state["step_outputs"], remove messages and step_outputs containing insert step from corresponding lists
        # Filter out messages containing "Insert Step"
        # If HumanMessage contains "Insert Step", remove that HumanMessage and its following AIMessage
        filtered_messages = []
        i = 0
        while i < len(state["messages"]):
            msg = state["messages"][i]
            if isinstance(msg, HumanMessage) and "Insert Step" in msg.content:
                # Skip HumanMessage containing "Insert Step"
                i += 1
                # If next message is AIMessage, also skip it
                if i < len(state["messages"]) and isinstance(state["messages"][i], AIMessage):
                    i += 1
            else:
                # Keep other messages
                filtered_messages.append(msg)
                i += 1
        
        # Filter out step_outputs containing "Insert Step"
        filtered_step_outputs = []
        for output in state["step_outputs"]:
            if "Insert Step" not in output:
                filtered_step_outputs.append(output)

        return {
            "messages": filtered_messages,
            "step_outputs": filtered_step_outputs,
            "current_step": state["current_step"],
            "phenotype_data": state["phenotype_data"],
            "step_prompts": updated_step_prompts,
            "final_result": state["final_result"],
            "case_analysis_result": case_analysis_result,
            "is_paused": False,
            "pause_step": state.get("pause_step", -1),
        }

def finalize_result(state: DiagnosisState) -> DiagnosisState:
    """Complete diagnosis and generate final result"""
    print(f"\n{'='*60}")
    print(f"🎉 Disease Diagnosis Process Completed!")
    print(f"{'='*60}")
    # print(f"📊 Total Steps Executed: {len(state['step_outputs'])}")
    # print(f"📝 Total Output Length: {sum(len(output) for output in state['step_outputs'])} characters")
    # print(f"🏥 Final Diagnosis Available in Results")
    # print(f"{'='*60}")

    # Update state["messages"] and state["step_outputs"], remove messages and step_outputs containing insert step from corresponding lists
    # Filter out messages containing "Insert Step"
    # If HumanMessage contains "Insert Step", remove that HumanMessage and its following AIMessage
    filtered_messages = []
    i = 0
    while i < len(state["messages"]):
        msg = state["messages"][i]
        if isinstance(msg, HumanMessage) and "Insert Step" in msg.content:
            # Skip HumanMessage containing "Insert Step"
            i += 1
            # If next message is AIMessage, also skip it
            if i < len(state["messages"]) and isinstance(state["messages"][i], AIMessage):
                i += 1
        else:
            # Keep other messages
            filtered_messages.append(msg)
            i += 1
    
    # Filter out step_outputs containing "Insert Step"
    filtered_step_outputs = []
    for output in state["step_outputs"]:
        if "Insert Step" not in output:
            filtered_step_outputs.append(output)
    
    final_result = {
        # "extracted_phenotypes": state["phenotype_data"].get("extracted_phenotypes", {}),
        "diagnosis_process": filtered_step_outputs,
    }
    
    return {
        "messages": filtered_messages,
        "step_outputs": filtered_step_outputs,
        "current_step": state["current_step"],
        "phenotype_data": state["phenotype_data"],
        "step_prompts": state["step_prompts"],
        "final_result": final_result,
        "case_analysis_result": state.get("case_analysis_result", {}),
        "is_paused": False,
        "pause_step": state.get("pause_step", -1),
    }

def disease_diagnosis(original_query: str, extracted_phenotypes: dict=None, disease_cases: dict=None) -> str:
    """
    Workflow Tool for Disease Diagnosis: Diagnose diseases based on extracted phenotypes and disease cases.
    
    This workflow tool performs step-by-step disease diagnosis using two pre-extracted dictionaries:
    1. extracted_phenotypes: from phenotype extractor
    2. disease_cases: from disease case extractor

    Args:
        extracted_phenotypes (dict): Phenotype data dictionary from phenotype extractor
            Format: {"phenotype_name": {"hpo_id": "HP:xxx", "phenotype abnormal category": "...", ...}, ...}
        disease_cases (dict): Disease cases dictionary from case extractor
            Format: {"Case 1": {"Disease name": "...", "Disease id": "...", ...}, ...}
        
    """
    
    # Validate required parameters
    if not extracted_phenotypes:
        return json.dumps({
            "success": False,
            "error": "Missing required parameter 'extracted_phenotypes'"
        }, ensure_ascii=False, indent=2)
    
    if not disease_cases:
        return json.dumps({
            "success": False,
            "error": "Missing required parameter 'disease_cases'"
        }, ensure_ascii=False, indent=2)
    
    if not isinstance(extracted_phenotypes, dict):
        return json.dumps({
            "success": False,
            "error": "'extracted_phenotypes' must be a dictionary"
        }, ensure_ascii=False, indent=2)
    
    if not isinstance(disease_cases, dict):
        return json.dumps({
            "success": False,
            "error": "'disease_cases' must be a dictionary"
        }, ensure_ascii=False, indent=2)
    
    if not original_query:
        return json.dumps({
            "success": False,
            "error": "Missing required parameter 'original_query'"
        }, ensure_ascii=False, indent=2)
    
    try:
        # Use pre-extracted phenotypes and disease cases
        print("DEBUG: Processing pre-extracted phenotypes and disease cases...")
        
        # Build phenotype details text from extracted_phenotypes
        phenotype_data = extracted_phenotypes
        pheno_details_text = json.dumps(phenotype_data, ensure_ascii=False, indent=2)
        phenotype_categories_json_text = pheno_details_text
        
        # Extract phenotypes list for counting
        phenotypes = []
        missing_category_phenotypes = []
        for phenotype_name, details in extracted_phenotypes.items():
            hpo_id = details.get("hpo_id", "")
            if hpo_id:
                phenotypes.append(hpo_id)
                
                # Check for missing category
                category = details.get("phenotype abnormal category", "")
                if not category or category == "Unknown category":
                    missing_category_phenotypes.append(phenotype_name)
        
        # Build step_prompts following vc_ranker.py logic
        pheno_list_length = len(phenotypes)
        missing_category_phenotypes_text = ", ".join(missing_category_phenotypes) if missing_category_phenotypes else "None"
        
        # print(f"DEBUG: pheno_ids: {phenotypes}")
        # Step 3: Use pre-extracted disease cases
        print("DEBUG: Processing disease cases from input...")
        ranked_diseases_text = ""
        
        if disease_cases and isinstance(disease_cases, dict):
            # Format the disease cases for use in prompts
            ranked_diseases_text = json.dumps(disease_cases, ensure_ascii=False, indent=2)
            # Trim whitespace first, then remove outer braces if present
            _trimmed = ranked_diseases_text.strip()
            if _trimmed.startswith("{") and _trimmed.endswith("}"):
                ranked_diseases_text = _trimmed[1:-1].strip()
            else:
                ranked_diseases_text = _trimmed
            
            print(f"DEBUG: Successfully processed {len(disease_cases)} disease cases")
        else:
            print("DEBUG: No disease cases found in input")
            ranked_diseases_text = "No disease cases available"

        # Build step_prompts array
        step_prompts_7steps = [
            f"""
**Rare Disease Diagnosis Chain-of-Thought (CoT) Prompt**

**Role**:
You are a top-tier rare disease expert and clinical geneticist, skilled in analyzing and inferring potential diagnoses from complex, multi-systemic combinations of phenotypes.

**Task**:
You have received a patient case with {pheno_list_length} detailed phenotypes:
{pheno_details_text}

**Step 1: Phenotype Deconstruction & Categorization**
First, do not treat all input phenotypes as a disordered list. Group them into meaningful clinical categories **(that is by the "phenotype abnormal category" of patient phenotypes)**. Create several core functional/system categories and assign all relevant phenotypes to them:

<-
- Abnormality of the genitourinary system: (if present) Collate all phenotypes related to the genitourinary system. 
- Abnormality of head or neck: (if present) Collate all phenotypes related to the head or neck.
- Abnormality of the eye: (if present) Collate all phenotypes related to the eye.
- Abnormality of the ear: (if present) Collate all phenotypes related to the ear.
- Abnormality of the nervous system: (if present) Collate all phenotypes related to the nervous system.
- Abnormality of the breast: (if present) Collate all phenotypes related to the breast.
- Abnormality of the endocrine system: (if present) Collate all phenotypes related to the endocrine system.
- Abnormality of prenatal development or birth: (if present) Collate all phenotypes related to the prenatal development or birth.
- Growth abnormality: (if present) Collate all phenotypes related to the growth abnormality.
- Abnormality of the integument: (if present) Collate all phenotypes related to the integument.
- Abnormality of the voice: (if present) Collate all phenotypes related to the voice.
- Abnormality of the cardiovascular system: (if present) Collate all phenotypes related to the cardiovascular system.
- Abnormality of blood and blood-forming tissues: (if present) Collate all phenotypes related to the blood and blood-forming tissues.
- Abnormality of metabolism/homeostasis: (if present) Collate all phenotypes related to the metabolism/homeostasis.
- Abnormality of the respiratory system: (if present) Collate all phenotypes related to the respiratory system.
- Neoplasm: (if present) Collate all phenotypes related to the neoplasm.
- Abnormality of the immune system: (if present) Collate all phenotypes related to the immune system.
- Abnormality of the digestive system: (if present) Collate all phenotypes related to the digestive system.
- Constitutional symptom: (if present) Collate all phenotypes related to the constitutional symptom.
- Abnormal cellular phenotype: (if present) Collate all phenotypes related to the abnormal cellular phenotype.
- Abnormality of the musculoskeletal system: (if present) Collate all phenotypes related to the musculoskeletal system.
- Abnormality of limbs: (if present) Collate all phenotypes related to the limbs.
- Abnormality of the thoracic cavity: (if present) Collate all phenotypes related to the thoracic cavity.

Phenotype categorization based on known phenotype classifications (JSON):
{phenotype_categories_json_text}

Phenotypes missing category (need to classification): **{missing_category_phenotypes_text}**. 
If this list is empty, skip this step; otherwise categorize these phenotypes lacking category information into the appropriate functional/system categories in the Step 1 output.
->

**Step 1 output format in json (must use json format, not markdown):**
<-
{{
    "Functional/System Categories":{{
        "[Categories]": [phenotypes1, phenotypes2, ...],
    }}
}}
->
""",

            f"""
**Step 2: Identifying Key Diagnostic Clues from Observable Symptoms**
Focus on **clinically observable symptoms and signs** to identify "anchor phenotypes" or "phenotypic combinations" with the highest diagnostic value. These are often rare, highly specific features that can significantly narrow the differential diagnosis.
The input phenotypes are arranged in descending order of disease specificity, so those toward the bottom of the list are less specific. Do not overemphasize a single low-specificity phenotypes during reasoning.

<-
- Key rule:
  - Focus first on the combinations of observable symptoms and signs that clearly point to a known syndrome.
  - If the patient exhibits **metabolic or laboratory abnormal phenotypes**, consider them as **explanatory evidence** to support or clarify the underlying pathophysiology of the key observable symptoms.

- Instructions for generating Step 2 output:
  - For "anchor clues": Identify the most central and clinically informative features in the patient's phenotype list. These features are pivotal for narrowing disease categories and are highly specific or distinctive, guiding clinical reasoning.
    - Example: ["phenotype1", "phenotype2"]
  - For "key phenotypic clusters": Identify groups of phenotypes that form a recognizable clinical pattern or appear together in a highly specific way, which suggest a particular disease category.
    - Example: ["[Description of key phenotypic cluster 1]": "A classic feature of [Disease Category], such as [disease1, disease2, ...]", "[Description of key phenotypic cluster 2]": "Indicates [Disease Category], such as [disease1, disease2, ...]", ..., "[Description of key phenotypic cluster N]": "Suggests [Disease Category], such as [disease1, disease2, ...]"],  
  - Please use specific combinations of phenotypes to represent each key phenotypic cluster.
  - Please provide three different decisions/verdicts with distinct focuses and record them under Judgment 1, Judgment 2, and Judgment 3.
->

**Step 2 output format in json (must use json format, not markdown):**
<-
{{
    Judgment 1:{{
        "anchor clues": [anchor phenotypes list],
        "key phenotypic clusters": [
            "[Description of key phenotypic cluster 1]": "A classic feature of [Disease Category], such as [disease1, disease2, ...]",
            "[Description of key phenotypic cluster 2]": "Indicates [Disease Category], such as [disease1, disease2, ...]",
            ...
            "[Description of key phenotypic cluster N]": "Suggests [Disease Category], such as [disease1, disease2, ...]",
        ]
    }},
    Judgment 2:{{
        "anchor clues": [anchor phenotypes list],
        "key phenotypic clusters": [
            "[Description of key phenotypic cluster 1]": "A classic feature of [Disease Category], such as [disease1, disease2, ...]",
            "[Description of key phenotypic cluster 2]": "Indicates [Disease Category], such as [disease1, disease2, ...]",
            ...
            "[Description of key phenotypic cluster N]": "Suggests [Disease Category], such as [disease1, disease2, ...]",
        ]
    }},
    Judgment 3:{{
        "anchor clues": [anchor phenotypes list],
        "key phenotypic clusters": [
            "[Description of key phenotypic cluster 1]": "A classic feature of [Disease Category], such as [disease1, disease2, ...]",
            "[Description of key phenotypic cluster 2]": "Indicates [Disease Category], such as [disease1, disease2, ...]",
            ...
            "[Description of key phenotypic cluster N]": "Suggests [Disease Category], such as [disease1, disease2, ...]",
        ]
    }}
}}
->
""",

            f"""
**Step 3: Formulating a Core Clinical Hypothesis**
Based on the key phenotypic clusters from the **Step 2 output**, formulate a unifying hypothesis and a concurrent diseases hypothesis. This hypothesis must not focus on a single organ system. Instead, it must define the underlying pathophysiological process that you believe connects the most significant, yet seemingly disparate, clinical clusters.

<-
- Instructions for generating Step 3 output: Your hypothesis must be framed from a systemic perspective. Identify the fundamental disease process that can independently explain the different key phenotypic clusters. The diagnostic challenge is to identify an entity within this class of disease that can account for these core features as well as other significant involvements.

- **Hypothesis Template Example (in json format):**

  "Unifying Hypothesis":
    [
      - The most parsimonious explanation is a single systemic disease process, that is [Description of a Broad Disease Category].
      - This hypothesis is based on the pathophysiological mechanism of [X] causing both [Phenotypic Cluster A] and [Phenotypic Cluster B] simultaneously.
      - Supporting Evidence: Key findings that strongly support this single diagnosis include: [List the most compelling evidence that fits this one hypothesis].
      - Inconsistencies/Red Flags: [List clinical findings that are difficult to explain or contradict this hypothesis].
    ]

  "Concurrent Diseases Hypothesis":
    [
      - An alternative, and often more probable, explanation is that the patient has two or more distinct conditions occurring concurrently. This hypothesis is proposed to resolve the inconsistencies identified in the Unifying Hypothesis.
      - The most likely combination of conditions is:
        - Primary Condition: A [Description of a Broad Disease Category] to explain the most severe or defining features, such as [Phenotypic Cluster A].
        - Co-existing Condition(s): A [Description of a Broad Disease Category] to explain the remaining key finding(s) of [Phenotypic Cluster B].
      - Supporting Evidence: This combination is plausible because [Explain why this combination makes clinical sense].
      - Inconsistencies/Red Flags: [List aspects this combination fails to explain].
    ]
->

**Step 3 output format in json (must use json format, not markdown):**
<-
{{
    "Unifying Hypothesis": "[Description of Unifying Hypothesis according to the above template]",
    "Concurrent Diseases Hypothesis": "[Description of Concurrent Diseases Hypothesis according to the above template]",
}}
->
""",

            f"""
**Insert Step 1: Generating Candidate Rare Diseases**
Taking into account the above **hypothesis** and the overlap of **anchor phenotypes** and **key phenotypic clusters** between the known patients and the patient to be diagnosed, systematically identify and select 20 rare diseases that are most likely to be relevant as candidate diagnoses.
**Note: You must strictly select exactly 20 candidate rare diseases as potential diagnoses—no more, no less.**

**Insert Step 1 output format in json (must use json format, not markdown):**
<-
{{
    "20 Candidate Rare Diseases": {{
        "[Rare Disease Name]": [List the key patient phenotypes this disease can explain],
    }}
}}
->
""",

            f"""
**Insert Step 2: Generating and Ranking the Differential Diagnosis**
Please analyze the 20 candidate rare diseases listed above **in greater detail**, and **sort the 20 Candidate Rare Diseases** to get the 10 most likely diagnoses.

Output Template:

1. [Rare Disease Name] (Primary Diagnosis)
  - Matched: [List key patient phenotypes this disease explains.]
  - Unmatched: [List key patient phenotypes this disease does not explain.]
  - Reasoning: [Explain why this is the best fit.]

2. [Rare Disease Name] (Secondary Diagnosis)
  - Matched: [List phenotypes explained.]
  - Unmatched: [List phenotypes NOT explained.]
  - Reasoning: [Explain why this is a plausible but secondary option.]

(Continue this format for the top 10 diagnoses.)
**Note: The diagnosis list needs to be sorted by matching degree from high to low.**

**Insert Step 2 output format in json (must use json format, not markdown):**
{{
    "FINAL ANSWER":{{
        "[INDEX. Rare Disease Name]": {{
            "Matched": [List key patient phenotypes this disease explains.],
            "Unmatched": [List key patient phenotypes this disease does not explain.],
            "Reasoning": "[Rationale for ranking the disease of diagnosis]",
        }},
    }}
}}
""",

            f"""
**Step 6: Extracting Candidate Rare Diseases from Known Cases**
The following known patients (no ranking implied) have overlapping phenotypes with the patient to be diagnosed, among which:
<-
{ranked_diseases_text}\n<INSERT_STEP_2_OUTPUT_PLACEHOLDER>
The key features of the diseases manifested by the known patients are as follows:
->
<CASE_ANALYSIS_RESULT_PLACEHOLDER>

Taking into account the above **hypothesis** and the overlap of **anchor phenotypes** and **key phenotypic clusters** between the known patients and the patient to be diagnosed, systematically identify and select 20 rare diseases that are most likely to be relevant as candidate diagnoses.
The Candidate Rare Diseases must select from the disease of the **known patients**.
**Note: You must strictly select exactly 20 candidate rare diseases as potential diagnoses—no more, no less.**

**Step 6 output format in json (must use json format, not markdown):**
<-
{{
    "20 Candidate Rare Diseases": {{
        "[Disease Name (Patient INDEX)]": ["Reason for selecting this as a candidate diagnose"],
    }}
}}
->
""", 
            f"""
**Step 7: Generating and Ranking the Differential Diagnosis**
Please analyze the 20 candidate rare diseases listed above **in greater detail**, and **sort the 20 Candidate Rare Diseases** to get the 10 most likely diagnoses.

Output Template:

1. [Rare Disease Name] (Primary Diagnosis)
  - Matched: [List key patient phenotypes this disease explains.]
  - Unmatched: [List key patient phenotypes this disease does not explain.]
  - Reasoning: [Explain why this is the best fit.]

2. [Rare Disease Name] (Secondary Diagnosis)
  - Matched: [List phenotypes explained.]
  - Unmatched: [List phenotypes NOT explained.]
  - Reasoning: [Explain why this is a plausible but secondary option.]

(Continue this format for the top 10 diagnoses.)
**Note: The diagnosis list needs to be sorted by matching degree from high to low.**

**Step 7 output format in json (must use json format, not markdown):**
{{
    "FINAL ANSWER":{{
        "[INDEX. Rare Disease Name]": {{
            "Matched": [List key patient phenotypes this disease explains.],
            "Unmatched": [List key patient phenotypes this disease does not explain.],
            "Reasoning": "[Rationale for ranking the disease of diagnosis]",
        }},
    }}
}}
"""
        ]

        step_prompts_2steps = [
            f"""Insert Step 1: Generating and Ranking the Differential Diagnosis
You are an expert in rare diseases.

A patient presents with the following detailed phenotypes:
{pheno_details_text}

**Task:** Identify exactly 10 potential candidate rare diseases **based solely on the phenotypes**.

**Rules for reasoning:**
1. Use ONLY the phenotypes provided.
2. Consider clusters of phenotypes by organ/system to detect disease patterns.
3. The patient may have multiple underlying conditions; not all phenotypes need to fit one disease.
4. Treat all phenotypes as potentially relevant; do not over-prioritize any single feature.
5. Do not exclude a disease if some phenotypes are missing.
6. Consider inferred patient factors (age, sex, onset, progression) from phenotypes.
7. Prioritize rare and well-documented diseases with high phenotype specificity.

**Output:** List exactly 10 candidate rare diseases in order of relevance, no explanations, and the output must start with "FINAL ANSWER:".

FINAL ANSWER:
1. DISEASE_NAME
2. DISEASE_NAME
3. DISEASE_NAME
4. DISEASE_NAME
5. DISEASE_NAME
6. DISEASE_NAME
7. DISEASE_NAME
8. DISEASE_NAME
9. DISEASE_NAME
10. DISEASE_NAME
""",

            f"""Step 1: Generating and Ranking the Differential Diagnosis
You are an expert in rare diseases.

A patient presents with the following detailed phenotypes:
{pheno_details_text}

The following diseases are considered highly relevant candidates (**unordered list; no ranking implied**), list with the associated phenotypes: 

{ranked_diseases_text}\n<INSERT_STEP_1_OUTPUT_PLACEHOLDER>

Please follow the structured reasoning approach below to identify the most likely diseases for this patient.  

---

**IMPORTANT ADDITIONAL INSTRUCTION:**  
For each candidate disease, you must not only rely on the explicitly listed phenotype associations above,  
but also incorporate your own medical and genetic knowledge base to identify additional potential phenotype-disease associations.  
This ensures that diseases are not unfairly excluded due to incomplete associations in the provided list.  
If your knowledge base indicates possible links between the candidate disease and the patient's phenotypes, you should include that in your reasoning.  

**REASONING APPROACH (Strict):**

1. **List all provided phenotypes.**
2. **Cluster phenotypes by organ/system**.  
3. **Identify key phenotype patterns/signatures** that indicate likely disease classes.  
4. **Cross-match phenotype clusters with the provided candidate diseases.**  
   - The patient may have multiple diseases; not all phenotypes must fit one disease. 
   - Do not over-prioritize any single phenotype; consider all phenotypes as relevant.
   - Do not exclude a disease solely because some provided phenotypes are missing, as not all phenotypes must appear in a single disease.
   - Diseases with higher overlap of phenotype patterns should be ranked higher.  
5. **Assess progression, age, and sex implications** inferred from the phenotypes.  
6. **Select at least 10 candidate diseases** that best fit the phenotype distribution.  

---

**FINAL ANSWER FORMAT (strict):**  
List the 10 most likely candidate diseases in order, with no explanations, and the output must start with "FINAL ANSWER":  

FINAL ANSWER:
1. DISEASE_NAME
2. DISEASE_NAME
3. DISEASE_NAME
4. DISEASE_NAME
5. DISEASE_NAME
6. DISEASE_NAME
7. DISEASE_NAME
8. DISEASE_NAME
9. DISEASE_NAME
10. DISEASE_NAME
"""

        ]
        
        
        # Create LangGraph with memory
        checkpointer = InMemorySaver()
        
        # Build the main diagnosis graph
        builder = StateGraph(DiagnosisState)
        builder.add_node("step", step_node)
        builder.add_node("pause_for_case_analysis", pause_for_case_analysis)
        builder.add_node("case_analysis", case_analysis_node)
        builder.add_node("finalize", finalize_result)
        
        # Add edges
        builder.add_edge(START, "step")
        builder.add_conditional_edges(
            "step",
            should_continue,
            {
                "continue": "step",
                "pause_for_case_analysis": "pause_for_case_analysis",
                "end": "finalize"
            }
        )
        builder.add_edge("pause_for_case_analysis", "case_analysis")
        builder.add_edge("case_analysis", "step")  # Resume to step after case analysis
        builder.add_edge("finalize", END)
        
        # Compile the graph with checkpointer for memory
        graph = builder.compile(checkpointer=checkpointer)
        
        # Prepare initial state
        initial_state = {
            "messages": [],
            "step_outputs": [],
            "current_step": 0,
            "phenotype_data": {
                "original_input": original_query,
                "extracted_phenotypes": extracted_phenotypes,
                "ranked_diseases_text": ranked_diseases_text
            },
            "step_prompts": step_prompts_2steps,
            "final_result": {},
            "case_analysis_result": {},
            "is_paused": False,
            "pause_step": -1
        }
        
        # Execute the graph with memory
        config = {"configurable": {"thread_id": f"diagnosis_{hash(original_query)}"}}
        
        # Stream the execution
        final_state = None
        for chunk in graph.stream(initial_state, config, stream_mode="values"):
            final_state = chunk
        
        # Return the final result
        if final_state and final_state.get("final_result"):
            # print(f"DEBUG: final_state: {final_state.get("final_result")}")
            return json.dumps(final_state["final_result"], ensure_ascii=False, indent=2)
        else:
            return json.dumps({
                "success": False,
                "error": "Failed to complete diagnosis process"
            }, ensure_ascii=False, indent=2)
        
    except json.JSONDecodeError as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to parse phenotype data: {str(e)}"
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Error in disease diagnosis: {str(e)}"
        }, ensure_ascii=False, indent=2)

def get_tool_definition() -> types.Tool:
    """Return tool definition"""
    return types.Tool(
        name="disease-diagnosis",
        description="Workflow Tool for Disease Diagnosis: Diagnose diseases based on extracted phenotypes and disease cases.",
        inputSchema={
            "type": "object",
            "properties": {
                "original_query": {
                    "type": "string",
                    "description": "Original user query"
                },
                "extracted_phenotypes": {
                    "type": "object",
                    "description": "Phenotype data dictionary from phenotype extractor. Format: {\"phenotype_name\": {\"hpo_id\": \"HP:xxx\", \"phenotype abnormal category\": \"...\", ...}, ...}"
                },
                "disease_cases": {
                    "type": "object",
                    "description": "Disease cases dictionary from case extractor. Format: {\"Case 1\": {\"Disease name\": \"...\", \"Disease id\": \"...\", ...}, ...}"
                }
            },
            "required": ["original_query", "extracted_phenotypes", "disease_cases"]
        }
    )

async def call_tool(context, arguments: Dict[str, Any]) -> List[types.ContentBlock]:
    """Execute disease diagnosis tool"""
    try:
        original_query = arguments.get("original_query", "")
        extracted_phenotypes = arguments.get("extracted_phenotypes")
        disease_cases = arguments.get("disease_cases")
        
        result = disease_diagnosis(original_query, extracted_phenotypes, disease_cases)
        
        return [types.TextContent(
            type="text",
            text=result
        )]
        
    except Exception as e:
        error_msg = {
            "success": False,
            "error": f"Error during disease diagnosis: {str(e)}"
        }
        return [types.TextContent(
            type="text",
            text=json.dumps(error_msg, ensure_ascii=False, indent=2)
        )]
