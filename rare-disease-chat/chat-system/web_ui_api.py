#!/usr/bin/env python3
"""
FastAPI Web UI for Phenotype to Disease Controller
提供Web界面API，支持流式输出
"""
import asyncio
import json
import re
import logging
from typing import List, Optional, AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 导入原有的controller模块
from phenotype_to_disease_controller_langchain_stream_api import (
    controller_pipeline,
    count_messages_tokens,
    MAX_TOKENS,
    HumanMessage,
    AIMessage,
    BaseMessage,
    InfoExtractionAgent,
    WorkflowAgent,
    EvaluationAgent,
    synthesize_results,
    classify_task_type,
    get_model
)
import time

logger = logging.getLogger("rdagent.web_ui_api")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Phenotype to Disease Controller API")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 存储会话状态（在实际生产环境中应使用Redis等）
sessions = {}

from web_ui_api_en import register_english_ui_routes


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    status: str


def translate_workflow_status(status_msg: str) -> str:
    """将WorkflowAgent返回的英文状态消息转换为中文"""
    if status_msg.startswith("Executing "):
        tool_name = status_msg.replace("Executing ", "").replace("...", "")
        tool_name_mapping = {
            "disease_diagnosis_tool": "疾病诊断分析"
        }
        display_name = tool_name_mapping.get(tool_name, tool_name)
        return f"正在进行{display_name}..."
    elif status_msg.startswith("Completed "):
        completed_item = status_msg.replace("Completed ", "")
        # 处理不同的完成状态
        if completed_item == "evaluation":
            return "[已完成] 诊断需求评估"
        else:
            tool_name_mapping = {
                "disease_diagnosis_tool": "疾病诊断分析"
            }
            display_name = tool_name_mapping.get(completed_item, completed_item)
            return f"[已完成] {display_name}"
    elif status_msg == "No workflow tool needed":
        return "当前查询无需进行疾病诊断"
    return status_msg  # 如果无法识别，返回原消息


class WorkflowStatusCallback:
    """WorkflowAgent状态回调类，用于管理状态消息"""
    
    def __init__(self, status_messages: List[str]):
        """
        初始化回调类
        
        Args:
            status_messages: 用于存储状态消息的列表
        """
        self.status_messages = status_messages
    
    def __call__(self, status_msg: str):
        """回调函数，用于在WorkflowAgent执行过程中发送状态更新"""
        # 将英文状态消息转换为中文
        translated_msg = translate_workflow_status(status_msg)
        self.status_messages.append(translated_msg)


async def controller_pipeline_stream(
    user_query: str, 
    conversation_messages: List[BaseMessage] = None
) -> AsyncGenerator[str, None]:
    """
    流式版本的controller_pipeline，通过生成器逐步输出结果
    
    Args:
        user_query: 用户查询
        conversation_messages: 对话历史
        
    Yields:
        str: 流式输出的文本块，格式为JSON: {"type": "status|content|error", "data": "..."}
    """
    try:
        start_time = time.time()
        

        # Phase 1: InfoExtractionAgent
        yield json.dumps({
            "type": "status",
            "data": "正在提取表型、症状或疾病信息...",
            "phase": "info_extraction"
        }, ensure_ascii=False) + "\n"
        
        completed_tools = []
        tool_call_counts = {}
        all_tool_results = {}
        
        info_extraction_agent = InfoExtractionAgent()
        info_extraction_results = await info_extraction_agent.run(user_query, conversation_messages)
        
        for tool_name in info_extraction_results.keys():
            completed_tools.append(tool_name)
            tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
        
        all_tool_results.update(info_extraction_results)
        
        yield json.dumps({
            "type": "status",
            "data": "[已完成] 信息提取",
            "phase": "info_extraction"
        }, ensure_ascii=False) + "\n"
        
        # Phase 2: WorkflowAgent
        yield json.dumps({
            "type": "status",
            "data": "正在评估诊断需求...",
            "phase": "workflow"
        }, ensure_ascii=False) + "\n"
        
        # 使用共享列表来存储状态消息（在回调函数和生成器之间共享）
        workflow_status_messages = []
        
        # 创建状态回调对象
        workflow_status_callback = WorkflowStatusCallback(workflow_status_messages)
        
        # 创建任务来运行WorkflowAgent，并同时监听状态消息
        workflow_agent = WorkflowAgent()
        workflow_task = asyncio.create_task(
            workflow_agent.run(
                user_query, 
                tool_results=all_tool_results, 
                conversation_messages=conversation_messages,
                status_callback=workflow_status_callback
            )
        )
        
        # 在任务执行期间，定期检查并输出状态消息
        last_status_count = 0
        while not workflow_task.done():
            # 检查是否有新的状态消息
            if len(workflow_status_messages) > last_status_count:
                # 输出新的状态消息
                for i in range(last_status_count, len(workflow_status_messages)):
                    yield json.dumps({
                        "type": "status",
                        "data": workflow_status_messages[i],
                        "phase": "workflow"
                    }, ensure_ascii=False) + "\n"
                last_status_count = len(workflow_status_messages)
            
            # 短暂休眠，避免CPU占用过高
            await asyncio.sleep(0.01)
        
        # 处理剩余的状态消息
        if len(workflow_status_messages) > last_status_count:
            for i in range(last_status_count, len(workflow_status_messages)):
                yield json.dumps({
                    "type": "status",
                    "data": workflow_status_messages[i],
                    "phase": "workflow"
                }, ensure_ascii=False) + "\n"
        
        # 获取最终结果
        workflow_info = await workflow_task
        
        # 检测是否有缺失的工具
        if workflow_info.get('missing_tools'):
            missing_tools = workflow_info['missing_tools']
            yield json.dumps({
                "type": "status",
                "data": "需要补充更多信息，正在重新提取信息...",
                "phase": "workflow"
            }, ensure_ascii=False) + "\n"
            
            retry_agent = InfoExtractionAgent(specified_tools=missing_tools)
            retry_results = await retry_agent.run(user_query, conversation_messages)
            
            for tool_name, result in retry_results.items():
                all_tool_results[tool_name] = result
                if tool_name not in completed_tools:
                    completed_tools.append(tool_name)
                tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
            
            # 清空之前的状态消息
            yield json.dumps({
                "type": "status",
                "data": "[已完成] 信息提取",
                "phase": "workflow"
            }, ensure_ascii=False) + "\n"

            yield json.dumps({
                "type": "status",
                "data": "正在重新评估诊断需求...",
                "phase": "workflow"
            }, ensure_ascii=False) + "\n"

            workflow_status_messages.clear()
            last_status_count = 0
            
            # 重新运行WorkflowAgent - 创建新实例确保状态回调正确
            workflow_agent = WorkflowAgent()
            workflow_task = asyncio.create_task(
                workflow_agent.run(
                    user_query, 
                    tool_results=all_tool_results, 
                    conversation_messages=conversation_messages,
                    status_callback=workflow_status_callback
                )
            )
            
            # 在任务执行期间，定期检查并输出状态消息
            while not workflow_task.done():
                # 检查是否有新的状态消息
                if len(workflow_status_messages) > last_status_count:
                    # 输出新的状态消息
                    for i in range(last_status_count, len(workflow_status_messages)):
                        yield json.dumps({
                            "type": "status",
                            "data": workflow_status_messages[i],
                            "phase": "workflow"
                        }, ensure_ascii=False) + "\n"
                    last_status_count = len(workflow_status_messages)
                
                # 短暂休眠，避免CPU占用过高
                await asyncio.sleep(0.05)
            
            # 处理剩余的状态消息
            if len(workflow_status_messages) > last_status_count:
                for i in range(last_status_count, len(workflow_status_messages)):
                    yield json.dumps({
                        "type": "status",
                        "data": workflow_status_messages[i],
                        "phase": "workflow"
                    }, ensure_ascii=False) + "\n"
            
            # 获取最终结果
            workflow_info = await workflow_task
        
        if workflow_info['workflow'] is not None:
            all_tool_results[workflow_info['workflow']] = workflow_info['result']
            completed_tools.append(workflow_info['workflow'])
            tool_call_counts[workflow_info['workflow']] = tool_call_counts.get(workflow_info['workflow'], 0) + 1
            yield json.dumps({
                "type": "status",
                "data": "[已完成] 疾病诊断分析",
                "phase": "workflow"
            }, ensure_ascii=False) + "\n"
        else:
            yield json.dumps({
                "type": "status",
                "data": "[已完成] 当前查询无需进行疾病诊断",
                "phase": "workflow"
            }, ensure_ascii=False) + "\n"

        # Phase 3: Task Classification（移动到Workflow之后，Synthesis之前）
        yield json.dumps({
            "type": "status",
            "data": "正在识别任务类型...",
            "phase": "classification"
        }, ensure_ascii=False) + "\n"
                
        if workflow_info['workflow'] is not None and workflow_info['workflow'] == 'disease_diagnosis_tool':
            task_type = "disease_diagnosis"
        else:
            # 这里还没有最终报告，使用workflow阶段的返回结果进行任务类型识别
            workflow_result_text = str(workflow_info.get('result', '') or '')
            task_type = await classify_task_type(user_query, workflow_result_text)

        if task_type == "disease_diagnosis" and workflow_info['workflow'] is None:
            # 检测到需要疾病诊断但未执行诊断工具，结束当前推理并重新开始
            yield json.dumps({
                "type": "status",
                "data": "检测到需要疾病诊断，但未执行诊断工具，将重新执行诊断流程...",
                "phase": "classification"
            }, ensure_ascii=False) + "\n"
            
            # 修改用户查询，添加强调信息
            enhanced_query = f"""{user_query}

**IMPORTANT: This query requires disease diagnosis. You MUST call the disease_diagnosis_tool to perform the diagnosis analysis."""
            
            # 递归调用，开启下一轮推理（直接继续流式输出，不发送done）
            async for chunk in controller_pipeline_stream(enhanced_query, conversation_messages):
                yield chunk
            
            return

        # 任务类型中文映射
        task_type_names = {
            "general_inquiry": "一般咨询",
            "phenotype_extraction": "表型提取",
            "disease_diagnosis": "疾病诊断",
            "disease_case_extraction": "疾病案例提取",
            "disease_information_retrieval": "疾病信息检索"
        }
        task_type_display = task_type_names.get(task_type, task_type)
        
        yield json.dumps({
            "type": "status",
            "data": f"[已完成] 任务类型识别: {task_type_display}",
            "phase": "classification"
        }, ensure_ascii=False) + "\n"
        
        # Phase 4: Evaluation（移动到Workflow之后；final_response使用workflow返回结果）
        evaluation_response_clean = ""
        if task_type == "disease_diagnosis":
            yield json.dumps({
                "type": "status",
                "data": "正在评估诊断结果...",
                "phase": "evaluation"
            }, ensure_ascii=False) + "\n"

            evaluation_agent = EvaluationAgent()
            workflow_result_for_eval = str(workflow_info.get('result', '') or '')
            evaluation_results = await evaluation_agent.run(
                user_query=user_query,
                all_tool_results=all_tool_results,
                conversation_messages=conversation_messages,
                final_response=workflow_result_for_eval,
                task_type=task_type
            )

            evaluation_response = evaluation_results.get('evaluation_response', '') or ''

            # 提取评估响应中的推理内容
            reasoning_matches = re.finditer(
                r'<think>(.*?)</think>',
                evaluation_response,
                re.DOTALL
            )

            # 发送所有推理内容
            for match in reasoning_matches:
                reasoning_text = match.group(1)
                if reasoning_text.strip():
                    yield json.dumps({
                        "type": "reasoning",
                        "data": reasoning_text,
                        "phase": "evaluation"
                    }, ensure_ascii=False) + "\n"

            # 移除所有<think>标签及其内容，得到可用于后续合成的评估文本
            evaluation_response_clean = re.sub(
                r'<think>.*?</think>',
                '',
                evaluation_response,
                flags=re.DOTALL
            )
            evaluation_response_clean = re.sub(r'.*?</think>', '', evaluation_response_clean, flags=re.DOTALL).strip()

            yield json.dumps({
                "type": "status",
                "data": "[已完成] 诊断结果评估",
                "phase": "evaluation"
            }, ensure_ascii=False) + "\n"

            # 将评估结果作为后续Synthesis的输入（不单独输出为content，避免与最终报告重复）
            if evaluation_response_clean:
                all_tool_results["diagnostic_evaluation"] = evaluation_response_clean

        # Phase 5: Synthesis（放到最后；评估结果作为输入）
        yield json.dumps({
            "type": "status",
            "data": "正在生成分析报告...",
            "phase": "synthesis"
        }, ensure_ascii=False) + "\n"

        model = await get_model("synthesizer")

        results_text = "\n\n".join([
            f"**{tool_name.upper()} Results:**\n{result}"
            for tool_name, result in all_tool_results.items()
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

        if 'disease_diagnosis_tool' in all_tool_results:
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
   - Provide a list of other differential diagnoses from the disease_diagnosis_tool results
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

        final_response_chunks = []
        accumulated_reasoning = []  
        reasoning_buffer = ""
        end_tag = '</think>'
        end_tag_len = len(end_tag)
        response_start_tag = 'FINAL_RESPONSE'
        response_start_tag_len = len(response_start_tag)
        found_split_point = False

        async for chunk in model.astream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                content = chunk.content

                if not found_split_point:
                    temp_content = reasoning_buffer + content
                    reasoning_buffer = ""

                    end_pos = temp_content.find(end_tag)
                    if end_pos != -1:
                        if end_pos > 0:
                            reasoning_data = temp_content[:end_pos]
                            accumulated_reasoning.append(reasoning_data)
                            yield json.dumps({
                                "type": "reasoning",
                                "data": reasoning_data,
                                "phase": "synthesis"
                            }, ensure_ascii=False) + "\n"

                        remaining = temp_content[end_pos + end_tag_len:]
                        remaining = remaining.replace(response_start_tag, '')
                        if remaining:
                            final_response_chunks.append(remaining)
                            yield json.dumps({
                                "type": "content",
                                "data": remaining,
                                "phase": "synthesis"
                            }, ensure_ascii=False) + "\n"

                        found_split_point = True
                        continue

                    response_start_pos = temp_content.find(response_start_tag)
                    if response_start_pos != -1:
                        if response_start_pos > 0:
                            reasoning_data = temp_content[:response_start_pos]
                            accumulated_reasoning.append(reasoning_data)
                            yield json.dumps({
                                "type": "reasoning",
                                "data": reasoning_data,
                                "phase": "synthesis"
                            }, ensure_ascii=False) + "\n"

                        remaining = temp_content[response_start_pos + response_start_tag_len:]
                        remaining = remaining.lstrip('\n\r\t ')
                        remaining = remaining.replace(response_start_tag, '')
                        if remaining:
                            final_response_chunks.append(remaining)
                            yield json.dumps({
                                "type": "content",
                                "data": remaining,
                                "phase": "synthesis"
                            }, ensure_ascii=False) + "\n"

                        found_split_point = True
                        continue

                    potential_tag = False
                    if len(temp_content) >= end_tag_len - 1:
                        for i in range(1, end_tag_len):
                            suffix = temp_content[-i:]
                            if end_tag.startswith(suffix):
                                potential_tag = True
                                reasoning_buffer = temp_content[-(end_tag_len - 1):]
                                output_reasoning = temp_content[:-(end_tag_len - 1)]
                                if output_reasoning:
                                    accumulated_reasoning.append(output_reasoning)
                                    yield json.dumps({
                                        "type": "reasoning",
                                        "data": output_reasoning,
                                        "phase": "synthesis"
                                    }, ensure_ascii=False) + "\n"
                                break

                    if not potential_tag and len(temp_content) >= response_start_tag_len - 1:
                        for i in range(1, response_start_tag_len):
                            suffix = temp_content[-i:]
                            if response_start_tag.startswith(suffix):
                                potential_tag = True
                                reasoning_buffer = temp_content[-(response_start_tag_len - 1):]
                                output_reasoning = temp_content[:-(response_start_tag_len - 1)]
                                if output_reasoning:
                                    accumulated_reasoning.append(output_reasoning)
                                    yield json.dumps({
                                        "type": "reasoning",
                                        "data": output_reasoning,
                                        "phase": "synthesis"
                                    }, ensure_ascii=False) + "\n"
                                break

                    if not potential_tag and temp_content:
                        accumulated_reasoning.append(temp_content)
                        yield json.dumps({
                            "type": "reasoning",
                            "data": temp_content,
                            "phase": "synthesis"
                        }, ensure_ascii=False) + "\n"
                else:
                    filtered_content = content.replace(response_start_tag, '')
                    if filtered_content:
                        final_response_chunks.append(filtered_content)
                        yield json.dumps({
                            "type": "content",
                            "data": filtered_content,
                            "phase": "synthesis"
                        }, ensure_ascii=False) + "\n"

        if reasoning_buffer.strip() and not found_split_point:
            accumulated_reasoning.append(reasoning_buffer)
            yield json.dumps({
                "type": "reasoning",
                "data": reasoning_buffer,
                "phase": "synthesis"
            }, ensure_ascii=False) + "\n"

        final_response = ''.join(final_response_chunks)
        if not final_response.strip():
            full_reasoning = ''.join(accumulated_reasoning)
            if full_reasoning.strip():
                yield json.dumps({
                    "type": "content",
                    "data": full_reasoning,
                    "phase": "synthesis"
                }, ensure_ascii=False) + "\n"

        yield json.dumps({
            "type": "status",
            "data": "[已完成] 分析报告生成",
            "phase": "synthesis"
        }, ensure_ascii=False) + "\n"
        
        total_time = time.time() - start_time
        
        yield json.dumps({
            "type": "status",
            "data": f"[全部完成] 分析已完成 (耗时: {total_time:.2f}秒)",
            "phase": "complete"
        }, ensure_ascii=False) + "\n"
        
        yield json.dumps({
            "type": "done",
            "data": "",
            "phase": "complete"
        }, ensure_ascii=False) + "\n"
        
    except Exception as e:
        # 仅在服务端记录完整堆栈，避免回传给前端造成隐私泄露
        logger.exception("controller_pipeline_stream failed")

        # 仅返回异常本身的信息（不包含Traceback）
        error_msg = str(e) if e else "Unknown error"
        yield json.dumps({
            "type": "error",
            "data": error_msg,
            "phase": "error"
        }, ensure_ascii=False) + "\n"


# 挂载前端页面目录（更专业命名：rdagent/）
import os
rdagent_dir = os.path.join(os.path.dirname(__file__), "rdagent")

# 先定义路由，确保 /rdagent/ 和 /rdagent 默认返回英文页面
@app.get("/rdagent", response_class=HTMLResponse)
@app.get("/rdagent/", response_class=HTMLResponse)
async def read_rdagent_root():
    """返回rdagent英文主页面（默认）"""
    if os.path.exists(rdagent_dir):
        index_file = os.path.join(rdagent_dir, "index_en.html")
        if os.path.exists(index_file):
            with open(index_file, "r", encoding="utf-8") as f:
                return f.read()
    return "<h1>Static files not found</h1>"

if os.path.exists(rdagent_dir):
    # 挂载静态文件目录（用于访问具体的HTML文件，如 index.html, index_en.html）
    app.mount("/rdagent", StaticFiles(directory=rdagent_dir), name="rdagent")
    # 兼容旧URL前缀（避免历史链接立即失效）
    app.mount("/static", StaticFiles(directory=rdagent_dir), name="static")
    # English UI routes (served from the same directory)
    register_english_ui_routes(app, rdagent_dir)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """返回主页面（英文，默认）"""
    index_file = os.path.join(rdagent_dir, "index_en.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Static files not found</h1>"


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式聊天API端点
    使用Server-Sent Events (SSE) 返回流式响应
    """
    # 获取或创建会话
    session_id = request.session_id or f"session_{len(sessions)}"
    if session_id not in sessions:
        sessions[session_id] = {
            "messages": [],
            "created_at": asyncio.get_event_loop().time()
        }
    
    session = sessions[session_id]
    
    # 添加用户消息到会话历史
    user_msg = HumanMessage(content=request.query)
    session["messages"].append(user_msg)
    
    # 准备对话历史（限制token数量）
    conversation_messages = session["messages"][:-1]  # 不包括当前用户消息
    
    # 检查token限制
    total_tokens = count_messages_tokens(conversation_messages)
    if total_tokens > MAX_TOKENS:
        # 移除最旧的消息
        while total_tokens > MAX_TOKENS and len(conversation_messages) > 0:
            conversation_messages.pop(0)
            total_tokens = count_messages_tokens(conversation_messages)
    
    async def generate():
        """生成SSE流"""
        full_response = ""
        try:
            async for chunk in controller_pipeline_stream(request.query, conversation_messages):
                # 发送SSE格式的数据
                yield f"data: {chunk}\n\n"
                
                # 解析chunk以累积完整响应
                try:
                    chunk_data = json.loads(chunk)
                    if chunk_data.get("type") == "content":
                        full_response += chunk_data.get("data", "")
                except:
                    pass
            
            # 添加AI响应到会话历史
            if full_response:
                ai_msg = AIMessage(content=full_response)
                session["messages"].append(ai_msg)
                
        except Exception as e:
            logger.exception("SSE generate() failed")

            # 仅返回异常本身的信息（不包含Traceback）
            error_msg = str(e) if e else "Unknown error"
            error_chunk = json.dumps({
                "type": "error",
                "data": error_msg,
                "phase": "error"
            }, ensure_ascii=False)
            yield f"data: {error_chunk}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    非流式聊天API端点（用于兼容）
    """
    session_id = request.session_id or f"session_{len(sessions)}"
    if session_id not in sessions:
        sessions[session_id] = {
            "messages": [],
            "created_at": asyncio.get_event_loop().time()
        }
    
    session = sessions[session_id]
    user_msg = HumanMessage(content=request.query)
    session["messages"].append(user_msg)
    
    conversation_messages = session["messages"][:-1]
    
    # 检查token限制
    total_tokens = count_messages_tokens(conversation_messages)
    if total_tokens > MAX_TOKENS:
        while total_tokens > MAX_TOKENS and len(conversation_messages) > 0:
            conversation_messages.pop(0)
            total_tokens = count_messages_tokens(conversation_messages)
    
    try:
        response = await controller_pipeline(request.query, conversation_messages)
        
        # 添加AI响应到会话历史
        ai_msg = AIMessage(content=response)
        session["messages"].append(ai_msg)
        
        return ChatResponse(
            session_id=session_id,
            response=response,
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """清除指定会话"""
    if session_id in sessions:
        del sessions[session_id]
        return {"status": "success", "message": f"会话 {session_id} 已清除"}
    return {"status": "not_found", "message": f"会话 {session_id} 不存在"}


@app.get("/api/sessions")
async def list_sessions():
    """列出所有会话"""
    return {
        "sessions": [
            {
                "session_id": sid,
                "message_count": len(sess["messages"]),
                "created_at": sess["created_at"]
            }
            for sid, sess in sessions.items()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    # 使用8080端口以匹配SSH隧道映射 (8899:localhost:8080)
    uvicorn.run(app, host="0.0.0.0", port=8080)
