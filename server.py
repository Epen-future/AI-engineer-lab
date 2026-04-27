import json
import logging
from typing import List, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from core.schemas import ChatRequest, ChatResponse, Message
from core.agent import Agent, Step  # 使用你已经具备流式能力的 Agent
import time

# ---------- 日志配置 ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------- 初始化 FastAPI ----------
app = FastAPI(title="AI Agent API", version="1.0.0")

# 允许跨域（前端调试用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局 Agent 实例（生产环境可改为依赖注入或单例池）
agent = Agent()

# ---------- 辅助函数 ----------
def build_history(history: List[Message]) -> list:
    """将 Pydantic 模型转换为 Agent 可识别的字典列表"""
    return [{"role": msg.role, "content": msg.content} for msg in history]

# ---------- 1. 非流式端点 ----------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"收到非流式请求: {request.query[:50]}...")
        history_dicts = build_history(request.history)

        # 调用 Agent 的同步 run 方法
        thoughts = []
        def collect_thought(msg):
            thoughts.append(msg)
        
        answer = agent.run(
            user_query=request.query,
            history=history_dicts,
            status_callback=collect_thought
        )
        
        return ChatResponse(answer=answer, thought_process=thoughts)
    except Exception as e:
        logger.error(f"非流式请求处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------- 2. 流式端点 (SSE) ----------
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            logger.info(f"收到流式请求: {request.query[:50]}...")
            history_dicts = build_history(request.history)
            
            # 运行流式 Agent，逐个产出 Step 或 token
            for item in agent.run_stream(request.query, history=history_dicts):
                if isinstance(item, Step):
                    # 思考步骤：通过 type: thought 事件发送
                    yield f"event: thought\ndata: {json.dumps({'message': item.message})}\n\n"
                else:
                    # 答案 token：通过 type: token 事件发送
                    yield f"event: token\ndata: {json.dumps({'token': item})}\n\n"
                time.sleep(0.02)  # 可选：给前端渲染时间，实现打字机效果
            
            # 发送结束事件
            yield "event: done\ndata: [DONE]\n\n"
        except Exception as e:
            logger.error(f"流式请求处理失败: {str(e)}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # 禁用 Nginx 缓冲
        }
    )

# ---------- 健康检查 ----------
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}

# ---------- 启动入口 ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)