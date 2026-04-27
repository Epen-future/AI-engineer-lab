from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import json
import logging
import os
from typing import List, AsyncGenerator
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from core.schemas import ChatRequest, ChatResponse, Message
from core.agent import Agent, Step  # 使用你已经具备流式能力的 Agent
from core.auth import verify_api_key
import time

# ---------- 日志配置 ----------
import logging
from logging.handlers import RotatingFileHandler

# 创建日志目录
os.makedirs("logs", exist_ok=True)

# 配置根日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("logs/server.log", maxBytes=5*1024*1024, backupCount=3),
        logging.StreamHandler()  # 仍然输出到控制台
    ]
)
logger = logging.getLogger(__name__)

# ---------- 初始化 FastAPI ----------
app = FastAPI(title="AI Agent API", version="1.0.0")

# 初始化限流器
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
@limiter.limit("10/minute")
async def chat(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    logger.info(f"非流式请求 来源={request.client.host} 问题={request.query[:50]}...")
    try:
        history_dicts = build_history(request.history)
        thoughts = []
        def collect_thought(msg):
            thoughts.append(msg)
        
        answer = agent.run(
            user_query=request.query,
            history=history_dicts,
            status_callback=collect_thought
        )
        logger.info(f"非流式请求完成 回答长度={len(answer)}")
        return ChatResponse(answer=answer, thought_process=thoughts)
    except Exception as e:
        logger.error(f"非流式请求失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ---------- 2. 流式端点 (SSE) ----------
@app.post("/chat/stream")
@limiter.limit("10/minute")
async def chat_stream(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    logger.info(f"流式请求 来源={request.client.host} 问题={request.query[:50]}...")

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            history_dicts = build_history(request.history)
            for item in agent.run_stream(request.query, history=history_dicts):
                if isinstance(item, Step):
                    yield f"event: thought\ndata: {json.dumps({'message': item.message})}\n\n"
                else:
                    yield f"event: token\ndata: {json.dumps({'token': item})}\n\n"
                await asyncio.sleep(0.02)
            yield "event: done\ndata: [DONE]\n\n"
            logger.info("流式请求完成")
        except Exception as e:
            logger.error(f"流式请求失败: {str(e)}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
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