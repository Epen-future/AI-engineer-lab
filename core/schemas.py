from pydantic import BaseModel, Field
from typing import List, Optional

class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str

class ChatRequest(BaseModel):
    query: str = Field(..., description="用户最新输入")
    history: Optional[List[Message]] = Field(default=[], description="对话历史，按时间顺序")

class ChatResponse(BaseModel):
    answer: str
    thought_process: Optional[List[str]] = Field(default=[], description="Agent思考步骤")