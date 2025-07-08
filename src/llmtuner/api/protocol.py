from enum import Enum, unique
from typing import List, Optional
from ..extras.constants import MAX_TOKENS, TEMPERATURE, TOP_P

from pydantic import BaseModel


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    TOOL = "tool"


@unique
class Finish(str, Enum):
    STOP = "stop"


class ChatMessage(BaseModel):
    role: Role
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "string"
    chat_history: List[ChatMessage] = []
    query: str
    is_zh: bool
    topk: int
    fusion_weight: float
    is_es: bool
    max_tokens: Optional[int] = MAX_TOKENS
    temperature: Optional[float] = TEMPERATURE
    top_p: Optional[float] = TOP_P


class ChatCompletionTestRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    tools: Optional[list] = []
    do_sample: bool = True
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: int = 1
    max_tokens: Optional[int] = None
    stream: bool = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: str=None
    finish_reason: Finish


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    message: str=None
    finish_reason: Optional[Finish] = None


class ChatCompletionResponse(BaseModel):
    success: bool
    content: str = ''


class ChatCompletionStreamResponse(BaseModel):
    success: bool
    content: str = ''
