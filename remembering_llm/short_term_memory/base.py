import datetime as dt
from abc import ABC, abstractmethod

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel

_MESSAGE_TYPES: dict[str, type[BaseMessage]] = {
    "human": HumanMessage,
    "ai": AIMessage,
    "system": SystemMessage,
    "tool": ToolMessage,
    "function": FunctionMessage,
    "chat": ChatMessage,
}


class MemoryMessage(BaseModel):
    id: int
    message: BaseMessage
    timestamp: dt.datetime


class BaseShortTermMemory(ABC):
    @abstractmethod
    async def add_message(self, user_id, message: BaseMessage) -> MemoryMessage:
        pass

    @abstractmethod
    async def add_message_back(self, user_id, message: BaseMessage) -> MemoryMessage:
        pass

    @abstractmethod
    async def get_dialog(self, user_id) -> list[MemoryMessage]:
        pass

    @abstractmethod
    async def last_message(self, user_id) -> MemoryMessage | None:
        pass

    @abstractmethod
    async def count_messages(self, user_id) -> int:
        pass

    @abstractmethod
    async def delete_messages(self, user_id, ids: list):
        pass
