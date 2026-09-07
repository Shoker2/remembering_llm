import base64
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from io import BytesIO
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger()

MEDIA_TOOLS: set[str] = set()


def media_tool(func):
    """Регистрирует tool как источник медиафайлов.
    Такой tool должен вернуть строку JSON: {"media_id": "...", "media_type": "image"}
    и заранее положить данные через storage.put_media(...)."""
    MEDIA_TOOLS.add(func.name)
    return func


class BaseMediaStorage(ABC):
    @abstractmethod
    async def put_media(self, buffer: BytesIO, mime_type: str) -> str:
        """Сохраняет данные, возвращает media_id."""
        ...

    @abstractmethod
    async def get_media(self, media_id: str) -> tuple[BytesIO, str] | None:
        """Возвращает (buffer, mime_type) по id, либо None, если не найдено.
        Реализация сама решает, удалять ли запись после чтения."""
        ...


class MediaInjectionMiddleware(AgentMiddleware):
    def __init__(self, storage: BaseMediaStorage):
        self.storage = storage

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        result = await handler(request)
        tool_name = request.tool_call.get("name")

        if not isinstance(result, ToolMessage) or tool_name not in MEDIA_TOOLS:
            return result

        media_info = await self._extract_media_info(result.content)
        if media_info is None:
            logger.warning(f"Не удалось извлечь медиа-данные из результата {tool_name}")
            return result

        new_message = self._build_message_for_media(media_info)
        if new_message is None:
            return result

        return Command(update={"messages": [result, new_message]})

    async def _extract_media_info(self, content: str) -> dict | None:
        try:
            info = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Результат media_tool не является валидным JSON")
            return None

        media_id = info.get("media_id")
        if media_id is None:
            logger.warning("В результате отсутствует media_id")
            return None

        result = await self.storage.get_media(media_id)
        if result is None:
            logger.warning(f"media_id '{media_id}' не найден в хранилище")
            return None

        buffer, mime_type = result
        return {
            "buffer": buffer,
            "mime_type": mime_type,
            "media_type": info.get("media_type", "image"),
        }

    def _build_message_for_media(self, media_info: dict) -> HumanMessage | None:
        buffer: BytesIO = media_info["buffer"]
        mime = media_info["mime_type"]

        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode("utf-8")

        return HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded}"},
                }
            ]
        )
