import json
import logging
from collections.abc import Awaitable, Callable
from io import BytesIO
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from .storage import BaseMediaStorage

logger = logging.getLogger()

ContentBlockBuilder = Callable[[BytesIO, str], list[dict] | dict]

# имя tool -> builder, знающий, как собрать content-блок(и) под этот тип медиа
MEDIA_TOOLS: dict[str, ContentBlockBuilder] = {}


def media_tool(builder: ContentBlockBuilder):
    """Регистрирует tool как источник медиафайлов.

    Такой tool должен вернуть JSON: {"media_id": "...", "media_type": "..."}
    и заранее положить данные через storage.put_media(...).

    builder — функция (buffer, mime_type) -> content-блок(и), определяющая,
    как именно данные превращаются в формат, понятный конкретной модели.
    По умолчанию — стандартный image_url-блок.

    Пример:
        @media_tool(_build_image_block)
        @tool
        async def generate_chart(data: str) -> str: ...

        # или с дефолтным builder'ом для картинок:
        @media_tool()
        @tool
        async def get_local_image(path: str) -> str: ...
    """

    def decorator(func):
        MEDIA_TOOLS[func.name] = builder
        return func

    return decorator


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

        builder = MEDIA_TOOLS.get(tool_name)
        if not isinstance(result, ToolMessage) or builder is None:
            return result

        media_info = await self._extract_media_info(result.content)
        if media_info is None:
            logger.warning(f"Не удалось извлечь медиа-данные из результата {tool_name}")
            return result

        new_message = self._build_message(media_info, builder)
        if new_message is None:
            return result

        return Command(update={"messages": [result, new_message]})

    async def _extract_media_info(self, content: str) -> dict | None:
        try:
            info = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Результат media_tool не является валидным JSON")
            return None

        media_id = info["media_id"]
        media_type = info["media_type"]

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
            "media_type": media_type,
        }

    def _build_message(
        self, media_info: dict, builder: ContentBlockBuilder
    ) -> HumanMessage | None:
        try:
            block = builder(media_info["buffer"], media_info["mime_type"], media_info)
        except Exception:
            logger.exception("Ошибка при сборке content-блока для медиа")
            return None

        content = block if isinstance(block, list) else [block]
        return HumanMessage(content=content)
