import asyncio
import logging
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    StateT_co,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableGenerator,
    RunnableLambda,
    RunnableSerializable,
)
from langchain_core.tools import BaseTool
from mem0 import AsyncMemory

from .context import LLMContext
from .request_analysis import RequestAnalysis
from .short_term_memory import BaseShortTermMemory, MemoryMessage

logger = logging.getLogger()

SystemPromptFn = Callable[["LLMContext"], str | Awaitable[str]]
SystemPromptType = str | SystemPromptFn


class RememberingLLM:
    def __init__(
        self,
        long_term_memory: AsyncMemory,
        short_term_memory: BaseShortTermMemory,
        main_llm: BaseChatModel,
        summarizer_llm: BaseChatModel | None,
        fast_llm: BaseChatModel | None,
        system_prompt: SystemPromptType = "",
        short_term_limit: int = 26,
        active_short_term_limit: int | None = None,
        top_k_memories: int = 10,
        tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
        middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    ):
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

        self._system_prompt = system_prompt

        self.long_term_memory = long_term_memory
        self.top_k_memories = top_k_memories
        self.short_term_memory = short_term_memory
        self.short_term_limit = short_term_limit
        self.active_short_term_limit = (
            active_short_term_limit
            if active_short_term_limit
            else short_term_limit // 2
        )

        self._llm_agent = create_agent(
            model=main_llm, tools=tools, middleware=middleware
        )

        self._summarizer_llm = summarizer_llm
        self._fast_llm = fast_llm

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                (
                    "system",
                    (
                        "# Информация из памяти:\n{memories}\n\n"
                        "Время в начале сообщений подставляется автоматические системой, а не тобой. Не пиши его"
                    ),
                ),
                ("placeholder", "{chat_history}"),
                ("placeholder", "{current_message}"),
            ]
        )

    def get_chain(self, user_id: str) -> RunnableSerializable:
        context = LLMContext()

        return (
            RunnableLambda(self._init_context).bind(context=context, user_id=user_id)
            | RunnableLambda(self._add_request_analysis).bind(context=context)
            | RunnableLambda(self._append_short_term_user_message).bind(context=context)
            | {
                "memories": RunnableLambda(self._get_memories),
                "chat_history": lambda _: context.chat_history,
                "current_message": lambda _: context.current_messages,
                "system_prompt": RunnableLambda(self._resolve_system_prompt).bind(
                    context=context
                ),
            }
            | self.prompt
            | debug_prompt
            | RunnableGenerator(self._stream_llm).bind(context=context)
            | StrOutputParser()
            | RunnableGenerator(self._upsert_short_term_memory).bind(context=context)
        )

    async def fetch_chat_history(self, *args, user_id: str) -> list[BaseMessage]:
        messages = []

        for message in await self.short_term_memory.get_dialog(user_id=user_id):
            if message.message.type == "human":
                message.message.content = format_message(message)

            messages.append(message.message)

        return messages

    async def search_memories(self, request: str, user_id):
        return await self.long_term_memory.search(
            request,
            filters={"user_id": user_id},
            top_k=self.top_k_memories,
        )

    async def _init_context(
        self,
        request: HumanMessage
        | str
        | list[str | dict[Any, Any]]
        | list[HumanMessage],
        context: LLMContext,
        user_id,
    ) -> list[HumanMessage]:
        context.clear()

        context.user_id = user_id
        context.remembering_llm = self

        if isinstance(request, HumanMessage):
            current_messages = [request]
        elif isinstance(request, list) and request and isinstance(request[0], HumanMessage):
            current_messages = request
        elif isinstance(request, list) and not request:
            raise ValueError("request list must not be empty")
        else:
            current_messages = [HumanMessage(content=request)]

        context.current_messages = current_messages
        return current_messages

    async def _resolve_system_prompt(self, context: "LLMContext") -> str:
        if callable(self._system_prompt):
            result = self._system_prompt(context)
            if hasattr(result, "__await__"):
                result = await result
            return result
        return self._system_prompt

    async def _add_request_analysis(
        self, current_messages: list[HumanMessage], context: LLMContext
    ):
        chat_history = await self.fetch_chat_history(user_id=context.user_id)

        prompt = (
            "Проанализируй сообщение пользователя.\n\n"
            f"История:\n{format_history(chat_history[-self.active_short_term_limit:])}\n\n"
            f"Сообщение:\n{format_history(current_messages)}"
        )
        analysis = await self._fast_llm.with_structured_output(RequestAnalysis).ainvoke(
            prompt
        )
        logger.info(f"RequestAnalysis: {analysis}")

        context.analysis = analysis
        context.chat_history = chat_history

        return current_messages

    async def _get_memories(self, context: LLMContext) -> str:
        if context.analysis.needs_memory_search:
            query = await self._get_search_query(
                request=context.analysis.search_query,
                chat_history=context.chat_history,
            )
        else:
            query = context.analysis.search_query

        result = []
        if query:
            relevant_memories = await self.search_memories(
                request=query,
                user_id=context.user_id,
            )

            for i, relevant_memory in enumerate(relevant_memories["results"]):
                result.append(f"{i}. {relevant_memory['memory']}")

        return "\n".join(result)

    async def _append_short_term_user_message(
        self, _, context: LLMContext
    ) -> LLMContext:
        for message in context.current_messages:
            await self.short_term_memory.add_message(
                user_id=context.user_id,
                message=message,
            )

        return context

    async def _upsert_short_term_memory(
        self, chunks: AsyncIterator[str], context: LLMContext
    ):
        async for chunk in chunks:
            yield chunk

        asyncio.create_task(self._compact_memory(context.user_id))

    async def _compact_memory(self, user_id: str):
        async with self._locks[user_id]:
            if (
                await self.short_term_memory.count_messages(user_id)
                >= self.short_term_limit
            ):
                chat_history = await self.short_term_memory.get_dialog(user_id)
                overflow = chat_history[: -self.active_short_term_limit]

                messages_for_mem0 = [
                    {
                        "role": (
                            "user"
                            if isinstance(m.message, HumanMessage)
                            else "assistant"
                        ),
                        "content": f"[{m.timestamp.isoformat()}]: {m.message.content}",
                    }
                    for m in overflow
                    if isinstance(
                        m.message, (HumanMessage, AIMessage)
                    )  # пропустить старый SystemMessage(summary)
                ]

                mem0_task = asyncio.create_task(
                    self.long_term_memory.add(messages_for_mem0, user_id=user_id)
                )

                try:
                    summary = await self._summary_dialog(
                        [msg.message for msg in overflow]
                    )
                except Exception:
                    logger.exception(
                        f"Summary failed for user_id={user_id}, keeping buffer as is"
                    )
                    mem0_task.cancel()
                    raise

                try:
                    await mem0_task
                except Exception:
                    logger.exception(f"Mem0 add failed for user_id={user_id}")
                    raise

                await self.short_term_memory.delete_messages(
                    user_id=user_id, ids=[msg.id for msg in overflow]
                )
                if summary:
                    await self.short_term_memory.add_message_back(
                        user_id=user_id, message=SystemMessage(content=summary)
                    )

    async def _summary_dialog(self, chat_history: list[BaseMessage]) -> str | None:
        if not self._summarizer_llm:
            return None

        old_summary = None
        rest = chat_history
        if chat_history and isinstance(chat_history[0], SystemMessage):
            old_summary = chat_history[0].content
            rest = chat_history[1:]

        messages = format_history(rest)

        prompt = (
            f"Текущая сводка диалога:\n{old_summary or '(сводки пока нет)'}\n\n"
            f"Новые сообщения, которые нужно учесть:\n{messages}\n\n"
            "Напиши ОБНОВЛЁННУЮ сводку с нуля, объединив старую сводку и новые сообщения "
            "в единый связный текст. Не дописывай к старой сводке — перепиши её заново целиком. "
            "Сохрани только самые важные факты и тон разговора, отбрось второстепенные детали. "
            "Уложись строго в 150-200 слов независимо от объёма исходного материала."
        )

        new_summary = await self._summarizer_llm.ainvoke(prompt)
        return new_summary.content

    async def _get_search_query(
        self, request: str, chat_history: list[BaseMessage]
    ) -> str:
        if not self._fast_llm:
            return request

        prompt = (
            f"История диалога:\n{format_history(chat_history)}\n\n"
            f"Текущее сообщение пользователя: {request}\n\n"
            "Сформулируй короткий поисковый запрос (не более 10 слов) для поиска "
            "релевантной информации о пользователе в базе памяти. "
            "Если сообщение самодостаточно, можешь использовать его как есть."
        )
        result = await self._fast_llm.ainvoke(prompt)
        return result.content

    async def _stream_llm(
        self, agent_input_stream: AsyncIterator[dict], context: LLMContext
    ) -> AsyncIterator[str]:
        final_messages = []
        agent_input: ChatPromptValue = None
        async for item in agent_input_stream:
            agent_input = item

        async for mode, chunk in self._llm_agent.astream(
            agent_input,
            stream_mode=["messages", "values"],
            config={"configurable": {"context": context}},
        ):
            if mode == "messages":
                token, _ = chunk
                if isinstance(token, AIMessageChunk) and token.content:
                    yield token.content

            elif mode == "values":
                final_messages = chunk["messages"]

        new_messages = final_messages[len(agent_input.messages) :]
        for msg in new_messages:
            await self.short_term_memory.add_message(context.user_id, msg)


def debug_prompt(a: ChatPromptValue):
    text = "debug_prompt:"

    for message in a.messages:
        text += f"\n{message.type}: {message.content}"
    text += f"\nmessages count: {len(a.messages)}"

    logger.info(text)
    return a


LLMContext.model_rebuild()


def format_history(chat_history: list[BaseMessage]):
    return "\n".join(
        f"{msg.type}: {describe_content(msg.content)}" for msg in chat_history
    )


def format_message(message: MemoryMessage):
    return f"[at {message.timestamp.isoformat()}] {describe_content(message.message.content)}"


def describe_content(content: str | list[dict]) -> str:
    """Текст + пометки медиа-блоков, единая логика для любого места,
    где content нужно превратить в читаемую строку."""
    text = extract_text(content)

    media_types = []
    if isinstance(content, list):
        media_types = [
            _describe_media_block(b) for b in content if b.get("type") != "text"
        ]

    suffix = f" [{', '.join(media_types)}]" if media_types else ""
    return f"{text}{suffix}"


def extract_text(content: str | list[str | dict[Any, Any]]) -> str:
    """Достаёт только текстовую часть из content, независимо от того,
    простая это строка или список мультимодальных блоков."""
    if isinstance(content, str):
        return content
    text_parts = [block["text"] for block in content if block.get("type") == "text"]
    return " ".join(text_parts)


def _describe_media_block(block: dict) -> str:
    block_type = block.get("type", "unknown")

    if block_type == "image_url":
        url = block.get("image_url", {}).get("url", "")
        if url.startswith("data:"):
            return url.split(";")[0].removeprefix("data:")
        return "image"

    if block_type == "input_audio":
        return block.get("input_audio", {}).get("format", "audio")

    if block_type in ("file", "document"):
        return block.get("mime_type") or block.get("source_type") or "file"

    return block_type
