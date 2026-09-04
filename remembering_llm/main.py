import asyncio
import logging
from collections import defaultdict
from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableGenerator,
    RunnableLambda,
    RunnableSerializable,
)
from mem0 import AsyncMemory

from .context import LLMContext
from .llm_models import MainLLMModel, SearchQueryLLMModel, SummarizerLLMModel
from .request_analysis import RequestAnalysis
from .short_term_memory import BaseShortTermMemory, MemoryMessage

# TODO: Возможность использовать tools
logger = logging.getLogger()


class RememberingLLM:
    def __init__(
        self,
        long_term_memory: AsyncMemory,
        short_term_memory: BaseShortTermMemory,
        main_llm: MainLLMModel,
        summarizer_llm: SummarizerLLMModel | None,
        fast_llm: SearchQueryLLMModel | None,
        system_prompt: str = "",
        short_term_limit: int = 20,
        active_short_term_limit: int | None = None,
        top_k_memories: int = 10,
    ):
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

        self.long_term_memory = long_term_memory
        self.top_k_memories = top_k_memories
        self.short_term_memory = short_term_memory
        self.short_term_limit = short_term_limit
        self.active_short_term_limit = (
            active_short_term_limit
            if active_short_term_limit
            else short_term_limit // 2
        )

        self.llm = main_llm
        self._summarizer_llm = summarizer_llm
        self._fast_llm = fast_llm

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
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
                "current_message": lambda _: [context.current_message],
            }
            | self.prompt
            | debug_prompt
            | RunnableGenerator(self._stream_llm_with_lock).bind(context=context)
            | StrOutputParser()
            | RunnableGenerator(self._upsert_short_term_memory).bind(context=context)
        )

    async def fetch_chat_history(self, *args, user_id: str) -> list[BaseMessage]:
        messages = []

        for message in await self.short_term_memory.get_dialog(user_id=user_id):
            if message.message.type == "human":
                message.message.content = self.format_message(message)

            messages.append(message.message)

        return messages

    @staticmethod
    def format_history(chat_history: list[BaseMessage]):
        return "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])

    @staticmethod
    def format_message(message: MemoryMessage):
        return f"[{message.timestamp.isoformat()}] {message.message.content}"

    async def search_memories(self, request: str, user_id):
        return await self.long_term_memory.search(
            request,
            filters={"user_id": user_id},
            top_k=self.top_k_memories,
        )

    async def _init_context(self, request, context: LLMContext, user_id):
        context.clear()

        context.user_id = user_id
        context.request = request

        current_message = HumanMessage(content=request)
        context.current_message = current_message

        return request

    async def _add_request_analysis(self, request, context: LLMContext):
        chat_history = await self.fetch_chat_history(user_id=context.user_id)

        prompt = (
            "Проанализируй сообщение пользователя.\n\n"
            f"История:\n{self.format_history(chat_history[-self.active_short_term_limit:])}\n\nСообщение: {request}"
        )
        analysis = await self._fast_llm.with_structured_output(RequestAnalysis).ainvoke(
            prompt
        )
        logger.info(f"RequestAnalysis: {analysis}")

        context.analysis = analysis
        context.chat_history = chat_history

        return request

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
        self, request: str, context: LLMContext
    ) -> LLMContext:
        await self.short_term_memory.add_message(
            user_id=context.user_id,
            message=HumanMessage(content=context.request),
        )

        return context

    async def _upsert_short_term_memory(
        self, chunks: AsyncIterator[str], context: LLMContext
    ):
        full_text = ""

        async for chunk in chunks:
            full_text += chunk
            yield chunk

        await self.short_term_memory.add_message(
            user_id=context.user_id,
            message=AIMessage(content=full_text),
        )

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

        messages = self.format_history(rest)

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
            f"История диалога:\n{self.format_history(chat_history)}\n\n"
            f"Текущее сообщение пользователя: {request}\n\n"
            "Сформулируй короткий поисковый запрос (не более 10 слов) для поиска "
            "релевантной информации о пользователе в базе памяти. "
            "Если сообщение самодостаточно, можешь использовать его как есть."
        )
        result = await self._fast_llm.ainvoke(prompt)
        return result.content

    async def _stream_llm_with_lock(
        self, input_iter: AsyncIterator, context: LLMContext
    ) -> AsyncIterator[str]:
        async for prompt_value in input_iter:
            async with self._locks[context.user_id]:
                try:
                    async for chunk in self.llm.astream(prompt_value):
                        yield chunk
                finally:
                    pass


def debug_prompt(a):
    text = "debug_prompt:"

    for message in a.messages:
        text += f"\n{message.type}: {message.content}"
    text += f"\nmessages count: {len(a.messages)}"

    logger.info(text)
    return a
