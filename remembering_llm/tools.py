import logging

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from .context import LLMContext

logger = logging.getLogger()


@tool
async def search_memory(query: str, config: RunnableConfig) -> str:
    """Найти конкретную информацию о пользователе в долговременной памяти.
    Сначала посмотри, нет ли уже нужного в предоставленном контексте (memories) —
    вызывай только если там этого нет, а факт из прошлого действительно нужен для ответа
    """
    context: LLMContext = config["configurable"]["context"]
    logger.info(f'search_memory "{query}" for {context.user_id}')

    results = await context.remembering_llm.long_term_memory.search(
        query,
        filters={
            "user_id": context.user_id,
        },
    )

    return "\n".join(r["memory"] for r in results.get("results", []))


@tool
async def add_memory(fact: str, config: RunnableConfig) -> str:
    """Сохранить факт о пользователе в долговременную память прямо сейчас.
    Используй только если пользователь явно просит запомнить что-то,
    или называет что-то критично важное, что нельзя упустить.
    Остальное попадёт в память автоматически, вызывать на каждый факт не нужно"""
    context: LLMContext = config["configurable"]["context"]

    logger.info(f'add_memory "{fact}" for {context.user_id}')

    await context.remembering_llm.long_term_memory.add(
        [{"role": "user", "content": fact}],
        user_id=context.user_id,
    )

    return "Запомнено"
