"""Самопроверка ветвления _init_context: str / HumanMessage / content-блоки / список HumanMessage."""

import asyncio

from langchain_core.messages import HumanMessage

from remembering_llm.context import LLMContext
from remembering_llm.main import RememberingLLM


async def demo():
    ctx = LLMContext()

    # str -> одно HumanMessage
    result = await RememberingLLM._init_context(None, "привет", ctx, "u1")
    assert result == [HumanMessage(content="привет")]
    assert ctx.current_messages == result

    # готовый HumanMessage -> оборачивается в список
    hm = HumanMessage(content="привет")
    result = await RememberingLLM._init_context(None, hm, ctx, "u1")
    assert result == [hm]

    # content-блоки одного мультимодального сообщения -> одно HumanMessage
    blocks = [{"type": "text", "text": "что на фото?"}, {"type": "image_url", "image_url": {"url": "x"}}]
    result = await RememberingLLM._init_context(None, blocks, ctx, "u1")
    assert result == [HumanMessage(content=blocks)]

    # список HumanMessage -> несколько сообщений за один ход
    msgs = [HumanMessage(content="первое"), HumanMessage(content="второе")]
    result = await RememberingLLM._init_context(None, msgs, ctx, "u1")
    assert result == msgs
    assert ctx.current_messages == msgs

    # пустой список -> ошибка, а не тихий проглот
    try:
        await RememberingLLM._init_context(None, [], ctx, "u1")
    except ValueError:
        pass
    else:
        raise AssertionError("empty list must raise ValueError")

    print("OK")


if __name__ == "__main__":
    asyncio.run(demo())
