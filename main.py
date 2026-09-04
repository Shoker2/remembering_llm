import asyncio
import logging
import os

from dotenv import load_dotenv
from mem0 import AsyncMemory
from mem0.configs.base import EmbedderConfig, LlmConfig, MemoryConfig, VectorStoreConfig

from remembering_llm import RememberingLLM
from remembering_llm.llm_models import MainLLMModel, SummarizerLLMModel
from remembering_llm.short_term_memory import SqliteShortTermMemory

logging.basicConfig(level=logging.INFO)
load_dotenv()

USER_ID = "alice"
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL = os.getenv("MODEL")

CONFIG = MemoryConfig(
    llm=LlmConfig(
        provider="openai",
        config={
            "model": MODEL,
            "openai_base_url": BASE_URL,
            "api_key": API_KEY,
            "temperature": 0.2,
            "max_tokens": 2000,
        },
    ),
    embedder=EmbedderConfig(
        provider="huggingface",
        config={
            "model": "BAAI/bge-m3",
        },
    ),
    vector_store=VectorStoreConfig(
        provider="chroma",
        config={
            "collection_name": "memories",
            "path": "./tmp/chroma",
        },
    ),
)

short_term_memory = SqliteShortTermMemory("./tmp/chroma/short_term.db")

llm = RememberingLLM(
    long_term_memory=AsyncMemory.from_config(CONFIG.model_dump()),
    short_term_memory=short_term_memory,
    system_prompt=(
        "Ты асситент пользователя, играющий роль Цундере с именем Моника.\n"
        "Твои ответы должны выглядеть как ответы в реальной жизни, то есть без смайликов, без форматирования и большого объёма текста"
    ),
    main_llm=MainLLMModel(
        base_url=BASE_URL,
        model=MODEL,
        api_key=API_KEY,
        timeout=30,
        max_retries=3,
    ),
    summarizer_llm=SummarizerLLMModel(
        base_url=BASE_URL,
        model=MODEL,
        api_key=API_KEY,
        timeout=30,
        max_retries=3,
    ),
    fast_llm=SummarizerLLMModel(
        base_url=BASE_URL,
        model=MODEL,
        api_key=API_KEY,
        max_retries=3,
        timeout=30,
        temperature=0.2,
    ),
)


async def ask_llm(request: str):
    llm_chain = llm.get_chain(USER_ID)

    async for chunk in llm_chain.astream(request):
        print(chunk, end="", flush=True)


async def main():
    await short_term_memory.initialize()

    print("\nГотово\n")

    while True:
        await ask_llm(input())
        print()


async def test():
    await short_term_memory.initialize()

    print()
    print(await llm.fetch_chat_history(user_id=USER_ID))
    # print(await llm.get_analyzed_request("Помнишь ли кто ты и кто я?", user_id=USER_ID))
    # print(await llm.get_analyzed_request("Поставь чайник на плиту", user_id=USER_ID))
    print()


asyncio.run(main())
# asyncio.run(test())
