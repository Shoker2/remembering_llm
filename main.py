import asyncio
import json
import logging
import os
from io import BytesIO

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool
from mem0 import AsyncMemory
from mem0.configs.base import EmbedderConfig, LlmConfig, MemoryConfig, VectorStoreConfig

from remembering_llm import RememberingLLM
from remembering_llm.llm_models import MainLLMModel, SummarizerLLMModel
from remembering_llm.middleware.media_injection import (
    InMemoryMediaStorage,
    MediaInjectionMiddleware,
    media_tool,
)
from remembering_llm.short_term_memory import SqliteShortTermMemory
from remembering_llm.tools import add_memory, search_memory

logging.basicConfig(level=logging.INFO)
load_dotenv()

USER_ID = "alice"
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL = os.getenv("MODEL")
WEATHER_API_TOKEN = os.getenv("WEATHER_API_TOKEN")

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


@tool
async def get_weather(city: str) -> str:
    """Получить текущую погоду в указанном городе. Название города на англиском"""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "http://api.weatherapi.com/v1/current.json",
            params={
                "q": city,
                "key": WEATHER_API_TOKEN,
            },
        )

        current = resp.json()["current"]
        return f"{current['temp_c']}°C, {current['condition']['text']}, ветер {current['wind_kph']} км/ч"


media_storage = InMemoryMediaStorage()
media_injection_middleware = MediaInjectionMiddleware(storage=media_storage)

MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024


@media_tool
@tool
async def download_image(url: str) -> str:
    """Скачать изображение по URL из интернета.
    Используй, когда пользователь прислал ссылку на картинку или просит показать
    изображение по конкретному адресу."""

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        return json.dumps(
            {"error": f"Не удалось скачать: сервер вернул {e.response.status_code}"}
        )
    except httpx.RequestError as e:
        return json.dumps({"error": f"Не удалось скачать: {e}"})

    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        return json.dumps(
            {
                "error": f"По ссылке не изображение, а {content_type or 'неизвестный тип'}"
            }
        )

    if len(response.content) > MAX_IMAGE_SIZE_BYTES:
        return json.dumps({"error": "Изображение слишком большое (>10 МБ)"})

    mime_type = content_type.split(";")[0].strip()  # отсекаем charset и т.п., если есть
    buffer = BytesIO(response.content)

    media_id = await media_storage.put_media(buffer, mime_type=mime_type)

    return json.dumps({"media_id": media_id, "media_type": "image"})


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
    tools=[add_memory, search_memory, get_weather, download_image],
    middleware=[media_injection_middleware],
)


async def ask_llm(request: str):
    llm_chain = llm.get_chain(USER_ID)

    # print(await llm_chain.ainvoke(request))
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
    # print(await llm.fetch_chat_history(user_id=USER_ID))
    # print(await llm.get_analyzed_request("Помнишь ли кто ты и кто я?", user_id=USER_ID))
    # print(await llm.get_analyzed_request("Поставь чайник на плиту", user_id=USER_ID))
    print()


asyncio.run(main())
# asyncio.run(test())
