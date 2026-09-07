# RememberingLLM

LangChain/LangGraph-агент с двухуровневой памятью: краткосрочный буфер диалога
и долговременная память на Mem0, с автокомпакцией, суммаризацией, tool calling
и поддержкой медиафйлов.

## Install

```bash
pip install remembering_llm
```

## Quick start

```python
from remembering_llm import RememberingLLM
from remembering_llm.short_term_memory import SqliteShortTermMemory
from remembering_llm.tools import add_memory, search_memory
from mem0 import AsyncMemory

short_term_memory = SqliteShortTermMemory("short_term.db")
await short_term_memory.initialize()

llm = RememberingLLM(
    long_term_memory=AsyncMemory.from_config(mem0_config),
    short_term_memory=short_term_memory,
    main_llm=main_llm,
    summarizer_llm=summarizer_llm,
    fast_llm=fast_llm,
    system_prompt="Ты ассистент пользователя",
    tools=[search_memory, add_memory],
)

chain = llm.get_chain(user_id="alice")
async for chunk in chain.astream("Привет!"):
    print(chunk, end="", flush=True)
```

## How it works

- **Short-term** — последние сообщения диалога, хранятся в `BaseShortTermMemory` (in-memory или SQLite).
- **Long-term** — консолидированные факты о пользователе в Mem0, извлекаются семантическим поиском по запросу.
- При переполнении буфера (`short_term_limit`) старая часть целиком уходит в Mem0 и сжимается в summary; активный хвост (`active_short_term_limit`) остаётся как есть.

## Tools

Обычные LangChain tools:

```python
@tool
async def get_weather_async(city: str) -> str:
    """Получить текущую погоду в городе."""
    ...
```

Доступ к текущему пользователю внутри tool — через `RunnableConfig`, не как
параметр, который заполняет модель:

```python
@tool
async def my_tool(query: str, config: RunnableConfig) -> str:
    context: LLMContext = config["configurable"]["context"]
    user_id = context.user_id
    ...
```

Готовые: `search_memory` (явный поиск по требованию модели) и `add_memory`
(немедленное сохранение факта, в обход обычной компакции) из `from remembering_llm.tools import add_memory, search_memory`

## Медиафайлы

```python
message = HumanMessage(content=[
    {"type": "text", "text": "Что на фото?"},
    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
])
async for chunk in chain.astream(message):
    ...
```

Поддержка зависит от модели — не все модели за OpenRouter понимают vision.

Для tools, которые сами генерируют/загружают изображение и должны показать
его модели в текущем ответе — `@media_tool` + `MediaInjectionMiddleware`
(см. `remembering_llm/media.py`).

## License

MIT