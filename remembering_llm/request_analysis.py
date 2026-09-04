from typing import Literal

from pydantic import BaseModel, Field


class RequestAnalysis(BaseModel):
    intent: Literal["talk", "command"] = Field(
        description="Категория сообщения пользователя"
    )
    needs_memory_search: bool = Field(
        description="Нужно ли искать в долгосрочной памяти"
    )
    search_query: str = Field(
        default=None, description="Переформулированный поисковый запрос"
    )
