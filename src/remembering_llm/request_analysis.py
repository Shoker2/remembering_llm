from pydantic import BaseModel, Field


class RequestAnalysis(BaseModel):
    needs_memory_search: bool = Field(
        description="Нужно ли искать в долгосрочной памяти"
    )
    search_query: str = Field(
        default=None, description="Переформулированный поисковый запрос"
    )
