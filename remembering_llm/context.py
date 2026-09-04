from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

from .request_analysis import RequestAnalysis


class LLMContext(BaseModel):
    user_id: str | None = None
    request: str | None = None
    current_message: HumanMessage | None = None
    analysis: RequestAnalysis | None = None
    chat_history: list[BaseMessage] | None = None

    def clear(self) -> None:
        """Сбрасывает все поля модели в None, независимо от их набора."""
        for field_name in self.__class__.model_fields:
            setattr(self, field_name, None)
