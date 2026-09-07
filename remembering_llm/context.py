from typing import TYPE_CHECKING

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict

from .request_analysis import RequestAnalysis

if TYPE_CHECKING:
    from .main import RememberingLLM


class LLMContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    remembering_llm: "RememberingLLM | None" = None
    user_id: str | None = None
    request: str | None = None
    current_message: HumanMessage | None = None
    analysis: RequestAnalysis | None = None
    chat_history: list[BaseMessage] | None = None

    def clear(self) -> None:
        """Сбрасывает все поля модели в None, независимо от их набора."""
        for field_name in self.__class__.model_fields:
            setattr(self, field_name, None)
