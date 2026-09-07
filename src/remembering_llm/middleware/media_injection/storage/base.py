from abc import ABC, abstractmethod
from io import BytesIO


class BaseMediaStorage(ABC):
    @abstractmethod
    async def put_media(self, buffer: BytesIO, mime_type: str) -> str:
        """Сохраняет данные, возвращает media_id."""
        ...

    @abstractmethod
    async def get_media(self, media_id: str) -> tuple[BytesIO, str] | None:
        """Возвращает (buffer, mime_type) по id, либо None, если не найдено.
        Реализация сама решает, удалять ли запись после чтения."""
        ...
