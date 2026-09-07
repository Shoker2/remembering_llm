import uuid
from io import BytesIO

from .middleware import BaseMediaStorage


class InMemoryMediaStorage(BaseMediaStorage):
    def __init__(self):
        self._storage: dict[str, dict] = {}

    async def put_media(self, buffer: BytesIO, mime_type: str) -> str:
        media_id = str(uuid.uuid4())
        self._storage[media_id] = {"buffer": buffer, "mime_type": mime_type}
        return media_id

    async def get_media(self, media_id: str) -> tuple[BytesIO, str] | None:
        stored = self._storage.pop(media_id, None)
        if stored is None:
            return None
        return stored["buffer"], stored["mime_type"]
