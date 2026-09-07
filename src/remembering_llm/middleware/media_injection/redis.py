import base64
import json
import uuid
from io import BytesIO
from typing import TYPE_CHECKING

from .middleware import BaseMediaStorage

if TYPE_CHECKING:
    from redis.asyncio import Redis  # type: ignore


class RedisMediaStorage(BaseMediaStorage):
    def __init__(self, redis_client: "Redis", ttl_seconds: int = 300):
        try:
            import redis.asyncio  # type: ignore  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Для использования RedisMediaStorage установи пакет: pip install redis"
            ) from e

        self.redis = redis_client
        self.ttl = ttl_seconds

    async def put_media(self, buffer: BytesIO, mime_type: str) -> str:
        media_id = str(uuid.uuid4())
        buffer.seek(0)
        await self.redis.set(
            f"media:{media_id}",
            json.dumps(
                {
                    "data": base64.b64encode(buffer.read()).decode(),
                    "mime_type": mime_type,
                }
            ),
            ex=self.ttl,
        )
        return media_id

    async def get_media(self, media_id: str) -> tuple[BytesIO, str] | None:
        raw = await self.redis.get(f"media:{media_id}")
        if raw is None:
            return None

        await self.redis.delete(f"media:{media_id}")
        data = json.loads(raw)
        return BytesIO(base64.b64decode(data["data"])), data["mime_type"]
