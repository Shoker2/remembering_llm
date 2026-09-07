import asyncio
import datetime as dt
import json
from typing import TYPE_CHECKING

from langchain_core.messages import BaseMessage

from .base import _MESSAGE_TYPES, BaseShortTermMemory, MemoryMessage

if TYPE_CHECKING:
    import aiosqlite  # type: ignore


def _serialize_message(message: BaseMessage) -> str:
    return json.dumps({"type": message.type, "data": message.model_dump(mode="json")})


def _deserialize_message(raw: str) -> BaseMessage:
    obj = json.loads(raw)
    cls = _MESSAGE_TYPES.get(obj["type"], BaseMessage)
    return cls(**obj["data"])


def _check_aiosqlite() -> None:
    try:
        import aiosqlite  # type: ignore # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Для использования SqliteShortTermMemory установи пакет: pip install aiosqlite"
        ) from e


class SqliteShortTermMemory(BaseShortTermMemory):
    def __init__(self, db_path: str):
        super().__init__()
        _check_aiosqlite()
        import aiosqlite  # type: ignore

        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()  # доп. защита поверх WAL — сериализация записей

    async def initialize(self):
        """Вызвать один раз перед использованием (например, при старте приложения)."""
        import aiosqlite

        self._conn = await aiosqlite.connect(self.db_path)
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS short_term_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                seq REAL NOT NULL,
                message_json TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """)
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_seq ON short_term_messages(user_id, seq)"
        )
        await self._conn.commit()

    async def close(self):
        if self._conn:
            await self._conn.close()

    async def add_message(self, user_id, message: BaseMessage) -> MemoryMessage:
        async with self._lock:
            cursor = await self._conn.execute(
                "SELECT COALESCE(MAX(seq), -1) FROM short_term_messages WHERE user_id = ?",
                (user_id,),
            )
            (max_seq,) = await cursor.fetchone()
            new_seq = max_seq + 1
            timestamp = dt.datetime.now(dt.UTC)

            await self._conn.execute(
                "INSERT INTO short_term_messages (user_id, seq, message_json, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (user_id, new_seq, _serialize_message(message), timestamp.isoformat()),
            )
            await self._conn.commit()

            count_cursor = await self._conn.execute(
                "SELECT COUNT(*) FROM short_term_messages WHERE user_id = ?", (user_id,)
            )
            (count,) = await count_cursor.fetchone()

            return MemoryMessage(id=count - 1, message=message, timestamp=timestamp)

    async def add_message_back(self, user_id, message: BaseMessage) -> MemoryMessage:
        async with self._lock:
            cursor = await self._conn.execute(
                "SELECT COALESCE(MIN(seq), 1) FROM short_term_messages WHERE user_id = ?",
                (user_id,),
            )
            (min_seq,) = await cursor.fetchone()
            new_seq = min_seq - 1
            timestamp = dt.datetime.now(dt.UTC)

            await self._conn.execute(
                "INSERT INTO short_term_messages (user_id, seq, message_json, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (user_id, new_seq, _serialize_message(message), timestamp.isoformat()),
            )
            await self._conn.commit()

            return MemoryMessage(id=0, message=message, timestamp=timestamp)

    async def get_dialog(self, user_id) -> list[MemoryMessage]:
        cursor = await self._conn.execute(
            "SELECT message_json, timestamp FROM short_term_messages "
            "WHERE user_id = ? ORDER BY seq ASC",
            (user_id,),
        )
        rows = await cursor.fetchall()

        return [
            MemoryMessage(
                id=i,
                message=_deserialize_message(message_json),
                timestamp=dt.datetime.fromisoformat(timestamp),
            )
            for i, (message_json, timestamp) in enumerate(rows)
        ]

    async def delete_messages(self, user_id, ids: list[int]):
        async with self._lock:
            cursor = await self._conn.execute(
                "SELECT id FROM short_term_messages WHERE user_id = ? ORDER BY seq ASC",
                (user_id,),
            )
            rows = await cursor.fetchall()
            to_delete = [rows[i][0] for i in ids if i < len(rows)]

            if to_delete:
                placeholders = ",".join("?" * len(to_delete))
                await self._conn.execute(
                    f"DELETE FROM short_term_messages WHERE id IN ({placeholders})",
                    to_delete,
                )
                await self._conn.commit()

    async def last_message(self, user_id) -> MemoryMessage | None:
        cursor = await self._conn.execute(
            "SELECT message_json, timestamp FROM short_term_messages "
            "WHERE user_id = ? ORDER BY seq DESC LIMIT 1",
            (user_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        message_json, timestamp = row
        count_cursor = await self._conn.execute(
            "SELECT COUNT(*) FROM short_term_messages WHERE user_id = ?", (user_id,)
        )
        (count,) = await count_cursor.fetchone()

        return MemoryMessage(
            id=count - 1,
            message=_deserialize_message(message_json),
            timestamp=dt.datetime.fromisoformat(timestamp),
        )

    async def count_messages(self, user_id) -> int:
        cursor = await self._conn.execute(
            "SELECT COUNT(*) FROM short_term_messages WHERE user_id = ?", (user_id,)
        )
        (count,) = await cursor.fetchone()
        return count
