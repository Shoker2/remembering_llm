import datetime as dt
from collections import defaultdict

from langchain_core.messages import BaseMessage

from .base import BaseShortTermMemory, MemoryMessage


class InMemoryShortTermMemory(BaseShortTermMemory):
    def __init__(self):
        super().__init__()
        self.memory = defaultdict(list[MemoryMessage])

    async def add_message(self, user_id, message: BaseMessage) -> MemoryMessage:
        msg_id = len(self.memory[user_id])
        self.memory[user_id].append(
            MemoryMessage(id=0, message=message, timestamp=dt.datetime.now(dt.UTC))
        )

        return MemoryMessage(
            id=msg_id, message=message, timestamp=dt.datetime.now(dt.UTC)
        )

    async def add_message_back(self, user_id, message):
        self.memory[user_id].insert(
            0, MemoryMessage(id=0, message=message, timestamp=dt.datetime.now(dt.UTC))
        )

        return MemoryMessage(id=0, message=message, timestamp=dt.datetime.now(dt.UTC))

    async def get_dialog(self, user_id) -> list[MemoryMessage]:
        msgs = []

        for i, msg in enumerate(self.memory[user_id]):
            msg = msg.model_copy()
            msg.id = i

            msgs.append(msg)

        return msgs

    async def delete_messages(self, user_id, ids: list):
        msgs = []

        for i, msg in enumerate(self.memory[user_id]):
            if i not in ids:
                msgs.append(msg)

        self.memory[user_id] = msgs

    async def last_message(self, user_id) -> MemoryMessage:
        if self.memory[user_id]:
            msg = self.memory[user_id][-1].model_copy()
            msg.id = len(self.memory[user_id]) - 1

            return msg

        return None

    async def count_messages(self, user_id):
        return len(self.memory[user_id])
