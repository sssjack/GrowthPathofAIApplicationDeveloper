import json
import os
from typing import Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

def get_history(session_id):
    return FileChatMessageHistory(session_id, storage_path="./chat_history")

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path):
        self.session_id = session_id       # 会话id
        self.storage_path = storage_path   # 不同会话id的存储文件，所在的文件夹路径
        # 完整的文件路径
        self.file_path = os.path.join(self.storage_path, self.session_id)
        # 确保文件夹是存在的
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        # Sequence序列 类似list、tuple
        all_messages = list(self.messages)    # 已有的消息列表
        all_messages.extend(messages)         # 新的和已有的融合成一个list

        # 将数据同步写入到本地文件中
        # 类对象写入文件 -> 一堆二进制
        # 为了方便，可以将BaseMessage消息转为字典（借助json模块以json字符串写入文件）
        messages_dict = [message_to_dict(msg) for msg in all_messages]
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(messages_dict, f, ensure_ascii=False, indent=4)

    @property
    def messages(self):
        # 读取本地文件，获取消息列表
        if not os.path.exists(self.file_path):
            return []
        with open(self.file_path, "r", encoding="utf-8") as f:
            messages_dict = json.load(f)
        return messages_from_dict(messages_dict)

    def clear(self) -> None:
        # 清空会话历史记录
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
