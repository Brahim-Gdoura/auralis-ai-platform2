from langchain.memory import ConversationBufferMemory
from db import conversations

class MongoConversationMemory(ConversationBufferMemory):
    def __init__(self, session_id: str, **kwargs):
        super().__init__(memory_key="chat_history", return_messages=True, **kwargs)
        object.__setattr__(self, "session_id", session_id)

        # Load history
        for msg in conversations.find({"session_id": session_id}).sort("_id", 1):
            self.chat_memory.add_user_message(msg["user_message"])
            self.chat_memory.add_ai_message(msg["bot_response"])

    def save_message(self, user_message: str, bot_response: str):
        conversations.insert_one({
            "session_id": self.session_id,
            "user_message": user_message,
            "bot_response": bot_response
        })

    def clear(self):
        """Allow manual reset from UI"""
        conversations.delete_many({"session_id": self.session_id})
        self.chat_memory.clear()