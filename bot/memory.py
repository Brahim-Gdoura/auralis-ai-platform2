# memory.py

from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from db import conversations


class MongoConversationMemory(ConversationBufferMemory):
    """Custom memory that persists conversation history to MongoDB"""
    
    def __init__(self, session_id: str, **kwargs):
        super().__init__(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",  # Important: matches ConversationalRetrievalChain output
            **kwargs
        )
        object.__setattr__(self, "session_id", session_id)

        # Load existing conversation history from MongoDB
        self._load_history()

    def _load_history(self):
        """Load conversation history from MongoDB on initialization"""
        history_docs = conversations.find({"session_id": self.session_id}).sort("_id", 1)
        
        count = 0
        for msg in history_docs:
            self.chat_memory.add_message(HumanMessage(content=msg["user_message"]))
            self.chat_memory.add_message(AIMessage(content=msg["bot_response"]))
            count += 1
        
        if count > 0:
            print(f"📚 Loaded {count} previous messages for session {self.session_id}")

    def save_context(self, inputs: dict, outputs: dict):
        """Override to save to MongoDB when context is saved"""
        # Save to LangChain's in-memory buffer
        super().save_context(inputs, outputs)
        
        # Also persist to MongoDB
        user_message = inputs.get("input") or inputs.get("question", "")
        bot_response = outputs.get("output") or outputs.get("answer", "")
        
        if user_message and bot_response:
            conversations.insert_one({
                "session_id": self.session_id,
                "user_message": user_message,
                "bot_response": bot_response
            })
            print(f"💾 Saved conversation to MongoDB for session {self.session_id}")

    def clear(self):
        """Clear both in-memory and MongoDB history"""
        conversations.delete_many({"session_id": self.session_id})
        self.chat_memory.clear()
        print(f"🗑️ Cleared history for session {self.session_id}")