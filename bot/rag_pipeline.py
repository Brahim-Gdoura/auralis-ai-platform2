# rag_pipeline.py

import os
from typing import Optional, List
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.base import LLM
from google import genai
from google.genai import types
from memory import MongoConversationMemory

# ✅ In-memory registry for active conversations
active_chats = {}  # { session_id: (ConversationalRetrievalChain, MongoConversationMemory) }


class GeminiGenAI(LLM):
    model_name: str = "gemini-2.0-flash-exp"
    temperature: float = 0.3
    client: object = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY")
        object.__setattr__(self, "client", genai.Client(api_key=api_key))

    @property
    def _llm_type(self):
        return "gemini-genai"

    def _call(self, prompt: str, stop: Optional[List[str]] = None):
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        config = types.GenerateContentConfig(temperature=self.temperature, response_modalities=["TEXT"])
        response = self.client.models.generate_content(
            model=self.model_name, contents=contents, config=config
        )
        return response.text


def init_rag():
    """Initialize RAG pipeline only once"""
    docs = []
    if not os.path.exists("data"):
        raise FileNotFoundError("❌ Folder 'data' missing")

    for file in os.listdir("data"):
        path = os.path.join("data", file)
        if file.endswith(".txt"):
            docs.extend(TextLoader(path, encoding="utf-8").load())
        elif file.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()

    llm = GeminiGenAI(model_name="gemini-2.0-flash-exp", temperature=0.3)

    print("✅ RAG pipeline initialized")
    return retriever, llm


# Load once globally
retriever, llm = init_rag()


def get_or_create_chain(session_id: str):
    """Return existing chain or create a new one for a given session"""
    if session_id in active_chats:
        return active_chats[session_id]

    memory = MongoConversationMemory(session_id=session_id)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        output_key="answer",
    )

    active_chats[session_id] = (chain, memory)
    print(f"🧠 New chain created for session {session_id}")
    return chain, memory
