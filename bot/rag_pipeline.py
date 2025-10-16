# rag_pipeline.py

import os
import csv
from typing import Optional, List
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
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


def load_documents():
    """Load documents from data folder (txt, pdf, csv)"""
    docs = []
    
    if not os.path.exists("data"):
        raise FileNotFoundError("❌ Folder 'data' missing")

    for file in os.listdir("data"):
        path = os.path.join("data", file)
        try:
            if file.endswith(".txt"):
                docs.extend(TextLoader(path, encoding="utf-8").load())
                print(f"✅ Loaded: {file}")
            elif file.endswith(".pdf"):
                docs.extend(PyPDFLoader(path).load())
                print(f"✅ Loaded: {file}")
            elif file.endswith(".csv"):
                with open(path, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    csv_content = []
                    for row in reader:
                        product_text = f"Product: {row.get('name', 'N/A')}\n"
                        product_text += f"Category: {row.get('category', 'N/A')}\n"
                        product_text += f"Price: ${row.get('price', 'N/A')}\n"
                        product_text += f"Stock: {row.get('stock', 'N/A')} units available\n"
                        product_text += f"Description: {row.get('description', 'N/A')}\n"
                        product_text += f"ID: {row.get('product_id', 'N/A')}\n"
                        csv_content.append(product_text)
                    
                    full_content = "\n---\n".join(csv_content)
                    docs.append(Document(page_content=full_content, metadata={"source": file}))
                    print(f"✅ Loaded CSV: {file} with {len(csv_content)} products")
        except Exception as e:
            print(f"⚠️ Error loading {file}: {e}")

    return docs


def init_rag():
    """Initialize RAG pipeline only once"""
    docs = load_documents()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="\n---\n")
    documents = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = GeminiGenAI(model_name="gemini-2.0-flash-exp", temperature=0.3)

    print("✅ RAG pipeline initialized")
    return retriever, llm, vectorstore


# Load once globally
retriever, llm, vectorstore = init_rag()


def _create_qa_chain():
    """Create a RetrievalQA chain with custom prompt"""
    template = """You are Auralis, a friendly shopping assistant. Be brief, natural, and helpful.

RULES:
1. Keep responses SHORT (2-3 sentences max)
2. Ask ONLY 1 question if needed
3. If context has price/stock info, include it
4. NO bullet points - write naturally
5. Use exact product details from context (name, price, stock)
6. If info is in context, provide it directly - don't say you don't know

Context: {context}

Question: {question}

Response (SHORT and helpful):"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain


def get_or_create_chain(session_id: str):
    """Return existing chain or create a new one for a given session"""
    if session_id in active_chats:
        return active_chats[session_id]

    memory = MongoConversationMemory(session_id=session_id)
    qa_chain = _create_qa_chain()

    active_chats[session_id] = (qa_chain, memory)
    print(f"🧠 New chain created for session {session_id}")
    return qa_chain, memory


class RAGSystem:
    """Enhanced RAG System with query augmentation and chitchat handling"""
    
    def __init__(self, chain, memory, llm, vectorstore):
        self.chain = chain
        self.memory = memory
        self.llm = llm
        self.vectorstore = vectorstore

    def query(self, user_query: str):
        """Process user query with chitchat detection and query augmentation"""
        
        # Handle chitchat first
        chitchat_response = self._handle_chitchat(user_query)
        if chitchat_response:
            return {
                "result": chitchat_response,
                "source_documents": [],
                "type": "chitchat"
            }
        
        # Augment query for better retrieval
        print(f"\n📝 Original query: {user_query}")
        augmented_queries = self._augment_query(user_query)
        print(f"🔍 Augmented queries: {augmented_queries}")
        
        # Retrieve documents with deduplication
        all_docs = []
        seen_content = set()
        
        for aug_query in augmented_queries:
            docs = self.vectorstore.similarity_search(aug_query, k=3)
            for doc in docs:
                if doc.page_content not in seen_content:
                    all_docs.append(doc)
                    seen_content.add(doc.page_content)
        
        print(f"📚 Retrieved {len(all_docs)} unique documents")
        
        if all_docs:
            result = self.chain({"query": user_query})
            result["type"] = "rag"
            return result
        else:
            return {
                "result": "I don't have specific information about that in my knowledge base. Could you rephrase your question or ask something else?",
                "source_documents": [],
                "type": "no_results"
            }

    def _augment_query(self, query: str) -> List[str]:
        """Generate alternative query phrasings for better retrieval"""
        query_lower = query.lower()
        
        gift_keywords = ['gift', 'present', 'recommend', 'suggest', 'what should i', 'help me find', 'looking for', 'need something']
        is_gift_query = any(keyword in query_lower for keyword in gift_keywords)
        
        if is_gift_query:
            augmentation_prompt = f"""Given this shopping query: "{query}"

Generate 3 alternative search queries that would help find relevant products or gift options.
Focus on product categories, occasions, and recipient types.

For example:
- If query is "I want to buy a gift", alternatives could be: "gift ideas products", "popular gift items", "best selling products"
- If query is "what do you recommend", alternatives could be: "recommended products", "top products", "featured items"
- If query is "gift for mom", alternatives could be: "mother gifts", "products for women", "mom birthday present"

Return only the 3 alternatives, one per line, without numbering or explanation."""
        else:
            augmentation_prompt = f"""Given this user query: "{query}"

Generate 3 alternative phrasings or related questions that have the same intent.
Include synonyms and different ways someone might ask the same thing.

Return only the 3 alternatives, one per line, without numbering or explanation."""

        try:
            augmented = self.llm._call(augmentation_prompt)
            alternatives = [line.strip() for line in augmented.split('\n') if line.strip()]
            return [query] + alternatives[:3]
        except Exception as e:
            print(f"⚠️ Query augmentation failed: {e}")
            return [query]

    def _handle_chitchat(self, query: str) -> Optional[str]:
        """Detect and respond to common chitchat patterns"""
        query_lower = query.lower().strip()
        
        greetings = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening', 'howdy', 'what\'s up', 'whats up']
        if any(greet in query_lower for greet in greetings) and len(query_lower.split()) <= 5:
            return "Hello! 👋 I'm Auralis, your shopping assistant. I'm here to help you find the perfect products. Are you looking for something specific today, or would you like some recommendations?"
        
        manager_keywords = ['manager', 'supervisor', 'escalate', 'speak to someone', 'human agent', 'talk to a person', 'real person']
        if any(keyword in query_lower for keyword in manager_keywords):
            return "I understand you'd like to speak with a manager or supervisor. While I'm here to help answer your questions, if you need further assistance, please contact our support team directly. Is there anything specific I can help you with right now?"
        
        thanks = ['thank', 'thanks', 'appreciate', 'thx']
        if any(thank in query_lower for thank in thanks) and len(query_lower.split()) <= 5:
            return "You're very welcome! 😊 Is there anything else I can help you find today?"
        
        goodbye = ['bye', 'goodbye', 'see you', 'take care', 'gotta go', 'have a good']
        if any(bye in query_lower for bye in goodbye):
            return "Goodbye! Thanks for shopping with us. Feel free to come back anytime if you need help. Have a wonderful day! 🌟"
        
        how_are_you = ['how are you', 'how r u', 'how do you do']
        if any(phrase in query_lower for phrase in how_are_you):
            return "I'm doing great, thank you for asking! I'm excited to help you find what you're looking for. What can I assist you with today?"
        
        return None


def get_or_create_rag_system(session_id: str) -> RAGSystem:
    """Get or create an enhanced RAG system for a session"""
    chain, memory = get_or_create_chain(session_id)
    return RAGSystem(chain, memory, llm, vectorstore)