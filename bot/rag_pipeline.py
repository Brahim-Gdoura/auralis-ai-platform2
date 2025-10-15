import os
from typing import Optional, List
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from google import genai
from google.genai import types
from memory import MongoConversationMemory

active_chats = {}


class GeminiGenAI(LLM):
    model_name: str = "gemini-2.0-flash-exp"
    temperature: float = 0.7
    client: object = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY")
        object.__setattr__(self, "client", genai.Client(api_key=api_key))

    @property
    def _llm_type(self) -> str:
        return "gemini-genai"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        config = types.GenerateContentConfig(temperature=self.temperature, response_modalities=["TEXT"])
        response = self.client.models.generate_content(
            model=self.model_name, contents=contents, config=config
        )
        return response.text


class RAGSystem:
    def __init__(self, qa_chain, llm, vectorstore, retriever):
        self.qa_chain = qa_chain
        self.llm = llm
        self.vectorstore = vectorstore
        self.retriever = retriever

    def query(self, user_query: str):
        chitchat_response = self._handle_chitchat(user_query)
        if chitchat_response:
            return {
                "result": chitchat_response,
                "source_documents": [],
                "type": "chitchat"
            }

        print(f"\nOriginal query: {user_query}")
        augmented_queries = self._augment_query(user_query)
        print(f"Augmented queries: {augmented_queries}")

        all_docs = []
        seen_content = set()

        for aug_query in augmented_queries:
            docs = self.vectorstore.similarity_search(aug_query, k=3)
            for doc in docs:
                if doc.page_content not in seen_content:
                    all_docs.append(doc)
                    seen_content.add(doc.page_content)

        print(f"Retrieved {len(all_docs)} unique documents")

        if all_docs:
            result = self.qa_chain({"query": user_query})
            result["type"] = "rag"
            return result
        else:
            return {
                "result": "I don't have specific information about that in my knowledge base. Could you rephrase your question or ask something else?",
                "source_documents": [],
                "type": "no_results"
            }

    def _augment_query(self, query: str) -> List[str]:
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
            print(f"Query augmentation failed: {e}")
            return [query]

    def _handle_chitchat(self, query: str) -> Optional[str]:
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


def load_documents():
    docs = []

    if not os.path.exists("data"):
        print("Warning: 'data' folder not found")
        return docs

    for file in os.listdir("data"):
        path = os.path.join("data", file)
        try:
            if file.endswith(".txt"):
                loader = TextLoader(path, encoding='utf-8')
                docs.extend(loader.load())
                print(f"Loaded: {file}")
            elif file.endswith(".pdf"):
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
                print(f"Loaded: {file}")
            elif file.endswith(".csv"):
                import csv
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

                    from langchain.docstore.document import Document
                    full_content = "\n---\n".join(csv_content)
                    docs.append(Document(page_content=full_content, metadata={"source": file}))
                    print(f"Loaded CSV: {file} with {len(csv_content)} products")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    return docs


def build_rag():
    raw_docs = load_documents()

    if not raw_docs:
        print("No documents found")
        return None

    print(f"{len(raw_docs)} document(s) loaded")

    text_splitter = CharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separator="\n---\n"
    )
    documents = text_splitter.split_documents(raw_docs)
    print(f"{len(documents)} chunks created")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    print("Initializing Gemini LLM...")
    llm = GeminiGenAI(
        model_name="gemini-2.0-flash-exp",
        temperature=0.7
    )

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
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    print("RAG pipeline ready!")

    return RAGSystem(qa_chain, llm, vectorstore, retriever)


def get_or_create_chain(session_id: str):
    if session_id in active_chats:
        return active_chats[session_id]

    raw_docs = load_documents()
    if not raw_docs:
        raise ValueError("No documents found")

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()

    llm = GeminiGenAI(model_name="gemini-2.0-flash-exp", temperature=0.3)

    memory = MongoConversationMemory(session_id=session_id)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer",
    )

    active_chats[session_id] = (chain, memory)
    print(f"New chain created for session {session_id}")
    return chain, memory