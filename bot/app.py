# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_pipeline import get_or_create_rag_system, get_or_create_chain
from db import conversations

app = Flask(__name__)
CORS(app)


@app.before_request
def initialize():
    """Initialize RAG pipeline on first request"""
    try:
        get_or_create_chain("test_init")
        print("✅ RAG pipeline ready!")
    except Exception as e:
        print(f"⚠️ Warning during initialization: {e}")


@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat requests with full conversational memory"""
    try:
        data = request.json
        query = data.get("query", "")
        session_id = data.get("session_id")
        
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400
        
        if not query:
            return jsonify({"error": "query is required"}), 400
        
        print(f"\n💬 Session {session_id}: {query}")
        
        # Get RAG system with memory
        rag_system = get_or_create_rag_system(session_id)
        
        # Process query (memory is automatically used)
        response = rag_system.query(query)
        answer = response.get("result", "")
        
        # Extract and format source documents
        sources = []
        for doc in response.get("source_documents", []):
            sources.append({
                'content': doc.page_content[:200],
                'metadata': doc.metadata
            })
        
        return jsonify({
            "session_id": session_id,
            "answer": answer,
            "sources": sources,
            "type": response.get("type", "unknown"),
            "has_memory": True
        })
    
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=["POST"])
def get_history():
    """Retrieve conversation history for a session"""
    try:
        data = request.json
        session_id = data.get("session_id")
        
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400
        
        # Fetch messages from MongoDB
        history_docs = list(conversations.find({"session_id": session_id}).sort("_id", 1))
        
        if not history_docs:
            return jsonify({"history": [], "count": 0})
        
        # Format history as conversation pairs
        formatted = []
        for msg in history_docs:
            timestamp_str = str(msg.get("_id", ""))
            
            formatted.append({
                "role": "user",
                "content": msg["user_message"],
                "timestamp": timestamp_str
            })
            formatted.append({
                "role": "bot",
                "content": msg["bot_response"],
                "type": msg.get("type", "unknown"),
                "timestamp": timestamp_str
            })
        
        return jsonify({"history": formatted, "count": len(formatted)})
    
    except Exception as e:
        print(f"❌ Error in history endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/clear", methods=["POST"])
def clear_history():
    """Clear conversation history for a session"""
    try:
        data = request.json
        session_id = data.get("session_id")
        
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400
        
        # Get the RAG system and clear its memory
        rag_system = get_or_create_rag_system(session_id)
        rag_system.memory.clear()
        
        return jsonify({
            "success": True,
            "message": f"History cleared for session {session_id}"
        })
    
    except Exception as e:
        print(f"❌ Error clearing history: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    try:
        rag_system = get_or_create_rag_system("health_check")
        return jsonify({
            "status": "ok",
            "message": "RAG system with conversational memory is ready",
            "rag_initialized": True,
            "memory_enabled": True
        })
    except Exception as e:
        print(f"⚠️ Health check failed: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "rag_initialized": False
        }), 500


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Alternative API endpoint"""
    try:
        data = request.get_json()
        message = data.get("message", "")
        session_id = data.get("session_id", f"api_session_{request.remote_addr}")
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        rag_system = get_or_create_rag_system(session_id)
        response = rag_system.query(message)
        
        sources = []
        for doc in response.get("source_documents", []):
            sources.append({
                "content": doc.page_content[:200],
                "metadata": doc.metadata
            })
        
        return jsonify({
            "response": response.get("result", ""),
            "sources": sources,
            "type": response.get("type", "unknown"),
            "session_id": session_id,
            "has_memory": True
        })
    
    except Exception as e:
        print(f"❌ Error in API chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)