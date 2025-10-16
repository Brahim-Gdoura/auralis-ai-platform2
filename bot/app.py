# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_pipeline import get_or_create_rag_system, get_or_create_chain
from db import conversations

app = Flask(__name__)
CORS(app)  # 🔹 Active CORS pour toutes les routes


@app.before_request
def initialize():
    """Initialize RAG pipeline on first request"""
    try:
        # Test if RAG system is accessible
        get_or_create_chain("test_init")
        print("✅ RAG pipeline ready!")
    except Exception as e:
        print(f"⚠️ Warning during initialization: {e}")


@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat requests with session management and history"""
    try:
        data = request.json
        query = data.get("query", "")
        session_id = data.get("session_id")
        
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400
        
        if not query:
            return jsonify({"error": "query is required"}), 400
        
        # Get or create RAG system for this session
        rag_system = get_or_create_rag_system(session_id)
        response = rag_system.query(query)
        
        answer = response.get("result", "")
        
        # Extract and format source documents
        sources = []
        for doc in response.get("source_documents", []):
            sources.append({
                'content': doc.page_content[:200],
                'metadata': doc.metadata
            })
        
        # Save to MongoDB for history
        conversations.insert_one({
            "session_id": session_id,
            "user_message": query,
            "bot_response": answer,
            "type": response.get("type", "unknown"),
            "sources_count": len(sources)
        })
        
        return jsonify({
            "session_id": session_id,
            "answer": answer,
            "sources": sources,
            "type": response.get("type", "unknown")
        })
    
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500


def get_or_create_rag_system(session_id: str):
    """Get or create an enhanced RAG system for a session"""
    from rag_pipeline import get_or_create_rag_system as _get_rag_system
    return _get_rag_system(session_id)


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
            return jsonify({"history": []})
        
        # Format history as conversation pairs
        formatted = []
        for msg in history_docs:
            # Convert ObjectId to string for JSON serialization
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


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    try:
        # Test if system is responsive
        rag_system = get_or_create_rag_system("health_check")
        return jsonify({
            "status": "ok",
            "message": "RAG system is initialized and ready",
            "rag_initialized": True
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
    """Alternative API endpoint (compatible with partner's implementation)"""
    try:
        data = request.get_json()
        message = data.get("message", "")
        session_id = data.get("session_id", f"api_session_{request.remote_addr}")
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        # Get or create RAG system for this session
        rag_system = get_or_create_rag_system(session_id)
        response = rag_system.query(message)
        
        sources = []
        for doc in response.get("source_documents", []):
            sources.append({
                "content": doc.page_content[:200],
                "metadata": doc.metadata
            })
        
        # Save to MongoDB
        conversations.insert_one({
            "session_id": session_id,
            "user_message": message,
            "bot_response": response.get("result", ""),
            "type": response.get("type", "unknown"),
            "endpoint": "api_chat"
        })
        
        return jsonify({
            "response": response.get("result", ""),
            "sources": sources,
            "type": response.get("type", "unknown"),
            "session_id": session_id
        })
    
    except Exception as e:
        print(f"❌ Error in API chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def api_health():
    """Alternative health endpoint (compatible with partner's implementation)"""
    try:
        rag_system = get_or_create_rag_system("api_health_check")
        return jsonify({
            "status": "ok",
            "rag_initialized": True
        })
    except Exception as e:
        print(f"⚠️ API health check failed: {e}")
        return jsonify({
            "status": "error",
            "rag_initialized": False
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)