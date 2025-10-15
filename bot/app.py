from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_pipeline import build_rag

app = Flask(__name__)
CORS(app)

rag_system = None

@app.before_request
def initialize():
    global rag_system
    if rag_system is None:
        print("Initializing RAG pipeline...")
        rag_system = build_rag()
        if rag_system:
            print("RAG pipeline ready!")
        else:
            print("Failed to initialize RAG pipeline")

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        query = data.get('message', '')
        
        if not query:
            return jsonify({'error': 'Message is required'}), 400
        
        if rag_system is None:
            return jsonify({'error': 'RAG system not initialized'}), 500
        
        response = rag_system.query(query)
        
        sources = []
        for doc in response.get("source_documents", []):
            sources.append({
                'content': doc.page_content[:200],
                'metadata': doc.metadata
            })
        
        return jsonify({
            'response': response["result"],
            'sources': sources,
            'type': response.get("type", "unknown")  
        })
    
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'rag_initialized': rag_system is not None
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)