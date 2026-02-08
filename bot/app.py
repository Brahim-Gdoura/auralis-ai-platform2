from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_pipeline import build_rag, get_or_create_chain
from db import conversations
from product_formatter import extract_products_from_response, should_show_products

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
    """Chat endpoint with session history support and smart product filtering"""
    try:
        data = request.get_json()
        query = data.get('message', '') or data.get('query', '')
        session_id = data.get('session_id', None)
        
        if not query:
            return jsonify({'error': 'Message is required'}), 400
        
        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400
        
        # Get chain with memory for this session
        chain, memory = get_or_create_chain(session_id)
        
        # Use invoke instead of __call__ to avoid deprecation warning
        response = chain.invoke({"question": query})
        
        answer = response["answer"]
        
        # Save to MongoDB
        memory.save_message(user_message=query, bot_response=answer)
        
        # Extract sources if available
        sources = []
        source_documents = response.get("source_documents", [])
        
        for doc in source_documents:
            sources.append({
                'content': doc.page_content[:200],
                'metadata': doc.metadata
            })
        
        # 🎯 CORRECTION CRITIQUE : Passer user_query pour le filtrage intelligent !
        products = []
        if should_show_products(query, answer):
            print(f"\n{'='*70}")
            print(f"🛍️  EXTRACTION DE PRODUITS PERTINENTS")
            print(f"{'='*70}")
            print(f"📝 Requête utilisateur: {query}")
            print(f"🤖 Réponse du bot: {answer[:100]}...")
            
            # ⚠️ IMPORTANT : Passer query comme user_query pour le filtrage !
            products = extract_products_from_response(
                answer, 
                source_documents, 
                user_query=query  # ← C'ÉTAIT MANQUANT !
            )
            
            print(f"\n{'='*70}")
            print(f"✅ {len(products)} produit(s) pertinent(s) seront affichés")
            print(f"{'='*70}\n")
        
        return jsonify({
            'session_id': session_id,
            'response': answer,
            'answer': answer,  # Keep both for compatibility
            'sources': sources,
            'products': products,  # Produits filtrés intelligemment
            'type': response.get("type", "unknown")
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['POST'])
@app.route('/history', methods=['POST'])
def get_history():
    """Get conversation history for a session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400
        
        # Fetch messages from MongoDB
        history = list(conversations.find({"session_id": session_id}).sort("_id", 1))
        
        formatted = []
        for msg in history:
            formatted.append({"role": "user", "content": msg["user_message"]})
            formatted.append({"role": "bot", "content": msg["bot_response"]})
        
        return jsonify({'history': formatted})
        
    except Exception as e:
        print(f"Error in history endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'rag_initialized': rag_system is not None
    })

@app.route('/chat', methods=['POST'])
def chat_legacy():
    """Legacy chat endpoint (redirects to new API)"""
    return chat()

if __name__ == '__main__':
    app.run(debug=False, port=5000)