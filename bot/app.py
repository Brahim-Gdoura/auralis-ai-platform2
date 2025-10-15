from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_pipeline import get_or_create_chain
from db import conversations

app = Flask(__name__)
CORS(app)  # 🔹 Active CORS pour toutes les routes

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query", "")
    session_id = data.get("session_id", None)

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    chain, memory = get_or_create_chain(session_id)

    response = chain({"question": query})
    answer = response["answer"]

    # Sauvegarder dans MongoDB
    memory.save_message(user_message=query, bot_response=answer)

    return jsonify({
        "session_id": session_id,
        "answer": answer
    })

@app.route("/history", methods=["POST"])
def get_history():
    data = request.json
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    # Fetch messages from MongoDB
    history = list(conversations.find({"session_id": session_id}).sort("_id", 1))
    formatted = []
    for msg in history:
        formatted.append({"role": "user", "content": msg["user_message"]})
        formatted.append({"role": "bot", "content": msg["bot_response"]})

    return jsonify({"history": formatted})

if __name__ == "__main__":
    app.run(port=5005, debug=True)
