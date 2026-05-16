import json
import os
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

from scraper import load_scraped_data, search_content, update_scraped_data, get_gemini_response

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    print(f"Gemini API Key loaded successfully (starts with: {GEMINI_API_KEY[:5]}...)")
else:
    print("Warning: GEMINI_API_KEY not found in environment variables.")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_FILE = os.path.join(BASE_DIR, "scraped_data.json")

app = Flask(__name__)

# Ensure data exists on startup
if not os.path.exists(DATA_FILE):
    print("Initializing scraped data...")
    update_scraped_data(DATA_FILE)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json() or {}
    question = payload.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please type a question about the college."})

    data = load_scraped_data(DATA_FILE)
    if not data:
        return jsonify({"answer": "I'm sorry, I don't have any data yet. Please try again later or refresh the data."})
    
    # Get relevant content first
    context_search = search_content(question, data, max_results=10)
    
    if GEMINI_API_KEY:
        answer = get_gemini_response(GEMINI_API_KEY, question, context_search)
    else:
        answer = context_search
        
    return jsonify({"answer": answer})

@app.route("/refresh", methods=["POST"])
def refresh():
    update_scraped_data(DATA_FILE)
    return jsonify({"message": "Scraped content refreshed."})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
