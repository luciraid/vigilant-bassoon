import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import openai

load_dotenv()

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv('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')

if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please add it to your .env file.")

def get_chatbot_response(message):
    prompt = f"Respond with the intelligence of Tony Stark, the charm of Lucifer Morningstar, and the assertiveness of Harvey Specter to the following message: '{message}'. Include knowledge of conversations, astrology, and astronomy when relevant."
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a witty, intelligent, and charming AI assistant with knowledge of conversations, astrology, and astronomy. You combine the best traits of Tony Stark, Lucifer Morningstar, and Harvey Specter."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    
    return response.choices[0].message['content'].strip()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data["message"]
    
    response = get_chatbot_response(user_input)
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
