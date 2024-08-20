import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import openai

app = Flask(__name__)
CORS(app)

# Debugging: Print all environment variables (excluding the actual values for security)
print("Environment variables keys:", list(os.environ.keys()))

# Debugging: Check if the API key is set (don't print the actual key)
api_key_set = "OPENAI_API_KEY" in os.environ
print("OPENAI_API_KEY is set:", api_key_set)

openai.api_key = os.environ.get('OPENAI_API_KEY')

if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please add it to your environment variables.")

def get_chatbot_response(message):
    prompt = f"Respond with the intelligence of Tony Stark, the charm of Lucifer Morningstar, and the assertiveness of Harvey Specter to the following message: '{message}'. Include knowledge of conversations, astrology, and astronomy when relevant."
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a witty, intelligent, and charming AI assistant with knowledge of conversations, astrology, and astronomy. You combine the best traits of Tony Stark, Lucifer Morningstar, and Harvey Specter."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error in OpenAI API call: {str(e)}")
        return "I apologize, but I'm having trouble connecting to my knowledge base right now. Could you please try again later?"

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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
