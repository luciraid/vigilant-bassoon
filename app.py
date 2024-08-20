import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import openai

load_dotenv()

app = Flask(__name__)
CORS(app)

# Ensure OpenAI API key is loaded
openai.api_key = os.getenv('OPENAI_API_KEY')

if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please add it to your .env file.")

def get_character_response(character, message):
    prompts = {
        'ironman': f"As Tony Stark (Iron Man), respond to: '{message}'",
        'harvey': f"As Harvey Specter from Suits, respond to: '{message}'",
        'lucifer': f"As Lucifer Morningstar, respond to: '{message}'"
    }
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompts.get(character, f"Respond to: '{message}'"),
        max_tokens=150
    )
    
    return response.choices[0].text.strip()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data["message"]
    character = data.get("character", "default")
    
    if character == "default":
        response = get_character_response('default', user_input)
    else:
        response = get_character_response(character, user_input)
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
