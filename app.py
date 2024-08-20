from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import openai
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Use the pre-trained model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
print("Using pre-trained GPT-2 model")

openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def get_character_response(character, message):
    prompts = {
        'ironman': f"As Tony Stark (Iron Man), respond to: '{message}'",
        'harvey': f"As Harvey Specter from Suits, respond to: '{message}'",
        'lucifer': f"As Lucifer Morningstar, respond to: '{message}'"
    }
    
    response = openai.Completion.create(
      engine="text-davinci-002",
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
        response = generate_text(user_input)
    else:
        response = get_character_response(character, user_input)
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
