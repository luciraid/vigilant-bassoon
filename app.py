from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

try:
    tokenizer = GPT2LMHeadModel.from_pretrained("gpt2")
    model = GPT2Tokenizer.from_pretrained("gpt2")
    app.logger.info("Models loaded successfully")
except Exception as e:
    app.logger.error(f"Error loading models: {str(e)}")

@app.route("/")
def index():
    try:
        return render_template("index.html")
    except Exception as e:
        app.logger.error(f"Error in index route: {str(e)}")
        return str(e), 500

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json["message"]
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
        output = model.generate(input_ids, max_length=100, num_return_sequences=1)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return jsonify({"response": response})
    except Exception as e:
        app.logger.error(f"Error in chat route: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
