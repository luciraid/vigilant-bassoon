import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load the model
generator = pipeline('text-generation', model='gpt2')  # You can try other models too

def get_chatbot_response(message):
    prompt = f"Respond with the intelligence of Tony Stark, the charm of Lucifer Morningstar, and the assertiveness of Harvey Specter to the following message: '{message}'. Include knowledge of conversations, astrology, and astronomy when relevant."
    
    try:
        response = generator(prompt, max_length=150, num_return_sequences=1)
        return response[0]['generated_text'].strip()
    except Exception as e:
        print(f"Error in text generation: {str(e)}")
        return f"I apologize, but I'm having trouble generating a response right now. Error: {str(e)}"

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
