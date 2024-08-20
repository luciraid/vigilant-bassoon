import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from inference_pipeline import InferencePipeline

app = Flask(__name__)
CORS(app)

# Initialize the inference pipeline
pipeline = InferencePipeline("./fine_tuned_gpt2")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data["message"]
    response = pipeline(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
