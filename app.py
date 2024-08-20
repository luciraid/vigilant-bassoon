import os
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from inference_pipeline import InferencePipeline

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Initialize the inference pipeline
pipeline = InferencePipeline("./fine_tuned_gpt2")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data["message"]
    try:
        response = pipeline(user_input)
        return jsonify({"response": response})
    except Exception as e:
        logging.error(f"Error during inference: {str(e)}")
        return jsonify({"error": "An error occurred during the inference process."}), 500

if __name__ == "__main__":
    # Load configuration from environment variables or a config file
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', False)

    logging.info(f"Starting the application in {'debug' if debug else 'production'} mode on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
