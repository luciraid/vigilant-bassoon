import random
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)
CORS(app)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Define character traits
traits = {
    'tony': ['genius', 'inventor', 'sarcastic', 'confident'],
    'lucifer': ['charming', 'witty', 'rebellious', 'hedonistic'],
    'harvey': ['assertive', 'successful', 'sharp', 'loyal']
}

# Define response templates
templates = [
    "As {character} would say, {response}",
    "Channeling my inner {character}, I'd say {response}",
    "Here's a {character}-inspired response: {response}",
    "In true {character} fashion: {response}"
]

# Define knowledge bases
astronomy_facts = [
    "Did you know that a day on Venus is longer than its year?",
    "The Great Red Spot on Jupiter is a storm that has been raging for over 400 years.",
    "There are more stars in the universe than grains of sand on all the beaches on Earth."
]

astrology_facts = [
    "In astrology, your sun sign is determined by the position of the sun at your time of birth.",
    "There are 12 zodiac signs, each associated with different personality traits.",
    "The study of astrology dates back to ancient civilizations like the Babylonians."
]

def generate_response(message):
    tokens = word_tokenize(message.lower())
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    character = random.choice(['tony', 'lucifer', 'harvey'])
    trait = random.choice(traits[character])
    
    if any(word in tokens for word in ['space', 'star', 'planet', 'galaxy']):
        response = random.choice(astronomy_facts)
    elif any(word in tokens for word in ['zodiac', 'horoscope', 'sign']):
        response = random.choice(astrology_facts)
    else:
        response = f"Well, that's an interesting point. As a {trait} individual, I find that quite intriguing."
    
    return random.choice(templates).format(character=character.capitalize(), response=response)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data["message"]
    
    response = generate_response(user_input)
    
    return jsonify({"response": response})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
