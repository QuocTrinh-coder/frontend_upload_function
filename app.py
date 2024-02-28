from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import tkinter
from tkinter import *
from chatapp import ChatApp as cA  # Importing ChatApp class from chatapp module
import os  # Importing the os module for path operations
from keras.models import load_model #  A function from Keras to load a pre-trained neural network model.
import nltk
import utils as u
import json

# Load pre-trained model and necessary data
pre_trained_model = load_model('chatbot_model.h5')
lemmatizer = nltk.stem.WordNetLemmatizer()
words = u.load_pickle(os.path.join('pickles', 'words.pkl'))
classes = u.load_pickle(os.path.join('pickles', 'classes.pkl'))
intents = json.loads(open('data.json').read())

# Create ChatApp instance with pre-trained model and data
ex = cA(pre_trained_model, lemmatizer, words, classes, intents)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('home.html')

@app.route("/upload", methods=["POST"])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    if request.method == 'POST':
        image = request.files['file']
        img = Image.open(image)
        img = np.array(img)

    print(img)
    print(img.shape)

    return jsonify({"message": "Your image has been uploaded"})

@app.route("/chat", methods=["POST"])
def chat():
    msg = request.form.get("msg")  # Get the message from the form
    response = get_chat_response(msg)  # Get the chatbot response
    return jsonify({"response": response})  # Return the response to the client

def get_chat_response(text):
    # Let's chat for 5 lines
    if text != '':
        res = ex.chatbot_response(text)  # Using the instance of ChatApp
        return res  # Return the response
    else:
        return "Message is empty"  # Return an error message if the message is empty

if __name__ == "__main__":
    app.run(port=5001)
