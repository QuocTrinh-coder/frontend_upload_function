from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

app = Flask(__name__)

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

@app.route('/', methods=['GET', 'POST'])
def home():
    # this is to open the website, when we click the link,, the request method is get so the function return the website with empty message on the website
    if request.method == 'GET':
        return render_template('index.html', msg='')

    
    image = request.files['file']
    img = Image.open(image)
    img = img.resize((224, 224))  # Resize image to match ResNet50 input size
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]  # Keep only the RGB channels, discard the alpha channel

    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)

    # Use ResNet50 to predict the image class
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Display the top three predicted labels
    top_predictions = [(label, round(score * 100, 2)) for (_, label, score) in decoded_predictions]
    print(top_predictions)
    return render_template('index.html', msg='Your image has been uploaded', predictions=top_predictions)

if __name__ == '__main__':
    app.run()
