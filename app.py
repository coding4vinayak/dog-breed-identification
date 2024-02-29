from PIL import Image
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import io
import numpy as np

app = Flask(__name__)

# Load the pre-trained InceptionV3 model
model = InceptionV3(weights='imagenet')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    # Get the image file from the request
    file = request.files['image']
    
    # Read image file
    img = Image.open(io.BytesIO(file.read()))
    img = img.resize((299, 299))  # Resize image to match InceptionV3 input size

    # Convert image to array and preprocess
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Prepare predictions to display
    results = [{'label': label, 'confidence': f"{confidence * 100:.2f}%"} for _, label, confidence in decoded_predictions]

    return render_template('index.html', image_file=file, predictions=results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
