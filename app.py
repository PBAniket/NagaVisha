from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the trained model
model = load_model('snake_classifier_model.h5')

# Ensure uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def prepare_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/classify')
def classify():
    return render_template('classify.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = prepare_image(filepath)
    prediction = model.predict(img)[0][0]
    result = "Venomous 🐍" if prediction > 0.5 else "Non-venomous 🐍"

    os.remove(filepath)
    return result

if __name__ == '__main__':
    app.run(debug=True)
