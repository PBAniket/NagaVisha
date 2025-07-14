# 🐍 Nagavisha: Snake Image Classifier

Nagavisha is a deep learning–powered web application that classifies snake images as **venomous** or **non-venomous**. Built using a fine-tuned **MobileNetV2** model, it provides a fast and lightweight way to predict snake danger levels from photos.

---

## 🚀 Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask (Python)
- **Model**: TensorFlow + MobileNetV2 (pre-trained, fine-tuned)
- **Image Preprocessing**: `ImageDataGenerator` from Keras

---

## 🧠 Model Architecture

- Base: MobileNetV2 (ImageNet weights, frozen)
- Added Layers:
  - Global Average Pooling
  - Dense (128 units, ReLU)
  - Dropout (0.5)
  - Output Layer: Dense (1 unit, Sigmoid)

Trained on a custom dataset of venomous and non-venomous snake images.

---

## 🖼️ Input & Output

- **Input**: Snake image uploaded via browser
- **Output**: Model prediction with label (e.g., `Venomous 🐍` or `Non-Venomous 🐍`) and confidence

---



## 💻 Local Setup

```bash
git clone https://github.com/PBAniket/Nagavisha.git
cd Nagavisha

# (Recommended) Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# or
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
