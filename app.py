import os
import numpy as np
from tensorflow.keras.models import model_from_json
from flask import Flask, request, render_template
import base64
from PIL import Image
from io import BytesIO

# -------------------- MODEL LOADING --------------------
def load_model(json_path, weights_path):
    with open(json_path, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(weights_path)
    print("✅ Model loaded successfully!")
    return model

model = load_model('./model.json', './model_weights.h5')

# -------------------- PREPROCESSING --------------------
def preprocess_image(image, target_size=(28, 28)):
    image = image.resize(target_size)
    image_array = np.array(image, dtype=np.float32)
    image_array = np.where(image_array <= 128, 1, 0)  # Invert to match training format
    image_array = np.expand_dims(image_array, axis=(0, -1))  # (1, 28, 28, 1)
    return image_array

def predict_image(model, image):
    image_array = preprocess_image(image)
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return chr(96 + predicted_class)  # 1 → 'a', 2 → 'b', etc.

# -------------------- FLASK APP --------------------
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        try:
            # Lấy ảnh từ form
            data = request.form['image']
            _, encoded = data.split(';base64,')
            image_bytes = base64.b64decode(encoded)
            image = Image.open(BytesIO(image_bytes)).convert('L')  # Grayscale
            result = predict_image(model, image)
        except Exception as e:
            result = f"Lỗi: {str(e)}"
    return render_template('index.html', predict=result)

# -------------------- SERVER START --------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Railway yêu cầu chạy đúng cổng
    app.run(host='0.0.0.0', port=port)
