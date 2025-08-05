import os
import numpy as np
from tensorflow.keras.models import model_from_json
from flask import Flask, request, render_template
import base64
from PIL import Image
from io import BytesIO

# Load model
def load_model(json_path, weights_path):
    with open(json_path, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(weights_path)
    print("✅ Model loaded successfully!")
    return model

# Preprocessing image
def preprocess_image(image, target_size=(28, 28)):
    image = image.resize(target_size)
    image_array = np.array(image, dtype=np.float32)
    image_array = np.where(image_array <= 128, 1, 0)
    image_array = np.expand_dims(image_array, axis=(0, -1))
    return image_array

# Predict character
def predict_image(model, image, target_size=(28, 28)):
    image_preprocess = preprocess_image(image, target_size)
    predictions = model.predict(image_preprocess)
    predicted_class = np.argmax(predictions, axis=1)
    return chr(96 + predicted_class[0])  # Convert to letter (e.g., 1 -> 'a')

# Load model
model = load_model('./model.json', './model_weights.h5')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        try:
            data = request.get_data().decode()
            _, encoded = data.split(';base64,')
            image_encoded = base64.b64decode(encoded)
            image = Image.open(BytesIO(image_encoded)).convert('L')
            result = predict_image(model, image)
        except Exception as e:
            result = f"Lỗi: {str(e)}"
    return render_template('index.html', predict=result)

# Railway yêu cầu dùng cổng hệ thống cung cấp
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
