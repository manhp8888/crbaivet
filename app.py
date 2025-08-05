import numpy as np
from tensorflow.keras.models import model_from_json
from flask import Flask, request, render_template, jsonify
import base64
from PIL import Image
from io import BytesIO

def load_model(json_path, weights_path):
    """
    Loads a pre-trained model from JSON (architecture) and weights files.

    Args:
    - json_path (str): Path to the file containing the model architecture in JSON format.
    - weights_path (str): Path to the file containing the model weights.

    Returns:
    - model: The fully loaded model, including both architecture and weights.
    """
    # Load model architecture
    with open(json_path, 'r') as json_file:
        model_json = json_file.read()

    model = model_from_json(model_json)

    # Load weights into the model
    model.load_weights(weights_path)
    print("Model loaded successfully!")

    return model

def preprocess_image(image, target_size=(28, 28)):
    """
    Preprocesses an image to make it ready for model prediction.

    Args:
    - image (PIL.Image): The input image as a PIL Image object.
    - target_size (tuple): The target size to which the image will be resized (default is (28, 28)).

    Returns:
    - image_array (numpy.ndarray): The preprocessed image as a NumPy array, including resizing, pixel normalization, and adding batch dimension.
    """
    # Resize the image to the target size
    image = image.resize(target_size)
    image_array = np.array(image, dtype=np.float32)
    # Normalize the pixel values to [0, 1]
    image_array = np.where(image_array <= 128, 1, 0)
    # Add a batch dimension (1, 28, 28, 1)
    image_array = np.expand_dims(image_array, axis=(0,-1))
    return image_array


def predict_image(model, image, target_size=(28, 28)):
    """
    Predicts the class of the input image using a pre-trained model.

    Args:
    - model: The pre-trained model.
    - image (PIL.Image): The input image as a PIL Image object.
    - target_size (tuple): The target size to which the image will be resized (default is (28, 28)).

    Returns:
    - predicted_class (int): The predicted class label with the highest probability.
    """
    # Preprocess the image
    image_preprocess = preprocess_image(image, target_size)
    # Predict
    predictions = model.predict(image_preprocess)
    predicted_class = np.argmax(predictions, axis=1)

    return predicted_class[0]

model = load_model('./model.json', './model_weights.h5')

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def home():
    print(request.method)
    if request.method == "GET":
       return render_template('index.html', predict = None)
    elif request.method == 'POST':
        try:
            data = request.get_data().decode()
            _, encoded = data.split(';base64,')
            image_encoded = base64.b64decode(encoded)
            image = Image.open(BytesIO(image_encoded)).convert('L')
            predict = predict_image(model, image)
            return jsonify(chr(96+ predict))
        except Exception as e:
            return jsonify(str(e)), 400


if __name__ == "__main__":
    app.run(debug = True)