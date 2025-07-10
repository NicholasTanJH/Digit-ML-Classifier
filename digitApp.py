from flask import Flask, request, render_template, jsonify
import numpy as np
from tensorflow import keras
import cv2
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
model = keras.models.load_model('digitModel.h5')

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    pixels = data.get("pixels", [])

    if len(pixels) != 784:
        return jsonify({"error": "Invalid pixel data"}), 400

    # Convert to 2D image
    image = np.array(pixels, dtype=np.uint8).reshape((28, 28))

    # Threshold to binary
    _, thresh = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)

    # Find bounding box of digit
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)

    # Crop and resize to 20x20
    digit = thresh[y:y+h, x:x+w]
    resized_digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)

    # Pad to 28x28 and center
    padded = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - 20) // 2
    y_offset = (28 - 20) // 2
    padded[y_offset:y_offset+20, x_offset:x_offset+20] = resized_digit

    # Normalize and reshape for model
    padded = padded.astype("float32") / 255.0
    padded = padded.reshape(1, 28, 28, 1)

    prediction = model.predict(padded)
    predicted_label = int(np.argmax(prediction))
    
    plt.figure(figsize=(5, 5))
    bars = plt.bar(range(10), prediction[0])
    bars[predicted_label].set_color('green')
    plt.title("Prediction Probabilities")
    plt.xlabel("Digit")
    plt.ylabel("Probability")
    plt.xticks(range(10))
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return jsonify({"prediction": predicted_label, "graph": image_base64})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
