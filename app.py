from flask import Flask, request, render_template, send_from_directory
import numpy as np
import tensorflow as tf
import os
import cv2 as cv
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Disable GPU usage for TensorFlow Lite
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Load the TFLite model using TensorFlow Lite Interpreter
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Class labels for predictions
CLASS_LABELS = {
    0: "Tomato___Bacterial_spot",
    1: "Tomato___Early_blight",
    2: "Tomato___Healthy",
    3: "Tomato___Late_blight",
    4: "Tomato___Leaf_Mold",
    5: "Tomato___Septoria_leaf_spot",
    6: "Tomato___Spider_mites (Two-spotted_spider_mite)",
    7: "Tomato___Target_Spot",
    8: "Tomato___Tomato_mosaic_virus",
    9: "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
}

# Function to check allowed file types
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess the image
def preprocess_image(image_path):
    image = cv.imread(image_path)
    if image is None:
        return None  # Handle cases where OpenCV fails to read the image

    image = cv.resize(image, (128, 128))  # Resize image to match model input size
    image = image.astype(np.float32) / 255.0  # Normalize pixel values
    image = np.reshape(image, (1, 128, 128, 3))  # Reshape to match input tensor shape
    return image

# Function to make predictions using TFLite model
def predict_tflite(image):
    # Ensure input shape matches model's expected shape
    input_data = np.array(image, dtype=np.float32)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Flask app routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"

        file = request.files["file"]
        if file.filename == "":
            return "No selected file"

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Preprocess the uploaded image
            image = preprocess_image(file_path)
            if image is None:
                return "Error: Invalid image file"

            # Make prediction
            prediction = predict_tflite(image)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_label = CLASS_LABELS.get(predicted_class, "Unknown")

            return render_template("index.html", filename=filename, prediction=predicted_label)

    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
