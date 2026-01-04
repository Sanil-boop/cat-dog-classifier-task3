from flask import Flask, render_template, request
from joblib import load
import numpy as np
import cv2
from skimage.feature import hog

# ---- MUST MATCH TRAINING SETTINGS ----
IMG_SIZE = 64

model = load("svm_cat_dog_model.joblib")

app = Flask(__name__)

def extract_hog(image):
    return hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return render_template("index.html", error="No file uploaded")

    file = request.files["image"]

    if file.filename == "":
        return render_template("index.html", error="Please select an image")

    filename = file.filename

    # safely read uploaded file
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # invalid or unsupported image
    if img is None:
        return render_template(
            "index.html",
            error="Invalid image file. Please upload JPG or PNG.",
            filename=filename
        )

    # ---- SAME PIPELINE AS TRAINING ----
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = extract_hog(img)
    img = np.array(features).reshape(1, -1)

    # ---- PREDICT ----
    pred = model.predict(img)[0]
    label = "Cat 😺" if pred == 0 else "Dog 🐶"

    return render_template(
        "index.html",
        prediction=label,
        filename=filename
    )


if __name__ == "__main__":
    app.run(debug=True)
