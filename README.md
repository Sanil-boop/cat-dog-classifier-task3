🐾 Cat vs Dog Image Classifier — SVM (ML Project)

🔗 Live Demo: https://cat-dog-classifier-task3.onrender.com

🖥️ Built With: Python • Flask • OpenCV • Scikit-Learn • HOG + SVM

📌 Project Overview

This project is a Machine Learning–based web application that classifies whether an uploaded image is a Cat 🐱 or a Dog 🐶 using a Support Vector Machine (SVM) trained on HOG image features.

The project demonstrates:

✔ Image preprocessing & feature extraction
✔ HOG descriptor for texture & edge detection
✔ SVM classification
✔ Flask web deployment
✔ Model optimization for deployment size
✔ Render hosting workflow

✨ Features

🖼 Upload any image (JPG / PNG)
⚙️ Automatic preprocessing using OpenCV
🔍 Extracts HOG features
🤖 Classifies using Linear SVM
⚡ Fast inference & lightweight model
🌐 Live hosted web app
🎨 Modern UI with gradient & glass-effect styling

🧠 Machine Learning Pipeline
| Step                   | Description                           |
| ---------------------- | ------------------------------------- |
| 🗂 Dataset             | Cats vs Dogs (Kaggle)                 |
| 🧾 Preprocessing       | Resize + Grayscale                    |
| 🧩 Feature Extraction  | HOG (Histogram of Oriented Gradients) |
| 🤖 Model               | SVM (Linear Kernel)                   |
| ✂ Balanced Subsampling | Equal cats & dogs                     |
| 🗜 Model Optimization  | Compressed Joblib Model (~19MB)       |

HOG features help SVM detect:

✔ edges
✔ shapes
✔ gradients
✔ textures

which makes it effective for cat-vs-dog classification.

Task 3/
│
├── app.py                  # Flask Web App
├── svm_cat_dog_model.joblib   # Optimized ML Model (19MB)
├── image_shape.npy         # Image size metadata
├── requirements.txt        # Dependencies
├── Procfile                # Render startup command
│
├── templates/
│   └── index.html          # Frontend UI
│
├── static/
│   └── style.css           # Styling
│
└── svm_cats_dogs.ipynb     # Model Training Notebook

🚀 Live Demo (Hosted on Render)

🟢 Click to try the app:

👉 https://cat-dog-classifier-task3.onrender.com

⚙️ Quick Setup (Run Locally)
git clone https://github.com/Sanil-boop/cat-dog-classifier-task3.git
cd cat-dog-classifier-task3
pip install -r requirements.txt
python app.py

Then open in browser:

👉 http://127.0.0.1:5000/

