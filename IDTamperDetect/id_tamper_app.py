from flask import Flask, render_template, request
import os
import cv2
from skimage.metrics import structural_similarity
import numpy as np

# Folders
UPLOAD_FOLDER = 'uploads'
REFERENCE_FOLDER = 'reference'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REFERENCE_FOLDER, exist_ok=True)

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Reference image path
reference_path = os.path.join(REFERENCE_FOLDER, 'ref_id.png')

def detect_tampering(uploaded_path, reference_path):
    # Read images
    img1 = cv2.imread(reference_path)
    img2 = cv2.imread(uploaded_path)

    # Check if images are read correctly
    if img1 is None:
        return "Reference image not found or cannot be read."
    if img2 is None:
        return "Uploaded image cannot be read."

    # Resize uploaded image to reference image size if needed
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute similarity
    score, diff = structural_similarity(gray1, gray2, full=True)
    score_percent = score * 100

    if score_percent > 95:  # Threshold for tampering
        return f"ID Card seems authentic. Similarity: {score_percent:.2f}%"
    else:
        return f"Possible tampering detected! Similarity: {score_percent:.2f}%"

# Home route
@app.route('/')
def home():
    return render_template('upload.html')

# Upload route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Detect tampering
        result = detect_tampering(filepath, reference_path)
        return f"File '{file.filename}' uploaded successfully!<br>{result}"

if __name__ == '__main__':
    app.run(debug=True)
