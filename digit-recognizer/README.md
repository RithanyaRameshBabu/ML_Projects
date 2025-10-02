# MNIST Digit Classifier

This project implements a **Convolutional Neural Network (CNN)** to classify handwritten digits (0–9) from the MNIST dataset.

## 📂 Project Structure
- `notebooks/` → Jupyter Notebook with full code (`mnist_cnn.ipynb`)
- `results/` → Saved plots and model results (`training_accuracy.png`)
- `.gitignore` → Specifies files/folders to ignore on GitHub
- `requirements.txt` → Project dependencies

## 📊 Dataset
- **MNIST**: 70,000 grayscale images of handwritten digits (28x28 pixels)
- 60,000 images for training, 10,000 for testing

## 🚀 How it Works
1. CNN learns patterns in the images (edges, curves, shapes)
2. Model trains on 60,000 images
3. Predicts digits on new images with high accuracy (~98–99%)

## 📈 Results
- Training and validation accuracy plotted in `results/training_accuracy.png`
- Visualized sample predictions on test images

## 🔧 How to Run
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
