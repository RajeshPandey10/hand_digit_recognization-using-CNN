# Handwritten Digit Recognition Web App

## 🚀 Live Demo

Try it now: [Handwritten Digit Recognition App](https://handdigitrecognization-using-cnn-byrajeshpandey.streamlit.app/)

## Overview

This project is a deep learning-powered web app that recognizes handwritten digits in real time using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## How It Works

### ✨ Model Training

- Built with TensorFlow and Keras using the MNIST dataset of handwritten digits.
- The model architecture includes:
  - **Conv2D:** Extracts spatial features from images.
  - **BatchNormalization:** Stabilizes and speeds up training.
  - **MaxPooling2D:** Reduces spatial dimensions, focusing on important features.
  - **Dense:** Fully connected layers for classification.
  - **Dropout:** Prevents overfitting by randomly dropping neurons during training.
- Achieved over 99% accuracy on test data!
- Trained model is saved in `.keras` format for easy deployment.

### 🖥️ Interactive GUI

- Developed with Streamlit and `streamlit-drawable-canvas`.
- Users can draw any digit (0-9) on the canvas and click "Recognize" for instant predictions.
- The app preprocesses your drawing, feeds it to the trained CNN, and displays the predicted digit with confidence.

## 🛠️ How to Build

### 1. Train the Model

- Run `train_digit_recognizer.py` to train and save the CNN model.

### 2. Run the App

- Launch `gui_digit_recognizer.py` with Streamlit:
  ```bash
  streamlit run gui_digit_recognizer.py
  ```
- All dependencies are managed via `requirements.txt` (Python packages) and `packages.txt` (system packages for deployment).

### 3. Live Demo

- Visit the [live demo link](https://handdigitrecognization-using-cnn-byrajeshpandey.streamlit.app/) to try it out instantly!

## Tech Stack

- Python
- TensorFlow
- Keras
- OpenCV
- Pillow
- Streamlit
- streamlit-drawable-canvas

## What’s Special?

- The canvas lets you draw digits just like on paper.
- The CNN model uses advanced layers to learn and recognize patterns in your handwriting.
- Everything runs live in your browser—no installation needed!

## Contact

Feel free to connect for collaboration or questions!

---
