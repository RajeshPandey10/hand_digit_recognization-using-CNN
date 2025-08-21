import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import cv2
import streamlit_drawable_canvas as draw

# Load the trained MNIST model
model = load_model('/Users/rajeshpandey/Desktop/AI_all/ML(hw)/Deep_learning/mnist_model.keras', compile=False)

# Preprocess image
def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1)
    image = image.astype("float32") / 255.0
    return image

# Predict digit
def predict_digit(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0]
    digit = np.argmax(prediction)
    confidence = int(max(prediction) * 100)
    return digit, confidence

# Streamlit UI
st.title("🖌 Handwritten Digit Recognition")
st.markdown("Draw a digit below and click **Recognize** to predict.")

# Session state flag to reset canvas
if "clear_canvas" not in st.session_state:
    st.session_state.clear_canvas = False



# Set canvas key depending on clear flag
canvas_key = "canvas_reset" if st.session_state.clear_canvas else "canvas"

# Drawing canvas
canvas_result = draw.st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=300,
    width=300,
    drawing_mode="freedraw",
    key=canvas_key
)

# Reset the flag after canvas is created
st.session_state.clear_canvas = True

# Recognize button
if st.button("Recognize"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        img = Image.fromarray(img)
        digit, confidence = predict_digit(img)
        st.success(f"Predicted Digit: {digit} with {confidence}% confidence")
    else:
        st.warning("Please draw a digit before clicking Recognize.")
