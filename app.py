import streamlit as st
import numpy as np
from PIL import Image
import joblib
import os
import io

# --- Constants (Paths to be adjusted for Streamlit Cloud deployment) ---
# In Streamlit Cloud, models and data often need to be in the same repo as app.py
# For local Colab testing, we'll keep the full path.
MODEL_PATH = 'logistic_regression_model.joblib'

# --- Load the Trained Logistic Regression Model ---
@st.cache_resource # Cache the model loading for performance
def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at '{path}'.\nPlease ensure 'logistic_regression_model.joblib' is in the correct directory.")
        st.stop()

# --- Image Preprocessing Function ---
def preprocess_image(uploaded_image_bytes):
    # Open the image using PIL from bytes and convert to grayscale
    pil_image = Image.open(io.BytesIO(uploaded_image_bytes)).convert('L')

    # Resize to 28x28
    processed_pil_image = pil_image.resize((28, 28), Image.Resampling.LANCZOS)
    image_array = np.array(processed_pil_image)

    # Invert colors if necessary (MNIST is white digit on black background)
    # This is a heuristic and might need adjustment depending on the actual image.
    if np.mean(image_array) > 127: # Arbitrary threshold for white background
        image_array = 255 - image_array # Invert colors

    # Normalize pixel values to be between 0 and 1
    processed_image_flat = image_array.reshape(1, -1).astype('float32') / 255.0
    return processed_image_flat, image_array # Return image_array for display

# --- Streamlit App --- 
st.title('Handwritten Digit Recognition using Logistic Regression')
st.write('Upload a handwritten digit image (0-9) and the model will predict it.')

# Load the model
logistic_model = load_model(MODEL_PATH)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the bytes from the uploaded file
    image_bytes = uploaded_file.getvalue()

     # Convert image bytes to a PIL Image object for display
    uploaded_pil_image = Image.open(io.BytesIO(image_bytes))

    # Display the uploaded image
    st.image(uploaded_pil_image, caption='Uploaded Image', use_column_width=True)
    st.write("Processing image...")

    # Preprocess the image
    prediction_input, image_for_display = preprocess_image(image_bytes)

    # Display the processed image
    st.image(image_for_display, caption='Processed for Prediction (28x28 grayscale)', width=150)

    # Perform prediction
    with st.spinner('Predicting...'):
        prediction = logistic_model.predict(prediction_input)[0]
        st.success('Prediction complete!')

    st.write(f"## Predicted Digit: **{prediction}**")

else:
    st.write("Please upload an image to start the prediction.")
