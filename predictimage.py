import streamlit as st
import numpy as np
from PIL import Image
import joblib
import io
from streamlit_drawable_canvas import st_canvas

# --- Constants ---
MODEL_PATH = 'logistic_regression_model.joblib'
CANVAS_SIZE = 280 # Drawing canvas size (280x280 pixels)
IMAGE_PROCESS_SIZE = 28 # Model input size (28x28 pixels)

# --- Load the Trained Logistic Regression Model ---
@st.cache_resource # Cache the model loading for performance
def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at '{path}'.\nPlease ensure 'logistic_regression_model.joblib' is in the same directory as Preditimage.py")
        st.stop()

# --- Image Preprocessing Function ---
def preprocess_canvas_image(drawn_image_data):
    # Convert RGBA numpy array from st_canvas to PIL Image
    pil_image = Image.fromarray(drawn_image_data).convert('L') # Convert to grayscale

    # Resize to 28x28 using LANCZOS for high-quality downsampling
    processed_pil_image = pil_image.resize((IMAGE_PROCESS_SIZE, IMAGE_PROCESS_SIZE), Image.Resampling.LANCZOS)
    image_array = np.array(processed_pil_image)

    # Invert colors (MNIST is white digit on black background)
    # The canvas draws white on transparent/black. So, we just need to normalize.
    # No explicit inversion needed if the canvas background is black and drawing is white.
    
    # Normalize pixel values to be between 0 and 1 and flatten
    processed_image_flat = image_array.reshape(1, -1).astype('float32') / 255.0
    return processed_image_flat, image_array # Return image_array for display

# --- Streamlit App ---
st.set_page_config(layout="centered", page_title="Draw & Predict Digit")
st.title('Draw a Digit and Predict (Logistic Regression)')
st.write('Draw a single digit (0-9) on the canvas below, then click "Predict Digit"!')

# Load the model
logistic_model = load_model(MODEL_PATH)

# Create a canvas component
canvas_result = st_canvas(
    stroke_width=20,
    stroke_color="#FFFFFF", # White color for drawing
    background_color="#000000", # Black background
    height=CANVAS_SIZE,
    width=CANVAS_SIZE,
    key="canvas",
)

# Add a prediction button
if st.button('Predict Digit'):
    if canvas_result.image_data is not None:
        # Check if the canvas is empty (all black/transparent)
        if np.all(canvas_result.image_data[:, :, 3] == 0): # Check alpha channel for transparency
            st.warning("Please draw a digit on the canvas before predicting!")
        else:
            st.write("Processing drawn image...")

            # Preprocess the drawn image
            prediction_input, image_for_display = preprocess_canvas_image(canvas_result.image_data)

            # Display the processed 28x28 image
            st.image(image_for_display, caption='Processed for Prediction (28x28 grayscale)', width=100)

            # Perform prediction
            with st.spinner('Predicting...'):
                prediction = logistic_model.predict(prediction_input)[0]
                st.success('Prediction complete!')

            st.write(f"## Predicted Digit: **{prediction}**")
    else:
        st.warning("Please draw a digit on the canvas before predicting!")

# Optional: Display raw drawn image data (for debugging)
if canvas_result.image_data is not None:
    st.write("Raw Image Data (RGBA):")
    st.image(canvas_result.image_data)
