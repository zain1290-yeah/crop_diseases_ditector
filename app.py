import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
from PIL import Image
import gdown
import os

# -----------------------------
# 🔹 Download model if not present
# -----------------------------
MODEL_PATH = "best_finetuned.h5"

if not os.path.exists(MODEL_PATH):
    # ✅ Fixed direct download link
    url = "https://drive.google.com/uc?id=1Q4hPt2ZqWaq-Z2PY4sGAA_fvcthgefnK"
    gdown.download(url, MODEL_PATH, quiet=False)

# -----------------------------
# 🔹 Load model and labels
# -----------------------------
@st.cache_resource
def load_model():
    model = keras.models.load_model(MODEL_PATH)
    class_names = json.load(open("labels.json"))
    return model, class_names

model, class_names = load_model()

# -----------------------------
# 🔹 Streamlit UI
# -----------------------------
st.title("🌾 Crop Disease Classifier")
st.write("Upload a leaf image to detect the type of disease or if it’s healthy.")

# Upload image
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_array = np.array(image.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = float(np.max(preds))
    label = class_names[class_idx]

    # Result
    st.success(f"**Prediction:** {label}")
    st.info(f"**Confidence:** {confidence:.2%}")
