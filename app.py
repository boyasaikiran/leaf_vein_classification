import streamlit as st
import joblib
import numpy as np
import cv2
import os
import pandas as pd
from src.preprocessing import preprocess_image
from src.feature_extraction import extract_features

# --- Paths ---
MODEL_PATH = "models/random_forest_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# --- Load Model and Scaler ---
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model()

# --- Streamlit UI ---
st.title("🌿 Leaf Vein Classification System")
st.markdown("Upload a leaf image to extract its skeleton, analyze its features, and classify it using a trained RandomForest model.")

uploaded_file = st.file_uploader("📤 Upload a Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save temporary image
    temp_path = "temp_uploaded_leaf.png"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # --- Step 1️⃣: Preprocess Image ---
    st.subheader("🖼️ Preprocessed & Skeletonized Image")
    processed, binary, skeleton = preprocess_image(temp_path)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), caption="Processed Image", use_container_width=True)
    with col2:
        st.image(binary, caption="Binary Image", use_container_width=True)
    with col3:
        st.image(skeleton, caption="Skeletonized Image", use_container_width=True)

    # --- Step 2️⃣: Extract Features ---
    st.subheader("📊 Extracted Feature Values")
    features = extract_features(processed, binary, skeleton)
    feature_array = np.array(features).reshape(1, -1)
    st.dataframe(pd.DataFrame(feature_array, columns=[f"Feature {i+1}" for i in range(feature_array.shape[1])]))

    # --- Step 3️⃣: Predict ---
    st.subheader("🧠 Model Prediction")
    scaled_features = scaler.transform(feature_array)
    prediction = model.predict(scaled_features)[0]

    # Class names (same order as training)
    # Update these manually if you know your dataset class names
    class_names = os.listdir("D:/Plants_2/train")
    class_names = [d for d in class_names if os.path.isdir(os.path.join("D:/Plants_2/train", d))]

    predicted_label = class_names[prediction] if len(class_names) > prediction else f"Class {prediction}"
    st.success(f"✅ **Predicted Leaf Type:** {predicted_label}")

    # --- Step 4️⃣: Show model details ---
    st.subheader("📈 Model Details")
    st.write("Model Type: RandomForestClassifier")
    st.write("Trained on extracted leaf vein features (skeleton-based)")
    st.write(f"Number of features used: {feature_array.shape[1]}")

    st.markdown("---")
    st.caption("All outputs — Skeleton image, extracted features, and prediction — generated using your trained RandomForest model.")
