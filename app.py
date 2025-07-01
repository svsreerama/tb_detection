import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('models/vgg16_model.h5')

def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = img / 255.0
    return np.expand_dims(img, axis=0)

st.title("Tuberculosis Detection from Chest X-rays")
uploaded_file = st.file_uploader("Upload Chest X-ray Image")

if uploaded_file is not None:
    image = preprocess_image(uploaded_file)
    prediction = model.predict(image)[0][0]
    label = "Tuberculosis" if prediction > 0.5 else "Normal"
    st.image(uploaded_file, caption=f"Prediction: {label} ({prediction:.2f})")
