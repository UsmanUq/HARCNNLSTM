# app.py
import streamlit as st
import numpy as np
from tensorflow import keras
from keras.models import load_model
from keras.utils import to_categorical
import os

# Load the trained model
@st.cache_resource
def load_cnn_model():
    return load_model("har_weights.h5")

model = load_cnn_model()

# Activity labels
LABELS = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']

st.title("HAR Prediction using 2D CNN")
st.write("Upload a CSV file with 128 timesteps x 9 features for one sample.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    import pandas as pd
    data = pd.read_csv(uploaded_file, header=None)

    if data.shape != (128, 9):
        st.error("File must be of shape (128, 9) as one HAR sample.")
    else:
        # Reshape to model input
        sample = data.to_numpy().reshape(1, 128, 9)
        prediction = model.predict(sample)
        predicted_class = np.argmax(prediction, axis=1)[0]

        st.success(f"Predicted Activity: {LABELS[predicted_class]}")
