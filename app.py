import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import joblib
import os

#  PAGE CONFIG & STYLING 
st.set_page_config(page_title="Speech Emotion AI", page_icon="🎙️", layout="wide")
st.title("🎙️ Speech Emotion Recognition")
st.write("Upload a `.wav` file to detect the emotion in the voice.")

# FILE CHECKER
# This ensures all your saved assets exist before we try to load them
required_files = ['emotion_weights.weights.h5', 'scaler.pkl', 'label_encoder.pkl']
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    st.error(f"❌ Missing files in project folder: {', '.join(missing_files)}")
    st.stop() # Stop the app right here if files are missing

#  LOAD THE ASSETS
@st.cache_resource
def load_model_assets():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
    
    model = Sequential([
        Input(shape=(180, 1)),
        Conv1D(256, 8, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=5, strides=2, padding='same'),
        Conv1D(128, 8, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=5, strides=2, padding='same'),
        Dropout(0.5),
        Flatten(),
        Dense(8, activation='softmax')
    ])
    
    # Load the math (weights)
    model.load_weights('emotion_weights.weights.h5')
    
    # Load the processing tools
    scaler_obj = joblib.load('scaler.pkl')
    lb_obj = joblib.load('label_encoder.pkl')
    
    return model, scaler_obj, lb_obj

# Initialize the model and tools
model, scaler, lb = load_model_assets()

# --- 4. FEATURE EXTRACTOR ---
def extract_features(data, sr):
    # This matches the 180-feature extraction logic
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=data, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, mel])

# MAIN UI LOGIC
uploaded_file = st.file_uploader("Upload a .wav file", type=['wav'])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("🔍 Analyze Emotion"):
        with st.spinner("AI is analyzing the voice textures..."):
            # Load and Extract
            data, sr = librosa.load(uploaded_file, duration=3, offset=0.5)
            features = extract_features(data, sr)
            
            # Preprocess (Scale and Reshape)
            features = scaler.transform(features.reshape(1, -1))
            features = np.expand_dims(features, axis=2)
            
            # Predict
            prediction = model.predict(features, verbose=0)
            emotion = lb.classes_[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
            # Display Results
            st.divider()
            c1, c2 = st.columns(2)
            c1.success(f"### Detected Emotion: **{emotion.upper()}**")
            c2.info(f"### Confidence Score: **{confidence:.2f}%**")
            
            # Show Probability Chart
            probs = {lb.classes_[i]: float(prediction[0][i]) for i in range(len(lb.classes_))}
            st.bar_chart(probs)

# sidebar information 
st.sidebar.title("System Information")
st.sidebar.markdown(f"""
- **Model Accuracy:** 80.90%
- **Dataset:** RAVDESS
- **Features:** 180 (MFCC + Chroma + Mel)
""")