# 🎙️ Speech Emotion Recognition AI (SER)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Accuracy](https://img.shields.io/badge/Accuracy-79.17%25-brightgreen)

## 📌 Project Overview
This project is an end-to-end **Speech Emotion Recognition** system that identifies human emotions from audio clips. Using Deep Learning and Signal Processing, the model can "listen" to a voice and classify it into one of 8 emotional categories.

## 🚀 Final Performance
- **Test Accuracy:** **80.87%**
- **Model Type:** 1D Convolutional Neural Network (CNN)
- **Features Extracted:** 180 total (MFCC, Chroma, and Mel-Spectrogram)

## 🛠️ The Technology Stack
- **Audio Processing:** `Librosa` (Feature extraction, Augmentation)
- **Deep Learning:** `TensorFlow` & `Keras` (1D-CNN Architecture)
- **Data Manipulation:** `NumPy`, `Pandas`
- **User Interface:** `Streamlit` (Web-based local deployment)

## 📊 Dataset: RAVDESS
The model was trained on the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**.
- **24 Professional Actors** (12 Male, 12 Female)
- **8 Emotions:** Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised.
- **Data Augmentation:** To improve accuracy, the training data was tripled by adding **White Noise** and **Pitch Shifting** to original samples.

## 🧠 Model Architecture
To prevent overfitting (memorization of actors), the model uses:
- **1D Convolutional Layers** to detect temporal patterns in sound.
- **Batch Normalization** for training stability.
- **Dropout (0.5)** and **L2 Regularization** to ensure the model generalizes to new voices.

## 📁 Project Structure
- `app.py`: The Streamlit web application.
- `emotion_weights.weights.h5`: The trained neural network weights.
- `scaler.pkl`: The mathematical scaler for normalizing input audio.
- `label_encoder.pkl`: Mapping of numerical predictions back to emotion names.
- `requirements.txt`: List of necessary Python libraries.
- `notebook/`: Contains the original training process and data exploration.

## ⚙️ How to Run Locally
1. **Clone the Repo:**
   ```bash
   git clone https://github.com/tedd12t/CodeAlpha-Emotion-Recognition-From-Speech.git
   cd CodeAlpha-Emotion-Recognition-From-Speech

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Launch the App:**
   ```bash
   streamlit run app.py
