import sys
import librosa
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

def load_single_data(file_path):
    try:
        data, sample_rate = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
        mfcc_processed = np.mean(mfcc.T, axis=0)
        mfcc_processed = mfcc_processed.reshape(1, -1, 1)
        return mfcc_processed
    except Exception as e:
        print(f"Error loading audio data: {e}")
        sys.exit(1)

def predict_emotion(file_path, model_path, encoder_path, model_type):
    try:
        model = load_model(f"{model_path}_{model_type}.h5")
        encoder = joblib.load(encoder_path)
        features = load_single_data(file_path)
        prediction = model.predict(features)
        predicted_index = np.argmax(prediction, axis=1)
        predicted_label = encoder.inverse_transform(predicted_index)
        return predicted_label[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print(f"Received arguments: {sys.argv}")
    if len(sys.argv) != 3:
        print("Usage: python predict_emotion.py <path_to_audio_file> <model_type>")
        sys.exit(1)

    file_path = sys.argv[1]
    model_type = sys.argv[2]

    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        sys.exit(1)

    if model_type not in ['cnn', 'lstm', 'rnn']:
        print("Invalid model type! Use 'cnn' or 'lstm' or 'rnn'.")
        sys.exit(1)

    model_path = 'emotion_model'
    encoder_path = 'label_encoder.pkl'
    emotion = predict_emotion(file_path, model_path, encoder_path, model_type)
    print(f"The predicted emotion is: {emotion}")
