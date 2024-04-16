import librosa
import numpy as np
import joblib
from tensorflow.keras.models import load_model

def load_single_data(file_path):
    # Load the audio file
    data, sample_rate = librosa.load(file_path, sr=None)
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
    mfcc_processed = np.mean(mfcc.T, axis=0)
    
    # Reshape to fit model input
    mfcc_processed = mfcc_processed.reshape(1, -1, 1)
    return mfcc_processed

def predict_emotion(file_path, model_path, encoder_path):
    # Load the model and the label encoder
    model = load_model(model_path)
    encoder = joblib.load(encoder_path)
    
    # Load and preprocess the audio file
    features = load_single_data(file_path)
    
    # Make prediction
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction, axis=1)
    predicted_label = encoder.inverse_transform(predicted_index)
    
    return predicted_label[0]

if __name__ == "__main__":
    import sys
    
    # Command line argument parsing
    if len(sys.argv) != 2:
        print("Usage: python predict_emotion.py <path_to_audio_file>")
    else:
        file_path = sys.argv[1]
        model_path = 'emotion_model.h5'
        encoder_path = 'label_encoder.pkl'
        
        # Predict emotion
        emotion = predict_emotion(file_path, model_path, encoder_path)
        print(f"The predicted emotion is: {emotion}")
