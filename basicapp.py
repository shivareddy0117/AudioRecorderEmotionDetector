import os
import sys
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  # Ensure this import is added
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


from LSTM import lstm_model
from CNN import cnn_model
from RNN import rnn_model
import joblib

# Emotions in the RAVDESS dataset, different numbers represent different emotions
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}
base_dir = r"C:\Users\SAHITHYAMOGILI\Desktop\Projects\AudioRecorderEmotionDetector"
# Correct dataset path
dataset_path = os.path.join(base_dir, "audioData")
def load_data(dataset_path, emotions):
    X, y = [], []
    for file in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file)
        emotion = emotions[file.split("-")[2]]  # Extract emotion code from filename
        if emotion not in emotions.values():
            continue  # Skip files that do not correspond to the selected emotions
        data, sample_rate = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
        mfcc_processed = np.mean(mfcc.T, axis=0)
        X.append(mfcc_processed)
        y.append(emotion)
    return np.array(X), np.array(y)

# Load data
X, y = load_data(dataset_path, emotions)

# Encode labels into integers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)  # Convert labels to one-hot encoding

joblib.dump(encoder, os.path.join(base_dir, 'label_encoder.pkl' ))
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
joblib.dump(X_test, os.path.join(base_dir, 'X_test.pkl'))
joblib.dump(y_test, os.path.join(base_dir, 'y_test.pkl'))
# Model definition
model_type = sys.argv[1]
#input("Enter the model type you want to use (LSTM/CNN/RNN): ")
print(model_type)
# Update according to your data's shape
num_classes = y_train.shape[1]
# Training configurations
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
]

if model_type.lower() == 'cnn':
    # Reshape data for CNN input
    X_train_reshaped = X_train[..., np.newaxis]
    X_test_reshaped = X_test[..., np.newaxis]
    input_shape = (X_train_reshaped.shape[1], 1)
    model = cnn_model(input_shape, num_classes)  # Using the CNN model creation function
    history = model.fit(X_train_reshaped, y_train, validation_data=(X_test_reshaped, y_test), epochs=100, callbacks=callbacks)
    joblib.dump(history.history, os.path.join(base_dir, 'cnn_history.pkl'))
    model.save(os.path.join(base_dir, 'emotion_model_cnn.h5'))
elif model_type.lower() == 'lstm':
    # Reshape data for LSTM input
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])
    model = lstm_model(input_shape, num_classes)  # Using the LSTM model creation function
    history = model.fit(X_train_reshaped, y_train, validation_data=(X_test_reshaped, y_test), epochs=100, callbacks=callbacks)
    joblib.dump(history.history, os.path.join(base_dir, 'lstm_history.pkl'))
    model.save(os.path.join(base_dir, 'emotion_model_lstm.h5'))
elif model_type.lower() == 'rnn':
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])
    model = rnn_model(input_shape, num_classes)  # Using the RNN model creation function
    history = model.fit(X_train_reshaped, y_train, validation_data=(X_test_reshaped, y_test), epochs=30, batch_size=64, validation_split=0.2, callbacks=callbacks)
    joblib.dump(history.history, os.path.join(base_dir, 'rnn_history.pkl'))
    model.save(os.path.join(base_dir, 'emotion_model_rnn.h5'))

else:
    raise ValueError("Invalid model type entered. Please choose either 'LSTM' or 'CNN'.")

# Train the model



