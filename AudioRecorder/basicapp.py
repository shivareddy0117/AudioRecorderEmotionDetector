import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  # Ensure this import is added
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2


from model import create_model
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

# Correct dataset path
dataset_path = r"C:\Users\SAHITHYAMOGILI\Desktop\Projects\AudioRecorder\audioData"

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

joblib.dump(encoder, 'label_encoder.pkl')
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Model definition


# Reshape data to fit the model
X_train_reshaped = X_train[..., np.newaxis]
X_test_reshaped = X_test[..., np.newaxis]

# Create the model
input_shape = (X_train_reshaped.shape[1], 1)  # Update according to your data's shape
num_classes = y_train.shape[1]
model = create_model(input_shape, num_classes)

# Train the model

# Training configurations
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
]
history = model.fit(X_train_reshaped, y_train, validation_data=(X_test_reshaped, y_test), epochs=100, callbacks=callbacks)
model.save('emotion_model.h5')
