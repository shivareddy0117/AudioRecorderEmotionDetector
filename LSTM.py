from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.5),
        LSTM(128, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Example input shape and number of classes
# For instance, if your input data is 100 time steps with 40 features:
# input_shape = (100, 40)
# num_classes = 8 # for the number of emotions you have in your dataset

# To create the model, you would call the function with the specific input shape and number of classes:
# lstm_model = create_lstm_model(input_shape, num_classes)
