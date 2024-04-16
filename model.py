from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense

def create_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, kernel_size=5, activation='relu', input_shape=input_shape),
        Conv1D(128, kernel_size=5, activation='relu'),
        MaxPooling1D(3),
        Conv1D(128, kernel_size=5, activation='relu'),
        GlobalAveragePooling1D(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
























'''The error message indicates that the l1_l2 function from the regularizers module in TensorFlow/Keras has not been defined or imported in your script. This function is used to apply L1 and L2 regularization to the model layers, which helps prevent overfitting by penalizing large weights.

To resolve this error, you need to import the l1_l2 function correctly at the beginning of your script. Here's how you can do it:

Correct Import Statement
Add this line at the beginning of your model.py file where you define your model:

from tensorflow.keras.regularizers import l1_l2
This import statement will make the l1_l2 regularizer available in your script, allowing you to use it to add regularization to your model layers.

Updated Model Definition with Correct Imports
Here is how your updated model definition in model.py might look with the correct import and usage of l1_l2:

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense
from tensorflow.keras.regularizers import l1_l2  # Import the regularizers

def create_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(256, 5, padding='same', activation='relu', input_shape=input_shape),
        Conv1D(128, 5, padding='same', activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        Dropout(0.1),
        MaxPooling1D(pool_size=8),
        Conv1D(128, 5, padding='same', activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        Conv1D(128, 5, padding='same', activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        Dropout(0.5),
        GlobalAveragePooling1D(),
        Dense(num_classes, activation='softmax', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model'''