
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense
from tensorflow.keras.regularizers import l1_l2  # Make sure this import is at the top

def create_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(256, kernel_size=5, activation='relu', input_shape=input_shape),
        Conv1D(128, kernel_size=5, activation='relu'),
        MaxPooling1D(3),
        Conv1D(128, kernel_size=5, activation='relu'),
        GlobalAveragePooling1D(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


'''model = Sequential([
        Conv1D(256, 5, padding='same', activation='relu', input_shape=input_shape),
        Conv1D(128, 5, padding='same', activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        Dropout(0.1),
        MaxPooling1D(8),
        Conv1D(128, 5, padding='same', activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        Conv1D(128, 5, padding='same', activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        Dropout(0.5),
        GlobalAveragePooling1D(),
        Dense(num_classes, activation='softmax', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
              bias_regularizer=regularizers.l2(1e-4),
              activity_regularizer=regularizers.l2(1e-5))
    ])'''    

'''
model = Sequential([
        Conv1D(256, kernel_size=5, activation='relu', input_shape=input_shape),
        Conv1D(128, kernel_size=5, activation='relu'),
        MaxPooling1D(3),
        Conv1D(128, kernel_size=5, activation='relu'),
        GlobalAveragePooling1D(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])'''