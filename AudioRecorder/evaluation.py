# Plot training history
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense
from model import create_model
from sklearn.metrics import confusion_matrix, classification_report
from basicapp import history, encoder
plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Training History')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# Evaluate the model on test data
test_loss, test_accuracy = create_model.evaluate(X_test_reshaped, y_test)
print(f"Test accuracy: {test_accuracy:.2f}")

# Make predictions
predictions = create_model.predict(X_test_reshaped)
predicted_indices = np.argmax(predictions, axis=1)

# Decode labels
  # Assuming encoder is already fitted in your data preprocessing step
predicted_labels = encoder.inverse_transform(predicted_indices)
actual_labels = encoder.inverse_transform(np.argmax(y_test, axis=1))

# Display some predictions
for i in range(5):
    print(f"Sample {i + 1}: Actual Label - {actual_labels[i]}, Predicted Label - {predicted_labels[i]}")

# Confusion matrix and classification report
cm = confusion_matrix(actual_labels, predicted_labels, labels=encoder.classes_)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
print(classification_report(actual_labels, predicted_labels, target_names=encoder.classes_))
