import sys
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

base_dir = r"C:\Users\SAHITHYAMOGILI\Desktop\Projects\AudioRecorderEmotionDetector"

def load_data_and_model(model_type):
    # Load the saved history, model, and other data based on the model type
    history = joblib.load(os.path.join(base_dir, f'{model_type}_history.pkl'))
    model = load_model(os.path.join(base_dir, f'emotion_model_{model_type}.h5'))
    encoder = joblib.load(os.path.join(base_dir, 'label_encoder.pkl'))
    X_test = joblib.load(os.path.join(base_dir, 'X_test.pkl'))
    y_test = joblib.load(os.path.join(base_dir, 'y_test.pkl'))
    
    return history, model, encoder, X_test, y_test

def plot_history(history, model_type):
    plt.figure(figsize=(12, 5))
    plt.plot(history['accuracy'], label='Train accuracy')
    plt.plot(history['val_accuracy'], label='Validation accuracy')
    plt.title('Training History or Accuracy vs Epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plot_path = os.path.join(base_dir, f'{model_type}_accuracy_history.png')
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free up memory
    print(f"Accuracy history plot saved to {plot_path}")

def evaluate_model(model, X_test, y_test, encoder, model_type):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.2f}")

    predictions = model.predict(X_test)
    predicted_indices = np.argmax(predictions, axis=1)
    predicted_labels = encoder.inverse_transform(predicted_indices)
    actual_labels = encoder.inverse_transform(np.argmax(y_test, axis=1))

    for i in range(5):
        print(f"Sample {i + 1}: Actual Label - {actual_labels[i]}, Predicted Label - {predicted_labels[i]}")

    cm = confusion_matrix(actual_labels, predicted_labels, labels=encoder.classes_)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    confusion_matrix_path = os.path.join(base_dir, f'{model_type}_confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.close()
    print(f"Confusion matrix plot saved to {confusion_matrix_path}")

    report = classification_report(actual_labels, predicted_labels, target_names=encoder.classes_)
    print(report)
    report_path = os.path.join(base_dir, f'{model_type}_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate_models.py <model_type>")
        sys.exit(1)
    
    model_type = sys.argv[1].lower()  # Ensure the model type is lower case
    if model_type not in ['cnn', 'lstm', 'rnn']:
        print("Invalid model type entered. Please choose 'cnn', 'lstm', or 'rnn'.")
        sys.exit(1)

    history, model, encoder, X_test, y_test = load_data_and_model(model_type)
    plot_history(history, model_type)
    evaluate_model(model, X_test, y_test, encoder, model_type)
