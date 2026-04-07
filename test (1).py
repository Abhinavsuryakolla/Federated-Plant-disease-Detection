import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_model(model_path):
    test_dir = "test_dataset"

    if not os.path.exists(test_dir):
        raise FileNotFoundError("Test dataset not found.")

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found.")

    print("\nEvaluating Model:", model_path)

    data_gen = ImageDataGenerator(rescale=1./255)

    test_generator = data_gen.flow_from_directory(
        test_dir,
        target_size=(128,128),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    model = load_model(model_path,compile=False)

    predictions = model.predict(test_generator, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average='weighted',
        zero_division=0
    )

    print("\nEvaluation Metrics")
    print("------------------")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_model.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    evaluate_model(model_path)