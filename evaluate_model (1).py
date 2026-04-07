import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_all(round_number, is_final_round=False):
    test_dir = "test_dataset"
    if not os.path.exists(test_dir):
        raise FileNotFoundError("Test dataset not found.")
        
    print(f"\n--- Evaluating Round {round_number} Global Model ---")
    
    data_gen = ImageDataGenerator(rescale=1./255)
    test_generator = data_gen.flow_from_directory(
        test_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    def evaluate_single_model(model_path, model_name, save_confusion=False):
        if not os.path.exists(model_path):
            print(f"Model {model_path} not found.")
            return None

        model = load_model(model_path)
        
        # Predictions
        predictions = model.predict(test_generator, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes
        
        # Metrics
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        print(f"\n> Evaluating {model_name}...")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        
        # Confusion Matrix for final global model
        if save_confusion:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(16, 14))
            class_names = list(test_generator.class_indices.keys())
            sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Class')
            plt.xlabel('Predicted Class')
            plt.xticks(rotation=90)
            plt.tight_layout()
            
            output_file = f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png'
            plt.savefig(output_file)
            plt.close()
            print(f"Saved confusion matrix to {output_file}")
            
        return acc

    # Evaluate global model
    global_path = f"models/global_weights_round{round_number}.h5"
    global_acc = evaluate_single_model(global_path, f"Global Model Round {round_number}", save_confusion=is_final_round)

    # Save tracking metrics
    if global_acc is not None:
        with open('metrics_log.csv', 'a') as f:
            f.write(f"{round_number},{global_acc:.4f}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_model.py <round_number> [is_final]")
        sys.exit(1)
        
    r_num = int(sys.argv[1])
    is_final = False
    if len(sys.argv) > 2 and sys.argv[2].lower() == 'true':
        is_final = True
        
    evaluate_all(r_num, is_final_round=is_final)
