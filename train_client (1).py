import os
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from pathlib import Path

def train_client(client_name, round_number):
    print(f"\n--- Training {client_name} for Round {round_number} ---")
    
    global_model_path = f"models/global_weights_round{round_number - 1}.h5"
    if not os.path.exists(global_model_path):
        raise FileNotFoundError(f"Global model weights {global_model_path} not found.")
        
    # 1. Load global weights
    model = load_model(global_model_path)
    
    # Always recompile the loaded model to prevent optimizer state errors in TF>2.15
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Advanced: Transfer Learning Fine-Tuning
    # Unfreeze the base model layers after round 5 to fine-tune the feature extractor
    if round_number > 5:
        print("Round > 5: Unfreezing base MobileNetV2 layers for fine-tuning...")
        for layer in model.layers:
            layer.trainable = True
        
        # Recompile using a lower learning rate for fine-tuning
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    # 2 & 3. Load dataset & Apply preprocessing/augmentation
    # Advanced: Added brightness_range to simulate RandomContrast
    data_gen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=20,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2]
    )
    
    dataset_dir = Path("federated_dataset") / client_name
    
    train_generator = data_gen.flow_from_directory(
        dataset_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )
    
    # 4. Train locally
    # The user recommended 10 epochs. Since we're iterating over many rounds, we'll keep it at 10.
    model.fit(
        train_generator,
        epochs=10,
        verbose=1
    )
    
    # 5. Save trained weights
    os.makedirs('models', exist_ok=True)
    client_model_path = f"models/round{round_number}_{client_name}.h5"
    model.save(client_model_path)
    print(f"Client model saved to {client_model_path}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_client.py <client_name> <round_number>")
        sys.exit(1)
        
    c_name = sys.argv[1]
    r_num = int(sys.argv[2])
    train_client(c_name, r_num)
