import os
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path

def aggregate_weights(round_number, num_clients=4):
    print(f"\n--- Weighted Aggregating weights for Round {round_number} ---")
    
    # 1. Get dataset sizes for weighting
    client_image_counts = []
    for i in range(1, num_clients + 1):
        client_dir = Path("federated_dataset") / f"client{i}"
        # Count only files (images)
        count = sum(1 for _ in client_dir.rglob('*') if _.is_file())
        client_image_counts.append(count)
        
    total_images = sum(client_image_counts)
    weights = [count / total_images for count in client_image_counts]
    print(f"Client data weights based on size: {[f'{w:.4f}' for w in weights]}")
    
    # 2. Load models
    client_models = []
    for i in range(1, num_clients + 1):
        client_name = f"client{i}"
        model_path = f"models/round{round_number}_{client_name}.h5"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Client model {model_path} not found.")
        print(f"Loading {model_path}...")
        client_models.append(load_model(model_path))
        
    # 3. Weighted Averaging
    global_model = client_models[0]
    global_weights = global_model.get_weights()
    
    # Initialize global weights to 0
    import numpy as np
    for j in range(len(global_weights)):
        global_weights[j] = np.zeros_like(global_weights[j])
        
    # Apply weights
    for i in range(num_clients):
        client_weights = client_models[i].get_weights()
        for j in range(len(global_weights)):
            global_weights[j] += client_weights[j] * weights[i]
            
    # Set aggregated weights to the global model
    global_model.set_weights(global_weights)
    
    # Save the new global model
    global_model_path = f"models/global_weights_round{round_number}.h5"
    global_model.save(global_model_path)
    print(f"Aggregated global model saved to {global_model_path}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python server_aggregate.py <round_number>")
        sys.exit(1)
        
    r_num = int(sys.argv[1])
    aggregate_weights(r_num)
