import os
import random
import shutil
from pathlib import Path

def distribute_iid_data():
    dataset_path = Path(r"d:\Federated_PDD\PlantVillageDataset\color")
    output_dir = Path(r"d:\Federated_PDD\federated_dataset")
    num_clients = 4
    
    # Create federated directory and client subdirectories
    for i in range(1, num_clients + 1):
        client_dir = output_dir / f"client{i}"
        client_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each class
    classes = [d for d in dataset_path.iterdir() if d.is_dir()]
    print(f"Found {len(classes)} classes in the dataset.")
    
    for class_dir in classes:
        class_name = class_dir.name
        images = [f for f in class_dir.iterdir() if f.is_file()]
        random.shuffle(images)
        
        total_images = len(images)
        images_per_client = total_images // num_clients
        
        print(f"Class: {class_name:<40} Total: {total_images:<6} Per Client: {images_per_client}")
        
        # Create class subdirectories for all clients
        for i in range(1, num_clients + 1):
            (output_dir / f"client{i}" / class_name).mkdir(parents=True, exist_ok=True)
            
        # Distribute images
        for i in range(num_clients):
            start_idx = i * images_per_client
            # the last client gets all remaining images if not perfectly divisible
            if i == num_clients - 1:
                end_idx = total_images
            else:
                end_idx = start_idx + images_per_client
                
            client_images = images[start_idx:end_idx]
            
            client_dir = output_dir / f"client{i+1}" / class_name
            for img in client_images:
                shutil.copy2(img, client_dir / img.name)

    print("\nData distribution complete.")

    # Verification
    print("\n--- Verification ---")
    for i in range(1, num_clients + 1):
        client_dir = output_dir / f"client{i}"
        client_classes = [d for d in client_dir.iterdir() if d.is_dir()]
        total_client_images = sum(len(list(c.iterdir())) for c in client_classes)
        print(f"Client {i}: {len(client_classes)} classes, {total_client_images} total images")

if __name__ == "__main__":
    distribute_iid_data()
