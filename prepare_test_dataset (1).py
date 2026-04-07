import os
import shutil
import random
from pathlib import Path

def create_test_dataset(test_split=0.1):
    base_dir = Path("federated_dataset")
    test_dir = Path("test_dataset")
    num_clients = 4
    
    test_dir.mkdir(parents=True, exist_ok=True)
    
    classes = [d.name for d in (base_dir / "client1").iterdir() if d.is_dir()]
    
    for class_name in classes:
        (test_dir / class_name).mkdir(parents=True, exist_ok=True)
        
        for i in range(1, num_clients + 1):
            client_class_dir = base_dir / f"client{i}" / class_name
            
            images = [f for f in client_class_dir.iterdir() if f.is_file()]
            num_test = int(len(images) * test_split)
            
            # Select random images to move
            test_images = random.sample(images, num_test)
            
            for img in test_images:
                shutil.move(img, test_dir / class_name / img.name)
                
    print(f"Created test dataset in {test_dir} by moving {test_split*100}% of data from each client.")
    
if __name__ == "__main__":
    create_test_dataset()
