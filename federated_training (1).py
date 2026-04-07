import sys
import subprocess
import os

def run_federated_learning(total_rounds=10, num_clients=4):
    print("--- Initializing Global Model ---")
    
    # Reset tracking log
    if os.path.exists('metrics_log.csv'):
        os.remove('metrics_log.csv')
    with open('metrics_log.csv', 'w') as f:
        f.write("Round,Accuracy\n")
        
    subprocess.run([sys.executable, "initialize_global_model.py"], check=True)
    
    for r in range(1, total_rounds + 1):
        print(f"\n{'='*40}")
        print(f"      STARTING ROUND {r}/{total_rounds}")
        print(f"{'='*40}\n")
        
        # 1. Train local clients
        for i in range(1, num_clients + 1):
            client_name = f"client{i}"
            subprocess.run([sys.executable, "train_client.py", client_name, str(r)], check=True)
            
        # 2. Server Aggregation (Weighted FedAvg)
        subprocess.run([sys.executable, "server_aggregate.py", str(r)], check=True)

        # 3. Evaluation and Metrics tracking
        is_final = "True" if r == total_rounds else "False"
        subprocess.run([sys.executable, "evaluate_model.py", str(r), is_final], check=True)

    print(f"\n{'*'*40}")
    print("   FEDERATED TRAINING COMPLETED!")
    print(f"{'*'*40}\n")
    
    print("--- Final Global Model Accuracy Trend ---")
    if os.path.exists('metrics_log.csv'):
        with open('metrics_log.csv', 'r') as f:
            print(f.read())
            
    print("\nResult generated! Check the `confusion_matrix_*.png` for the heat map.")

if __name__ == "__main__":
    run_federated_learning(total_rounds=10, num_clients=4)
