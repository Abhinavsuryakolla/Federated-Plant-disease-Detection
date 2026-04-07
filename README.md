# Federated Plant Disease Detection

This project implements a Federated Learning approach for detecting plant diseases using a distributed version of the PlantVillage dataset. The model architecture used is MobileNetV2.

## Process Overview

Federated Learning allows multiple clients (e.g., different farms or institutions) to collaboratively train a centralized global machine learning model while keeping their training data localized. The typical workflow is:
1. Distribute a dataset among multiple simulated clients.
2. Prepare a centralized test dataset for evaluation.
3. Initialize a global model.
4. Iteratively run training rounds:
    - Train individual client models locally on their data starting from the global model.
    - Aggregate the updated weights from all clients on the server to form a newly updated global model (using Federated Averaging).
    - Evaluate the new global model on the test dataset.

## The Scripts

The project consists of several independent scripts that form the pipeline, and an orchestrator script that runs them together.

### Data Preparation
*   **`distribute_data (1).py`**: Reads the central dataset (PlantVillage) and divides it equally among a specified number of clients (default 4) into an IID (Independent and Identically Distributed) `federated_dataset` directory.
*   **`prepare_test_dataset (1).py`**: Takes a random portion (10% by default) of the clients' distributed data, removes it from the clients, and aggregates it into a centralized `test_dataset` to be used for evaluating the global model's performance.

### Federated Learning Pipeline
*   **`initialize_global_model (1).py`**: Sets up the initial global model. It uses the MobileNetV2 architecture pre-trained on ImageNet, customizes the output layers for the number of plant classes, and saves the initial weights (`round0`).
*   **`train_client (1).py`**: Represents the local training on a client. It loads the latest global model, trains it on the specific client's data (with data augmentation like flip, zoom, rotation, brightness adjustments), and saves the fine-tuned client model weights. It also has advanced logic to unfreeze the MobileNetV2 base layers for fine-tuning after round 5.
*   **`server_aggregate (1).py`**: The central aggregation step. It loads the models trained by all clients in the current round, computes a weighted average of their weights (based on the number of images each client holds), and saves the resulting weights as the new global model.
*   **`evaluate_model (1).py`**: Evaluates the newly aggregated global model against the centralized `test_dataset`. It calculates the model's accuracy, precision, recall, and F1-score and logs it to `metrics_log.csv`. In the final round, it also generates and saves a confusion matrix as an image.

### Orchestration and Standalone Testing
*   **`federated_training (1).py`**: The main orchestrator script. Running this script automates the entire Federated Learning process. It initializes the model and loops through the specified number of training rounds (default 10), executing the client training, server aggregation, and evaluation scripts via subprocesses in sequence.
*   **`test (1).py`**: A standalone evaluation script that calculates and prints out basic metrics for any given model file path against the test dataset.

## How to Run

1.  **Prepare the Data**: 
    Ensure you have the PlantVillage dataset available at the path specified in `distribute_data (1).py` and run:
    ```bash
    python "distribute_data (1).py"
    python "prepare_test_dataset (1).py"
    ```
2.  **Run Federated Training**:
    Start the automated federated learning process:
    ```bash
    python "federated_training (1).py"
    ```
    This script will take care of initializing the model, running all rounds of client training and server aggregation, and evaluating the results.

3.  **Evaluate a specific model**:
    ```bash
    python "test (1).py" <path_to_model_file.h5>
    ```

## Outputs
*   **`models/`**: Directory containing the saved model files (`.h5`) for the global model at each round, as well as the individual client models for each round.
*   **`federated_dataset/`**: Directory containing the distributed data for each client.
*   **`test_dataset/`**: Directory containing the data used for evaluation.
*   **`metrics_log.csv`**: A CSV file tracking the global model's accuracy across all rounds.
*   **`confusion_matrix_*.png`**: Heatmap of the confusion matrix generated after the final round.
