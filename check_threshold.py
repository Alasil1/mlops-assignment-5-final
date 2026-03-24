import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

def check_accuracy():
    # Read the run ID
    if not os.path.exists("model_info.txt"):
        print("Error: model_info.txt not found!")
        sys.exit(1)
        
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()
        
    if not run_id:
        print("Error: Run ID is empty!")
        sys.exit(1)
        
    print(f"Checking accuracy for Run ID: {run_id}")
    
    client = MlflowClient()
    
    try:
        run = client.get_run(run_id)
        metrics = run.data.metrics
        
        accuracy = metrics.get("accuracy", 0.0)
        print(f"Found accuracy: {accuracy:.4f}")
        
        if accuracy >= 0.85:
            print(f"Validation passed! Accuracy {accuracy:.4f} is >= 0.85")
            sys.exit(0)
        else:
            print(f"Validation failed! Accuracy {accuracy:.4f} is < 0.85")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error accessing MLflow run: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_accuracy()
