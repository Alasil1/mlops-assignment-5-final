import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train():
    print("Generating a large, complex dataset...")
    # Creating a much harder classification problem with 5000 samples and 4 distinct classes
    X, y = make_classification(
        n_samples=5000, 
        n_features=20, 
        n_informative=12, 
        n_redundant=2, 
        n_classes=4,
        flip_y=0.05, # adds a tiny bit of noise
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Training Random Forest model...")
    # To PASS the pipeline (> 0.85): Use n_estimators=100, max_depth=15
    # To FAIL the pipeline (< 0.85): Use n_estimators=2, max_depth=2
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    
    # Start MLflow run
    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Log params and metrics
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 15)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save Run ID to model_info.txt
        run_id = run.info.run_id
        with open("model_info.txt", "w") as f:
            f.write(run_id)
        
        print(f"Run ID {run_id} saved to model_info.txt")

if __name__ == "__main__":
    train()
