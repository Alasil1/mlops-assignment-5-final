import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train():
    # Load datase
    print("Loading breast cancer dataset...")
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # Train model
    # To intentionally lower accuracy below 0.85 (e.g., for testing failure screenshot), 
    # you can change n_estimators=1 and max_depth=1
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=1,max_depth = 1, random_state=42)
    
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
