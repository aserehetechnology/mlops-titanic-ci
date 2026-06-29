import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def main():
    # Load processed data
    df = pd.read_csv("namadataset_preprocessing/titanic_processed.csv")
    
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Set tracking URI to localhost
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment("Titanic_Prediction_Basic")
    
    # Enable autolog
    mlflow.sklearn.autolog()
    
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        print("Model training completed with MLflow autolog.")

if __name__ == "__main__":
    main()
