import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import joblib
import os

def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Train a Random Forest model for multi-label classification.
    
    Args:
        X_train, X_test: Training and testing features
        y_train, y_test: Training and testing labels
    
    Returns:
        trained_model: Trained Random Forest model
    """
    # Create and train the multi-output random forest
    base_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    rf_model = MultiOutputClassifier(base_rf)
    
    print("Training Random Forest model...")
    rf_model.fit(X_train, y_train)
    
    # Evaluate the model
    train_score = rf_model.score(X_train, y_train)
    test_score = rf_model.score(X_test, y_test)
    
    print(f"Random Forest Training Score: {train_score:.4f}")
    print(f"Random Forest Testing Score: {test_score:.4f}")
    
    # Save the model
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(rf_model, 'models/random_forest_model.joblib')
    
    return rf_model

if __name__ == "__main__":
    # Load preprocessed data
    print("Loading preprocessed data...")
    X_train = np.load('processed_data/X_train.npy')
    X_test = np.load('processed_data/X_test.npy')
    y_train = np.load('processed_data/y_train.npy')
    y_test = np.load('processed_data/y_test.npy')
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Train Random Forest model
    rf_model = train_random_forest(X_train, X_test, y_train, y_test)
    print("Training completed and model saved!") 