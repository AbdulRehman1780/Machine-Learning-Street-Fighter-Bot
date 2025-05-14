import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import joblib

def load_and_clean_data(csv_path='logs/game_data.csv'):
    """
    Load and clean the game data.
    
    Args:
        csv_path: Path to the CSV file containing game data
    
    Returns:
        cleaned_df: Cleaned pandas DataFrame
    """
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Remove rows where the game hasn't started or is over
    df = df[df['has_round_started'] == True]
    
    # Remove rows with missing values
    df = df.dropna()
    
    # Convert boolean columns to int (True -> 1, False -> 0)
    bool_columns = [
        'has_round_started', 'is_round_over',
        'p1_jumping', 'p1_crouching', 'p1_is_player_in_move',
        'p2_jumping', 'p2_crouching', 'p2_is_player_in_move',
        'a', 'b', 'x', 'y', 'l', 'r', 'start', 'select',
        'up', 'down', 'left', 'right'
    ]
    for col in bool_columns:
        df[col] = df[col].astype(int)
    
    return df

def encode_categorical_variables(df):
    """
    Encode categorical variables using LabelEncoder.
    
    Args:
        df: Input DataFrame
    
    Returns:
        df: DataFrame with encoded categorical variables
        encoders: Dictionary of fitted LabelEncoders
    """
    encoders = {}
    categorical_columns = ['fight_result', 'p1_character_id', 'p2_character_id']
    
    for col in categorical_columns:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col])
    
    return df, encoders

def normalize_numerical_features(df):
    """
    Normalize numerical features using StandardScaler.
    
    Args:
        df: Input DataFrame
    
    Returns:
        df: DataFrame with normalized numerical features
        scaler: Fitted StandardScaler
    """
    numerical_columns = [
        'timer', 'p1_health', 'p1_x', 'p1_y', 'p1_move_id',
        'p2_health', 'p2_x', 'p2_y', 'p2_move_id',
        'distance_between_players'
    ]
    
    # Create and fit scaler with feature names
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    # Save the scaler and feature names
    if not os.path.exists('models'):
        os.makedirs('models')
    scaler_path = 'models/scaler.joblib'
    feature_names_path = 'models/feature_names.joblib'
    
    joblib.dump(scaler, scaler_path)
    joblib.dump(numerical_columns, feature_names_path)
    print(f"Saved scaler to {scaler_path}")
    print(f"Saved feature names to {feature_names_path}")
    
    return df, scaler

def prepare_features_and_labels(df):
    """
    Prepare feature matrix X and label vector y.
    
    Args:
        df: Input DataFrame
    
    Returns:
        X: Feature matrix
        y: Label vector (using button presses as labels)
    """
    # Define features to use
    feature_columns = [
        'timer', 'p1_health', 'p1_x', 'p1_y', 'p1_jumping', 'p1_crouching',
        'p1_is_player_in_move', 'p1_move_id', 'p2_health', 'p2_x', 'p2_y',
        'p2_jumping', 'p2_crouching', 'p2_is_player_in_move', 'p2_move_id',
        'distance_between_players'
    ]
    
    # Button presses will be our labels
    label_columns = ['a', 'b', 'x', 'y', 'l', 'r', 'up', 'down', 'left', 'right']
    
    X = df[feature_columns]
    y = df[label_columns]
    
    return X, y

def preprocess_data(csv_path='logs/game_data.csv', test_size=0.2, random_state=42):
    """
    Main function to preprocess the game data.
    
    Args:
        csv_path: Path to the CSV file
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
    """
    print("Loading and cleaning data...")
    df = load_and_clean_data(csv_path)
    
    print("Normalizing numerical features...")
    df, scaler = normalize_numerical_features(df)
    
    print("Preparing features and labels...")
    X, y = prepare_features_and_labels(df)
    
    print("Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Save preprocessed data
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')
    
    np.save('processed_data/X_train.npy', X_train)
    np.save('processed_data/X_test.npy', X_test)
    np.save('processed_data/y_train.npy', y_train)
    np.save('processed_data/y_test.npy', y_test)
    
    print("Data preprocessing completed!")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_data() 