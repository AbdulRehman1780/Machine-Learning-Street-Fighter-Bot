from command import Command
from buttons import Buttons
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import socket
import json
from game_state import GameState
import sys
import pandas as pd

def connect(port):
    """For making a connection with the game"""
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(("127.0.0.1", port))
        server_socket.listen(5)
        print(f"Waiting for game connection on port {port}...")
        print("Please make sure:")
        print("1. The Street Fighter game is running in BizHawk")
        print("2. No other bot instances are running")
        print("3. The Lua script is loaded in BizHawk")
        (client_socket, _) = server_socket.accept()
        print("Connected to game!")
        return client_socket
    except OSError as e:
        if e.errno == 10048:  # Port already in use
            print(f"\nError: Port {port} is already in use!")
            print("This could mean:")
            print("1. Another instance of the bot is already running")
            print("2. The game hasn't been started yet")
            print("\nPlease:")
            print("1. Close any other running bots")
            print("2. Start Street Fighter in BizHawk")
            print("3. Load the Lua script in BizHawk")
            print("4. Try running this bot again")
        else:
            print(f"Network error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error while connecting: {e}")
        sys.exit(1)

def send(client_socket, command):
    """Send command to the game"""
    command_dict = command.object_to_dict()
    pay_load = json.dumps(command_dict).encode()
    client_socket.sendall(pay_load)

def receive(client_socket):
    """Receive game state from the game"""
    pay_load = client_socket.recv(4096)
    input_dict = json.loads(pay_load.decode())
    game_state = GameState(input_dict)
    return game_state

class MLBot:
    def __init__(self):
        """Initialize the ML-based bot."""
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        self.feature_names = self._load_feature_names()
        self.my_command = Command()
        
    def _load_model(self):
        """Load the trained Random Forest model."""
        model_path = 'models/random_forest_model.joblib'
        if not os.path.exists(model_path):
            raise FileNotFoundError("Random Forest model not found. Please train the model first.")
        return joblib.load(model_path)
    
    def _load_scaler(self):
        """Load the fitted scaler."""
        scaler_path = 'models/scaler.joblib'
        if not os.path.exists(scaler_path):
            raise FileNotFoundError("Scaler not found. Please run preprocessing first.")
        return joblib.load(scaler_path)
    
    def _load_feature_names(self):
        """Load the feature names used during training."""
        feature_names_path = 'models/feature_names.joblib'
        if not os.path.exists(feature_names_path):
            raise FileNotFoundError("Feature names not found. Please run preprocessing first.")
        return joblib.load(feature_names_path)
    
    def _prepare_state_features(self, game_state):
        """
        Prepare game state features for model input.
        
        Args:
            game_state: Current GameState object
        
        Returns:
            features: Numpy array of preprocessed features
        """
        # Create numerical features as a DataFrame with correct feature names
        numerical_data = {
            'timer': game_state.timer,
            'p1_health': game_state.player1.health,
            'p1_x': game_state.player1.x_coord,
            'p1_y': game_state.player1.y_coord,
            'p1_move_id': game_state.player1.move_id,
            'p2_health': game_state.player2.health,
            'p2_x': game_state.player2.x_coord,
            'p2_y': game_state.player2.y_coord,
            'p2_move_id': game_state.player2.move_id,
            'distance_between_players': abs(game_state.player1.x_coord - game_state.player2.x_coord)
        }
        
        # Convert to DataFrame with correct column order
        numerical_df = pd.DataFrame([numerical_data])[self.feature_names]
        
        # Scale numerical features
        scaled_numerical = self.scaler.transform(numerical_df)
        
        # Create boolean features
        boolean_features = np.array([
            float(game_state.player1.is_jumping),
            float(game_state.player1.is_crouching),
            float(game_state.player1.is_player_in_move),
            float(game_state.player2.is_jumping),
            float(game_state.player2.is_crouching),
            float(game_state.player2.is_player_in_move)
        ]).reshape(1, -1)
        
        # Combine scaled numerical and boolean features
        features = np.hstack([scaled_numerical, boolean_features])
        
        return features
    
    def _predict_actions(self, features):
        """
        Predict actions using the Random Forest model.
        
        Args:
            features: Preprocessed game state features
        
        Returns:
            actions: Predicted button presses
        """
        actions = self.model.predict(features)
        return actions[0]  # return first (and only) prediction
    
    def fight(self, game_state, player_num):
        """
        Decide actions based on current game state using the ML model.
        
        Args:
            game_state: Current GameState object
            player_num: Player number (1 or 2)
        
        Returns:
            command: Command object with predicted button presses
        """
        # Prepare features from game state
        features = self._prepare_state_features(game_state)
        
        # Get model predictions
        actions = self._predict_actions(features)
        
        # Create buttons object with predicted actions
        buttons = Buttons()
        buttons.A = bool(actions[0])
        buttons.B = bool(actions[1])
        buttons.X = bool(actions[2])
        buttons.Y = bool(actions[3])
        buttons.L = bool(actions[4])
        buttons.R = bool(actions[5])
        buttons.up = bool(actions[6])
        buttons.down = bool(actions[7])
        buttons.left = bool(actions[8])
        buttons.right = bool(actions[9])
        
        # Set buttons for appropriate player
        if player_num == '1':
            self.my_command.player_buttons = buttons
        else:
            self.my_command.player2_buttons = buttons
        
        return self.my_command

def main():
    """Main function to run the ML bot"""
    if len(sys.argv) != 2:
        print("Usage: python ml_bot.py <player_num>")
        print("player_num: 1 or 2")
        sys.exit(1)

    player_num = sys.argv[1]

    # Connect to the game
    port = 9999 if player_num == '1' else 10000
    client_socket = connect(port)
    
    # Initialize the ML bot
    try:
        bot = MLBot()
        print("Using Random Forest model")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure you have trained the model first.")
        sys.exit(1)

    # Main game loop
    current_game_state = None
    while (current_game_state is None) or (not current_game_state.is_round_over):
        try:
            current_game_state = receive(client_socket)
            bot_command = bot.fight(current_game_state, player_num)
            send(client_socket, bot_command)
        except Exception as e:
            print(f"Error during gameplay: {e}")
            break

if __name__ == "__main__":
    main() 