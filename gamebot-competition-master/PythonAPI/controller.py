import socket
import json
from game_state import GameState
#from bot import fight
import sys
from ml_bot import MLBot  # Import ML bot instead of original bot
import csv
import os
from datetime import datetime

def connect(port):
    #For making a connection with the game
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("127.0.0.1", port))
    server_socket.listen(5)
    (client_socket, _) = server_socket.accept()
    print ("Connected to game!")
    return client_socket

def send(client_socket, command):
    #This function will send your updated command to Bizhawk so that game reacts according to your command.
    command_dict = command.object_to_dict()
    pay_load = json.dumps(command_dict).encode()
    client_socket.sendall(pay_load)

def receive(client_socket):
    #receive the game state and return game state
    pay_load = client_socket.recv(4096)
    input_dict = json.loads(pay_load.decode())
    game_state = GameState(input_dict)

    return game_state

def log_game_data(game_state, player_num, command):
    """
    Logs game state and command data to a CSV file.
    
    Args:
        game_state: Current GameState object
        player_num: Player number (1 or 2)
        command: Command object containing button presses
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Use a fixed filename instead of timestamp-based name
    filename = f'logs/game_data.csv'
    
    # Define CSV headers with organized sections
    headers = [
        # Time and Game Info
        'timestamp',
        'player_num',
        'timer',
        'has_round_started',
        'is_round_over',
        'fight_result',
        
        # Player 1 State
        'p1_character_id',
        'p1_health',
        'p1_x',
        'p1_y',
        'p1_jumping',
        'p1_crouching',
        'p1_is_player_in_move',
        'p1_move_id',
        
        # Player 2 State
        'p2_character_id',
        'p2_health',
        'p2_x',
        'p2_y',
        'p2_jumping',
        'p2_crouching',
        'p2_is_player_in_move',
        'p2_move_id',
        
        # Distance between players
        'distance_between_players',
        
        # Button States (Actions)
        'a',
        'b',
        'x',
        'y',
        'l',
        'r',
        'start',
        'select',
        'up',
        'down',
        'left',
        'right'
    ]
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        
        if not file_exists:
            writer.writeheader()
        
        # Extract button states from command - using correct attribute names
        buttons = command.player_buttons if player_num == '1' else command.player2_buttons
        
        # Calculate distance between players
        distance = abs(game_state.player1.x_coord - game_state.player2.x_coord)
        
        # Create row data
        row = {
            # Time and Game Info
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
            'player_num': player_num,
            'timer': game_state.timer,
            'has_round_started': game_state.has_round_started,
            'is_round_over': game_state.is_round_over,
            'fight_result': game_state.fight_result,
            
            # Player 1 State
            'p1_character_id': game_state.player1.player_id,
            'p1_health': game_state.player1.health,
            'p1_x': game_state.player1.x_coord,
            'p1_y': game_state.player1.y_coord,
            'p1_jumping': game_state.player1.is_jumping,
            'p1_crouching': game_state.player1.is_crouching,
            'p1_is_player_in_move': game_state.player1.is_player_in_move,
            'p1_move_id': game_state.player1.move_id,
            
            # Player 2 State
            'p2_character_id': game_state.player2.player_id,
            'p2_health': game_state.player2.health,
            'p2_x': game_state.player2.x_coord,
            'p2_y': game_state.player2.y_coord,
            'p2_jumping': game_state.player2.is_jumping,
            'p2_crouching': game_state.player2.is_crouching,
            'p2_is_player_in_move': game_state.player2.is_player_in_move,
            'p2_move_id': game_state.player2.move_id,
            
            # Distance between players
            'distance_between_players': distance,
            
            # Button States (Actions)
            'a': buttons.A,
            'b': buttons.B,
            'x': buttons.X,
            'y': buttons.Y,
            'l': buttons.L,
            'r': buttons.R,
            'start': buttons.start,
            'select': buttons.select,
            'up': buttons.up,
            'down': buttons.down,
            'left': buttons.left,
            'right': buttons.right
        }
        
        writer.writerow(row)
        
        # Print progress every 100 frames
        if game_state.timer % 100 == 0:
            print(f"Logging frame {game_state.timer}...")

def main():
    if (sys.argv[1]=='1'):
        client_socket = connect(9999)
    elif (sys.argv[1]=='2'):
        client_socket = connect(10000)
    current_game_state = None
    #print( current_game_state.is_round_over )
    
    # Use ML bot instead of original bot
    bot = MLBot(model_type='rf')  # or 'nn' for neural network
    
    while (current_game_state is None) or (not current_game_state.is_round_over):
        current_game_state = receive(client_socket)
        bot_command = bot.fight(current_game_state, sys.argv[1])
        # Log the game data before sending the command
        log_game_data(current_game_state, sys.argv[1], bot_command)
        send(client_socket, bot_command)

if __name__ == '__main__':
   main()
