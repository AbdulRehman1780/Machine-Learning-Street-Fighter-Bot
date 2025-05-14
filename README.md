# Machine Learning Street Fighter Bot

This project implements a machine learning-based bot for Street Fighter II Turbo using supervised learning. The bot learns from gameplay data to predict optimal button combinations based on the game state.

## System Requirements

- Windows 7 or above (64-bit)
- Python 3.8 or above
- BizHawk emulator (provided in the single-player folder)

## Installation

1. Create and activate a Python virtual environment (recommended):
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate
```

2. Install required Python packages:
```bash
pip install numpy pandas scikit-learn joblib
```

## Project Structure

- `controller.py`: Original controller for data collection
- `preprocess_data.py`: Data preprocessing and feature engineering
- `train_models.py`: Model training script
- `ml_bot.py`: ML-based bot implementation
- `models/`: Directory containing trained models
- `logs/`: Directory containing gameplay data
- `processed_data/`: Directory containing preprocessed training data

## Usage

### Step 1: Data Collection

1. Start BizHawk:
   - Open the single-player folder
   - Run `EmuHawk.exe`
   - From File menu, choose Open ROM
   - Select "Street Fighter II Turbo (U).smc"
   - From Tools menu, open Tool Box (Shift+T)

2. Collect training data:
```bash
python controller.py 1
```

3. In BizHawk:
   - Click the Gyroscope Bot icon in Tool Box
   - Select your character in normal mode
   - Play several rounds to collect diverse gameplay data
   - Data will be saved to `logs/game_data.csv`

### Step 2: Train the ML Model

1. Preprocess the collected data:
```bash
python preprocess_data.py
```
This will:
- Clean and normalize the data
- Split into training/testing sets
- Save preprocessed data to `processed_data/`
- Save the feature scaler to `models/`

2. Train the Random Forest model:
```bash
python train_models.py
```
This will:
- Train a Random Forest classifier
- Print training and testing scores
- Save the model to `models/random_forest_model.joblib`

### Step 3: Run the ML Bot

1. Start BizHawk as in Step 1

2. Run the ML bot:
```bash
python ml_bot.py 1
```

3. In BizHawk:
   - Click the Gyroscope Bot icon
   - Select your character
   - Watch the ML bot play!

## Model Details

The bot uses a Random Forest classifier with the following features:
- Game state: timer, health, coordinates
- Player states: jumping, crouching, move IDs
- Spatial features: distance between players

Target variables:
- Button combinations (A, B, X, Y, L, R, directions)

## Troubleshooting

1. Port in use error:
   - Make sure no other bot instances are running
   - Restart BizHawk
   - Try running the bot again

2. Connection issues:
   - Ensure BizHawk is running
   - Click the Gyroscope Bot icon in Tool Box
   - Make sure you've selected a character

3. Model performance:
   - Collect more training data
   - Try adjusting model parameters in `train_models.py`
   - Ensure diverse gameplay scenarios in training data

## File Descriptions

- `controller.py`: Original controller implementation for data collection
- `preprocess_data.py`: Data preprocessing pipeline
  - Cleans raw gameplay data
  - Normalizes numerical features
  - Handles categorical variables
  - Splits data into train/test sets

- `train_models.py`: Model training script
  - Implements Random Forest classifier
  - Handles multi-label classification
  - Saves trained model and metrics

- `ml_bot.py`: ML bot implementation
  - Loads trained model and scaler
  - Processes real-time game state
  - Makes action predictions
  - Handles game communication

## Contributing

To improve the bot:
1. Collect more training data
2. Experiment with feature engineering
3. Try different ML models
4. Optimize model parameters

## Known Limitations

- Bot performance depends on training data quality
- May not handle rare game situations well
- Limited to moves seen in training data
- No explicit combo system knowledge

## Future Improvements

- Implement LSTM for temporal patterns
- Add explicit combo recognition
- Improve feature engineering
- Add reinforcement learning components

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review error messages
3. Verify system requirements
4. Check Python package versions 
