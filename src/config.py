# src/config.py

class Config:
    # --- Model and Dataset ---
    MODEL_CHECKPOINT = "facebook/bart-large-cnn"
    DATASET_NAME = "xsum"
    
    # --- Data Preprocessing ---
    MAX_INPUT_LENGTH = 1024  # Max length of the input article
    MAX_TARGET_LENGTH = 128   # Max length of the generated summary
    
    # --- Training Hyperparameters ---
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    NUM_TRAIN_EPOCHS = 3
    
    # --- Directory for saving results ---
    OUTPUT_DIR = "./results"