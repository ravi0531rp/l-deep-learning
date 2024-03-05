import argparse
import tensorflow as tf
from data.preprocess import DataPreprocessor
from models.neural_network import NeuralNetwork
from tensorflow.keras.callbacks import EarlyStopping
from loguru import logger
from datetime import datetime
from utils import plot_metrics

def setup_logging():
    """Setup logging configuration."""
    log_file = f"logs/train.log"
    logger.add(log_file, rotation="100 MB", level="DEBUG")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train neural network.")
    parser.add_argument("--data_path", type=str, default="datasets/ann_sensors.csv",
                        help="Path to the dataset CSV file.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training.")
    return parser.parse_args()

def main():
    """Main function for training the neural network."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging()
    
    # Load and preprocess data
    data_preprocessor = DataPreprocessor(args.data_path)
    data = data_preprocessor.load_data()
    X_train, X_val, X_test, y_train_regression_throttle, y_train_regression_steering, y_train_classification_brake, y_val_regression_throttle, y_val_regression_steering, y_val_classification_brake, y_test_regression_throttle, y_test_regression_steering, y_test_classification_brake = data_preprocessor.split_data(data)

    # Define model
    model = NeuralNetwork().build()
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss={'output_regression_throttle': 'mean_squared_error', 
                        'output_regression_steering': 'mean_squared_error', 
                        'output_classification_brake': 'binary_crossentropy'},
                  metrics={'output_regression_throttle': ['mae', 'mse'], 
                           'output_regression_steering': ['mae', 'mse'],
                           'output_classification_brake': ['accuracy']})
    
    # Define callbacks
    early_stopping_callback = EarlyStopping(monitor='val_output_classification_brake_accuracy', 
                                            patience=5, mode='max')
    
    # Train the model
    history = model.fit(X_train, [y_train_regression_throttle, y_train_regression_steering, y_train_classification_brake], 
                        epochs=args.epochs, batch_size=args.batch_size, 
                        validation_data=(X_val, [y_val_regression_throttle, y_val_regression_steering, y_val_classification_brake]),
                        callbacks=[early_stopping_callback])

    logger.info("Training completed.")
    
        # Evaluate on test set
    test_results = model.evaluate(X_test, [y_test_regression_throttle, y_test_regression_steering, y_test_classification_brake])
    logger.info("Test Results: {}".format(test_results))

  
    # Save the model
    model.save('./checkpoints/ann.keras')

if __name__ == "__main__":
    main()
