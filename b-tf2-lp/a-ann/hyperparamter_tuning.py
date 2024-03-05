import tensorflow as tf
from loguru import logger
from kerastuner.tuners import RandomSearch
from data.preprocess import DataPreprocessor
from models.neural_network import NeuralNetwork

logger.add("logs/train.log")

# Load and preprocess data
data_preprocessor = DataPreprocessor("datasets/ann_sensors.csv")
data = data_preprocessor.load_data()
train_data, val_data, test_data = data_preprocessor.split_data(data)

# Define model building function
def build_model(hp):
    model = NeuralNetwork()
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Define tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='hyperparameter_tuning',
    project_name='neural_network')

# Perform hyperparameter tuning
tuner.search(train_data, validation_data=val_data, epochs=5)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
logger.success(best_hps)