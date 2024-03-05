import tensorflow as tf
from data.preprocess import DataPreprocessor
from models.neural_network import NeuralNetwork
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from loguru import logger
from datetime import datetime

# Setup logging
logger.add("logs/train.log")

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 32

# Load and preprocess data
data_preprocessor = DataPreprocessor("datasets/ann_sensors.csv")
data = data_preprocessor.load_data()
X_train, X_val, X_test, y_train, y_val, y_test = data_preprocessor.split_data(data)

y_train_regression_throttle = y_train['throttle'].values
y_train_regression_steering = y_train['steering_angle'].values
y_train_classification_brake = y_train['brake'].values

y_val_regression_throttle = y_val['throttle'].values
y_val_regression_steering = y_val['steering_angle'].values
y_val_classification_brake = y_val['brake'].values

y_test_regression_throttle = y_test['throttle'].values
y_test_regression_steering = y_test['steering_angle'].values
y_test_classification_brake = y_test['brake'].values

logger.debug(X_train.columns)
# Define model
model = NeuralNetwork().build()

# Compile the model
model.compile(optimizer='adam',
              loss={'output_regression_throttle': 'mean_squared_error', 
                    'output_regression_steering': 'mean_squared_error', 
                    'output_classification_brake': 'binary_crossentropy'},
              metrics={'output_regression_throttle': ['mae', 'mse'], 
                       'output_regression_steering': ['mae', 'mse'],
                       'output_classification_brake': 'accuracy'})
# Define callbacks
# checkpoint_callback = ModelCheckpoint(filepath='checkpoints/checkpoint.h5', save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_output_classification_brake_accuracy', patience=5, mode='max')

# Train the model
history = model.fit(X_train, [y_train_regression_throttle, y_train_regression_steering, y_train_classification_brake], 
                    epochs=EPOCHS, batch_size=BATCH_SIZE, 
                    validation_data=(X_val, [y_val_regression_throttle, y_val_regression_steering, y_val_classification_brake]),
                    callbacks=[ early_stopping_callback]) # checkpoint_callback,
# Logging
logger.info("Training completed.")

# Evaluate on test set
test_loss = model.evaluate(X_test, [y_test_regression_throttle, y_test_regression_steering, y_test_classification_brake])

# Unpack the test_losses list
logger.debug(test_loss)
# Save model
model.save('my_model.keras')
# TensorBoard logs
log_dir = "logs/tensorboard/train/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
