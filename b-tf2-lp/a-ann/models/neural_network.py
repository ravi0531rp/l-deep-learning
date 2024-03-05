import tensorflow as tf

class NeuralNetwork():
    def __init__(self):
        self.dense1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units = 2)

    def build(self):
        inputs = tf.keras.Input(shape=(8,))
        x = tf.keras.layers.Dense(units=64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(units=32, activation='relu')(x)

        output_regression_throttle = tf.keras.layers.Dense(units=1, name='output_regression_throttle')(x)  # Regression for throttle
        output_regression_steering = tf.keras.layers.Dense(units=1, name='output_regression_steering')(x)  # Regression for steering angle
        output_classification_brake = tf.keras.layers.Dense(units=1, activation='sigmoid', name='output_classification_brake')(x)  # Binary classification for brake

        # Define the model
        model = tf.keras.Model(inputs=inputs, outputs=[output_regression_throttle, output_regression_steering, output_classification_brake])
        return model
    

