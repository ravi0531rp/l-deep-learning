import pandas as pd
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        return pd.read_csv(self.data_path)

    def split_data(self, data, test_size=0.2, validation_size=0.2):
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
        train_data, val_data = train_test_split(train_data, test_size=validation_size, random_state=42)
        
        # Extract input features
        X_train = train_data.drop(['throttle', 'steering_angle', 'brake'], axis=1).values
        X_val = val_data.drop(['throttle', 'steering_angle', 'brake'], axis=1).values
        X_test = test_data.drop(['throttle', 'steering_angle', 'brake'], axis=1).values
        
        # Extract target data for training
        y_train = train_data[['throttle', 'steering_angle', 'brake']]
        y_train_regression_throttle = y_train['throttle'].values
        y_train_regression_steering = y_train['steering_angle'].values
        y_train_classification_brake = y_train['brake'].values
        
        # Extract target data for validation
        y_val = val_data[['throttle', 'steering_angle', 'brake']]
        y_val_regression_throttle = y_val['throttle'].values
        y_val_regression_steering = y_val['steering_angle'].values
        y_val_classification_brake = y_val['brake'].values
        
        # Extract target data for testing
        y_test = test_data[['throttle', 'steering_angle', 'brake']]
        y_test_regression_throttle = y_test['throttle'].values
        y_test_regression_steering = y_test['steering_angle'].values
        y_test_classification_brake = y_test['brake'].values
        
        return X_train, X_val, X_test, y_train_regression_throttle, y_train_regression_steering, y_train_classification_brake, y_val_regression_throttle, y_val_regression_steering, y_val_classification_brake, y_test_regression_throttle, y_test_regression_steering, y_test_classification_brake
