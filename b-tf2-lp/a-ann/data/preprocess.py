import pandas as pd
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        return pd.read_csv(self.data_path)

    def split_data(self, data, test_size=0.2, validation_size=0.2):
        # Split features (X) and target variables (y)
        X = data.iloc[:, :8]  
        y = data.iloc[:, 8:]  

        # Split data into train, validation, and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=42)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
