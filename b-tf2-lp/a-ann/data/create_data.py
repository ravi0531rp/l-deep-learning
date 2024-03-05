import pandas as pd
import numpy as np
from utils import calculate_throttle, calculate_steering_angle, check_braking

num_samples = 30000

sensor_data = np.random.randint(0, 101, size=(num_samples, 8))

sensor_df = pd.DataFrame(sensor_data, columns=['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5', 'sensor6', 'sensor7', 'sensor8'])

sensor_df['throttle'] = sensor_df.apply(calculate_throttle, axis=1)
sensor_df['steering_angle'] = sensor_df.apply(calculate_steering_angle, axis=1)
sensor_df["brake"] = sensor_df.apply(check_braking, axis=1)
sensor_df.to_csv("../datasets/ann_sensors.csv", index=False)