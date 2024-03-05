import pandas as pd
import numpy as np

def calculate_throttle(sensor_data):
    avg_sensor_value = np.sqrt(np.mean(sensor_data))
    throttle = avg_sensor_value / 10  # Adjust scale if needed
    return throttle

def calculate_steering_angle(row):
    sum_first_four_sensors = row[['sensor1', 'sensor2', 'sensor3', 'sensor4']].sum()
    prod_last_four_sensors = row[['sensor5', 'sensor6', 'sensor7', 'sensor8']].prod()
    steering_angle = min(60, (prod_last_four_sensors/sum_first_four_sensors - 200) / 1000)  # Adjust scale if needed
    return steering_angle

def check_braking(row):
    sum_first_four_sensors = row[['sensor1', 'sensor2', 'sensor3', 'sensor4']].sum()
    sum_last_four_sensors = row[['sensor5', 'sensor6', 'sensor7', 'sensor8']].sum()
    if sum_first_four_sensors > sum_last_four_sensors:
        return 1
    else:
        return 0