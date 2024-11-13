# -*- coding: utf-8 -*-
"""
Trials for making time-series data smooth by Exponentially Weighted Average (EWA)

@author: alima
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


## Trial 1: with 10 data

def exponential_smoothing_trend(data, alpha):
      """
      Exponential smoothing with initial trend component.
    
      Args:
          data: The time series data to smooth.
          alpha: The smoothing parameter (0 < alpha < 1).
    
      Returns:
          The smoothed data with initial trend.
      """
    
      n = len(data)
      smoothed = np.zeros(n)
      # Calculate initial trend based on first two data points (adjust as needed)
      initial_trend = (data[1] - data[0]) / 1
    
      # Handle initial value (avoid NaN for the first smoothed value)
      smoothed[0] = data[0] + initial_trend
    
      for i in range(1, n):
        # Smoothed value with trend component
        smoothed[i] = alpha * (data[i] - smoothed[i-1]) + (1 - alpha) * (smoothed[i-1] + initial_trend)
    
      return smoothed

# Example usage
data = np.array([10, 12, 13, 15, 18, 17, 19, 21, 18, 16])  # Sample data
alpha = 0.3  # Smoothing parameter (adjust based on your data)

smoothed_data = exponential_smoothing_trend(data, alpha)

# Plot original and smoothed data
plt.figure(figsize=(10, 6))
plt.plot(data, marker = 'o', label = 'Original Data')
plt.plot(smoothed_data, marker = 'x', color = 'red', label = 'Smoothed Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Exponential Smoothing with Initial Trend (alpha={})'.format(alpha))
plt.legend()
plt.grid(True)
plt.show()




## Trial 2: With function 

def exponential_smoothing_trend(data, alpha):
      """
      Exponential smoothing with initial trend component for long time series.
    
      Args:
          data: The time series data to smooth (NumPy array).
          alpha: The smoothing parameter (0 < alpha < 1).
    
      Returns:
          The smoothed data with initial trend (NumPy array).
      """
    
      n = len(data)
      smoothed = np.zeros(n)
    
      # Handle initial value (avoid NaN for the first smoothed value)
      smoothed[0] = data[0]
    
      # Calculate initial trend based on a window at the beginning (adjust window size as needed)
      window_size = 10
      if window_size > n:
        raise ValueError("Window size for initial trend cannot exceed data length")
      initial_trend = np.mean(data[1:window_size]) - data[0]
    
      # Iterate through data, incorporating trend component
      for i in range(1, n):
        smoothed[i] = alpha * (data[i] - smoothed[i - 1]) + (1 - alpha) * (smoothed[i - 1] + initial_trend)
    
      return smoothed


def generate_sample_data(n_points):
      """
      Generates sample time series data with trend and noise.
    
      Args:
          n_points: The number of data points to generate.
    
      Returns:
          A NumPy array containing the sample time series data.
      """
    
      # Define trend slope and noise amplitude
      trend_slope = 0.5
      noise_amplitude = 5
    
      # Generate base data with linear trend
      base_data = np.arange(n_points) * trend_slope
    
      # Add random noise
      noise = np.random.randn(n_points) * noise_amplitude
    
      # Combine base data and noise
      data = base_data + noise
    
      return data

# Generate data with 100 and 1000 points
data_100 = generate_sample_data(100)
data_1000 = generate_sample_data(1000)


# Example usage (assuming you have your long time series data in a NumPy array named 'data')
alpha = 0.2  # Smoothing parameter (adjust based on your data)


## Data with 100 points
smoothed_data = exponential_smoothing_trend(data_100, alpha)
original_subset = data_100
smoothed_subset = smoothed_data


# Plot original and smoothed data (consider downsampling for visualization on long series)
plt.figure(figsize=(15, 8))  # Adjust figure size as needed
plt.plot(original_subset, marker = 'o', label = 'Original Data (Subset)')
plt.plot(smoothed_subset, marker = 'x', color = 'red', label = 'Smoothed Data (Subset)')
plt.xlabel('Time (Subset)')
plt.ylabel('Value')
plt.title('Exponential Smoothing with Trend (alpha={})'.format(alpha))
plt.legend()
plt.grid(True)
plt.show()


## Data with 1000 points

smoothed_data = exponential_smoothing_trend(data_1000, alpha)
original_subset = data_1000
smoothed_subset = smoothed_data

# Plot original and smoothed data (consider downsampling for visualization on long series)
plt.figure(figsize=(15, 8))  # Adjust figure size as needed
plt.plot(original_subset, marker = 'o', label = 'Original Data (Subset)')
plt.plot(smoothed_subset, marker = 'x', color = 'red', label = 'Smoothed Data (Subset)')
plt.xlabel('Time (Subset)')
plt.ylabel('Value')
plt.title('Exponential Smoothing with Trend (alpha={})'.format(alpha))
plt.legend()
plt.grid(True)
plt.show()


## Trial 3: 100 and 1000 data with 0.3 and 0.1 alpha for first 50 observations

def exponential_smoothing_trend(data, alpha):
      """
      Exponential smoothing with initial trend component.
    
      Args:
          data: The time series data to smooth (NumPy array).
          alpha: The smoothing parameter (0 < alpha < 1).
    
      Returns:
          The smoothed data with initial trend (NumPy array).
      """
    
      n = len(data)
      smoothed = np.zeros(n)
    
      # Handle initial value (avoid NaN for the first smoothed value)
      smoothed[0] = data[0]
    
      # Calculate initial trend based on a window at the beginning (adjust window size as needed)
      window_size = 10
      if window_size > n:
        raise ValueError("Window size for initial trend cannot exceed data length")
      initial_trend = np.mean(data[1:window_size]) - data[0]
    
      # Iterate through data, incorporating trend component
      for i in range(1, n):
        smoothed[i] = alpha * (data[i] - smoothed[i - 1]) + (1 - alpha) * (smoothed[i - 1] + initial_trend)
    
      return smoothed


# Smoothing parameters (adjust as needed)
alpha_100 = 0.3  # Smoothing parameter for 100 data points
alpha_1000 = 0.1  # Smoothing parameter for 1000 data points (adjust for different data lengths)

# Apply smoothing
smoothed_data_100 = exponential_smoothing_trend(data_100.copy(), alpha_100)
smoothed_data_1000 = exponential_smoothing_trend(data_1000.copy(), alpha_1000)

# Plotting (consider adjusting subset length for visualization)
subset_length = 50  # Adjust subset length to show a representative portion

plt.figure(figsize=(12, 6))
# Plot for 100 data points
plt.plot(data_100[:subset_length], marker = 'o', label = 'Original Data (100 pts)')
plt.plot(smoothed_data_100[:subset_length], marker = 'x', color = 'red', label = 'Smoothed Data (100 pts)')
# Plot for 1000 data points (subset)
plt.plot(data_1000[:subset_length], marker = 'o', alpha = 0.3, label = 'Original Data (1000 pts)')  # Reduce alpha for better visibility
plt.plot(smoothed_data_1000[:subset_length], marker = 'o', alpha = 0.1, label = 'Smoothed Data (1000 pts)')  # Reduce alpha for better visibility
plt.legend()
plt.grid(True)
plt.show()

         

## Trial 4: Data with big fluctuations with emw function
         
# Generate example time-series data with 100 data points
np.random.seed(0)
time_series = pd.Series(np.random.randn(100))

# Apply EWMA smoothing
smoothened_data = time_series.ewm(alpha=0.2, adjust=False).mean()

# Plotting the raw and smoothened data
plt.figure(figsize=(10, 6))
plt.plot(time_series, label='Raw Data')
plt.plot(smoothened_data, label='Smoothened Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Comparison of Raw and Smoothened Data')
plt.legend()
plt.show()



