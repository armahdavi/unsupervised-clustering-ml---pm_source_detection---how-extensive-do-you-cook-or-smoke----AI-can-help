# -*- coding: utf-8 -*-
"""
Testing LocalOutlierFactor over some randomly generated data

@author: alima
"""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt


# Generate baseline traffic with some randomness
base_traffic = 1000 + 100 * np.random.randn(365)

# Introduce outliers (traffic spikes) at random days
outlier_days = np.random.randint(0, 365, size=10)
outlier_values = np.random.randint(2000, 5000, size=10)

for i, day in enumerate(outlier_days):
  base_traffic[day] = outlier_values[i]

# Simulate weekends with lower traffic
weekend_days = np.arange(7, 365, step=7)
base_traffic[weekend_days] *= 0.7

# Combine baseline and weekend traffic
traffic_data = base_traffic.copy()

# Set the number of neighbors (k)
n_neighbors = 20

# Create LOF object
lof = LocalOutlierFactor(n_neighbors=n_neighbors)

# Fit LOF to the data (reshape for 2D)
lof.fit(traffic_data.reshape(-1, 1))

# Get outlier labels (higher score indicates outlier)
outlier_labels = lof.negative_outlier_factor_

# Threshold for outlier classification (adjust based on data)
threshold = -1.5

# Identify outliers based on threshold
outlier_indices = traffic_data[outlier_labels < threshold].tolist()

print("Outlier traffic days:", outlier_indices)

#############################################

# Generate baseline traffic with some randomness
base_traffic = 1000 + 100 * np.random.randn(365)

# Introduce outliers (traffic spikes) at random days
outlier_days = np.random.randint(0, 365, size=10)
outlier_values = np.random.randint(2000, 5000, size=10)

for i, day in enumerate(outlier_days):
    base_traffic[day] = outlier_values[i]

# Simulate weekends with lower traffic
weekend_days = np.arange(7, 365, step=7)
base_traffic[weekend_days] *= 0.7

# Combine baseline and weekend traffic
traffic_data = base_traffic.copy()

# LOF Anomaly Detection
from sklearn.neighbors import LocalOutlierFactor

# Set the number of neighbors (k)
n_neighbors = 20

# Create LOF object
lof = LocalOutlierFactor(n_neighbors=n_neighbors)

# Fit LOF to the data (reshape for 2D)
lof.fit(traffic_data.reshape(-1, 1))

# Get outlier labels (higher score indicates outlier)
outlier_labels = lof.negative_outlier_factor_

# Threshold for outlier classification (adjust based on data)
threshold = -1.5

# Identify outliers based on threshold
outlier_indices = traffic_data[outlier_labels < threshold].tolist()


# Plot raw and processed data
days = np.arange(365)

plt.figure(figsize=(12, 6))

# Plot raw traffic data
plt.plot(days, traffic_data, marker='o', linestyle='-', label='Raw Traffic')

# Plot outliers with different marker
outlier_days = [day for day, label in zip(days, outlier_labels) if label < threshold]
plt.scatter(outlier_days, traffic_data[outlier_days], marker='x', color='red', label='Outliers', s=250)  # set marker size to 100

plt.xlabel('Days')
plt.ylabel('Website Traffic')
title = 'Website Traffic with LOF Anomaly Detection'
plt.title(title)
plt.legend()
plt.grid(True)
plt.show()
