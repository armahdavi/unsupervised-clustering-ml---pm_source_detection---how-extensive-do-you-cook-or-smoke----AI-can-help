# -*- coding: utf-8 -*-

"""
Program to run feature engineering for k-means clustring. These features are assumed relevant:
    1) Indoor PM2.5 concentration after Savitzky-Golay filter (Step 1)
    2) Oputdoor PM2.5 concentration (Step 1)
    3) PM2.5 concentration indoor to outdoor ratio  (Step 1)
    4) Wind speed and direction (as they may detemine the AER) (Step 2)
    5) Proximity to the nearest peak value (calculated after nearest peak value algorithm) (Step 3)
    6) Proximity to baseline concentration  (Step 4)
    7) HVAC runtime (and loss) (step 5)
    

@author: alima
"""

### Importing essential modules and runnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime # Meteostat module to calculate wind direction and speed data
from meteostat import Hourly
from scipy.signal import savgol_filter, find_peaks
from sklearn.neighbors import LocalOutlierFactor
exec(open('C:/Life/5- Career & Business Development/Learning/Python Practice/Generic Codes/notion_corrections.py').read())

#######################################
### Step 1: Initial Data Extraction ###
#######################################

## Reading PM initial data from DustTrack
df = pd.read_excel(backslash_correct(r'C:\Life\5- Career & Business Development\Learning\Python Practice\Stata_Python_Booster\PhD - QFF\Processed\dt_drx.xlsx'))
df = df[df['visit'] != 5]
df.rename(columns = {'Date': 'Time'}, inplace = True)
df.drop('visit', axis = 1, inplace = True)

## Reading Outdoor PM data
odpm = pd.read_excel(backslash_correct(r'C:\Life\5- Career & Business Development\Learning\Python Practice\Stata_Python_Booster\PhD - QFF\Processed\outdoor_pm25.xlsx'))
df = pd.merge(df, odpm, on = 'Time', how = 'inner')
df.rename(columns = {'PM2.5_x': 'PM2.5', 'PM2.5_y': 'PM2.5_OD'}, inplace = True)

## Keeping relevant columns
df = df[['Time', 'PM1', 'PM10', 'TSP', 'PM2.5', 'PM2.5_OD']]

## I/O ratio
df['PM2.5_smooth'] = savgol_filter(df['PM2.5'], 50, 3)
df.loc[df['PM2.5_smooth'] < 0, 'PM2.5_smooth'] = 0.5
df['I/O'] = df['PM2.5_smooth']/df['PM2.5_OD']

## Sketching the original vs. smoothened data
for n in range(int(len(df)/1500)):
    plt.plot(df.loc[n*1500 : (n+1)*1500 + 1,'PM2.5'], label = r'Original $\mathrm{PM_{2.5}}$')
    plt.plot(df.loc[n*1500 : (n+1)*1500 + 1,'PM2.5_smooth'], label = r'Smoothed $\mathrm{PM_{2.5}}$')
    plt.plot(df.loc[n*1500 : (n+1)*1500 + 1,'PM2.5_OD'], label = 'Outdoor PM2.5')
    plt.ylim(0, 140) 
    plt.yticks(range(0, 141, 20))
    plt.xlabel('Time (Min) From the Start of the Study')
    plt.ylabel(r'$\mathrm{PM_{2.5}}$ (µg/m³)')
    legend = plt.legend(loc = "upper center", bbox_to_anchor = (0.5, 1.15), ncol = 3)
    legend.get_frame().set_edgecolor("black")  
    plt.show()


### Step 2: Wind Data Calculation
station_id = '72219'
start_date = datetime(year = 2018, month = 7, day = 18)
end_date = datetime(year = 2018, month = 8, day = 29)

## Extracting data from meteostats
data = Hourly(station_id, start_date, end_date)
data = data.fetch()

wind = data[['wdir', 'wspd']]
wind.reset_index(inplace = True)
wind.rename(columns = {'time':'Time'}, inplace = True)

## Merging with main dataframe
df = pd.merge(df, wind, on ='Time', how = 'outer')
df.dropna(subset = ['PM2.5'], inplace = True)
df.reset_index(drop = True, inplace = True)

## Inteprpolating to fill missing data
df['wdir'] = df['wdir'].interpolate(method = 'nearest')
df['wspd'] = df['wspd'].interpolate(method = 'nearest')
df.loc[:len(df)/2,'wdir'].fillna(method = 'bfill', inplace = True)
df.loc[len(df)/2:,'wdir'].fillna(method = 'ffill', inplace = True)
df.loc[:len(df)/2,'wspd'].fillna(method = 'bfill', inplace = True)
df.loc[len(df)/2:,'wspd'].fillna(method = 'ffill', inplace = True)


#############################################################################
### Step 3: Outlier/peak detection and nearest peak concentration locator ###
#############################################################################

smoothed_data = df['PM2.5_smooth'].values 
threshold_value = np.mean(smoothed_data) + 1 * np.std(smoothed_data)

'''
## finding the best min distance, prominence, and width with the data
tuple_dict = {}

for i in np.arange(1, 50, 1):
    for j in np.arange(0, 50, 1):
        for k in np.arange(0, 5, 0.5):
            minimum_distance = i  # Adjust based on time-series frequency and expected peak spacing
            minimum_prominence = j  # Adjust as needed for true vs. noise peaks
            minimum_width = k  # Based on expected peak duration
            
            # Find peaks in the smoothed data
            peaks, properties = find_peaks(
                smoothed_data,
                height = threshold_value,
                distance = minimum_distance,
                prominence = minimum_prominence,
                width = minimum_width
            )
            
            tuple_dict[(i,j,k)] = len(peaks)

minimum_distance, minimum_prominence, minimum_width = max(tuple_dict, key = tuple_dict.get)

minimum_distance = minimum_distance  
minimum_prominence = minimum_prominence 
minimum_width = minimum_width  
'''

minimum_distance = 10  
minimum_prominence = 1.5 
minimum_width = 5 

## Finding peaks in the smoothed data
peaks, properties = find_peaks(
    smoothed_data,
    height = threshold_value,
    distance = minimum_distance,
    prominence = minimum_prominence,
    width = minimum_width
)

peak_indices = peaks
peak_values =  smoothed_data[peaks]
  
## Initializing an array with NaN to store the nearest peak values
extended_peaks_nearest = np.full(len(smoothed_data), np.nan)

## Handling the section before the first peak
extended_peaks_nearest[:peak_indices[0]] = peak_values[0]

# Loop over each pair of consecutive peaks
for i in range(1, len(peak_indices)):
    # Calculate the midpoint between the current peak and the next peak
    midpoint = (peak_indices[i-1] + peak_indices[i]) // 2
    
    # Assign values from the previous peak to the midpoint as the previous peak's value
    extended_peaks_nearest[peak_indices[i-1]:midpoint] = peak_values[i-1]
    
    # Assign values from the midpoint to the current peak as the current peak's value
    extended_peaks_nearest[midpoint:peak_indices[i]] = peak_values[i]

# Handling the section after the last peak
extended_peaks_nearest[peak_indices[-1]:] = peak_values[-1]

# Adding the nearest extended peaks as a new column in the main DataFrame
df['Nearest Peak'] = extended_peaks_nearest


## Plotting PM2.5 (indoor and outdoor) and nearest peak for windows of 3000 minutes
for n in range(int(len(df)/3000)):
    df_select = df.iloc[n*3000: (n+1)*3000, :]

    # Plotting the main data and outliers
    plt.plot(df_select.index, df_select['PM2.5_smooth'], label = 'Main Data')
    plt.plot(df_select.index, df_select['PM2.5_OD'], label = 'Outdoor PM2.5')
    plt.plot(df_select.index, df_select['Nearest Peak'], label = 'Nearest Peak', c = 'r')
    
    plt.ylim(0, 100) 
    plt.yticks(range(0, 101, 20))
    
    plt.xlabel('Time (Min) From the Start of the Study')
    plt.ylabel(r'$\mathrm{PM_{2.5}}$ (µg/m³)')
    legend = plt.legend(loc = "upper center", bbox_to_anchor = (0.5, 1.15), ncol = 3)
    legend.get_frame().set_edgecolor("black")  
    plt.legend()
    plt.show()


## Plotting PM2.5 (indoor and outdoor) and nearest peak for the entire time-series
plt.plot(df.index, df['PM2.5_smooth'], label = 'Main Data')
plt.plot(df.index, df['PM2.5_OD'], label = 'Outdoor PM2.5')
plt.plot(df.index, df['Nearest Peak'], label = 'Nearest Peak', c = 'r')
plt.ylim(0, 100) 
plt.yticks(range(0, 101, 20))
plt.xlabel('Time (Min) From the Start of the Study')
plt.ylabel(r'$\mathrm{PM_{2.5}}$ (µg/m³)')
legend = plt.legend(loc = "upper center", bbox_to_anchor = (0.5, 1.15), ncol = 3)
legend.get_frame().set_edgecolor("black")  
plt.show()

## Calculating the nearest concentration peak proximity
df['Nearest peak Proximity'] = df['PM2.5_smooth']/df['Nearest Peak']
df['Der C'] = df['PM2.5_smooth'].diff()


############################################################################
### Step 4: Baseline concentration calculation and proximity to baseline ###
############################################################################

## Fitting the model and get outlier predictions and scores
lof = LocalOutlierFactor(n_neighbors = 5, contamination = 0.1)

data_reshaped =  np.array(df['PM2.5_smooth']).reshape(-1, 1)
outlier_labels = lof.fit_predict(data_reshaped)
lof_scores = lof.negative_outlier_factor_
upper_threshold = np.percentile(df['PM2.5_smooth'], 99)
df['Outlier'] = np.where(df['PM2.5_smooth'] >=  upper_threshold, df['PM2.5_smooth'], np.nan)

## Baseline calculation
baseline = df[df['Outlier'].isna()]['PM2.5_smooth'].median()
df['Proximity to Baseline'] = abs(df['PM2.5_smooth'] - baseline)

## Sketching for windows of 300 minutes
for n in range(int(len(df)/3000)):
    df_select = df.iloc[n*3000: (n+1)*3000, :]
    
    # Plot the original data
    plt.plot(df_select.index, df_select['PM2.5_smooth'], label = r'$\mathrm{PM_{2.5}}$', color = 'b')
    
    # Highlight outliers in red
    plt.scatter(df_select.index, df_select['Outlier'],
                color = 'r', label = 'Outliers', marker = 'o', s = 20) 
    
    plt.ylim(0, 100) 
    plt.yticks(range(0, 101, 20))
    
    plt.xlabel('Time (Min) From the Start of the Study')
    plt.ylabel(r'$\mathrm{PM_{2.5}}$ (µg/m³)')
    legend = plt.legend(loc = "upper center", bbox_to_anchor = (0.5, 1.15), ncol = 2)
    legend.get_frame().set_edgecolor("black")  
    plt.legend()
    plt.show()
    

## Sketching for the entire duration
plt.plot(df.index, df['PM2.5_smooth'], label = r'$\mathrm{PM_{2.5}}$', color = 'b')
plt.scatter(df.index, df['Outlier'],
            color = 'r', label = 'Outliers', marker = 'o', s = 20) 

plt.ylim(0, 100) 
plt.yticks(range(0, 101, 20))
plt.xlabel('Time (Min) From the Start of the Study')
plt.ylabel(r'$\mathrm{PM_{2.5}}$ (µg/m³)')
legend = plt.legend(loc = "upper center", bbox_to_anchor = (0.5, 1.15), ncol = 2)
legend.get_frame().set_edgecolor("black")  
plt.legend()
plt.show()

##########################################
### Step 5: HVAC runtime and HVAC loss ###
##########################################

rt = pd.read_excel(backslash_correct(r'C:\Life\5- Career & Business Development\Learning\Python Practice\Stata_Python_Booster\PhD - QFF\Processed\runtime_master.xlsx'))
rt = rt[['Time', 'Mode']]
rt['Mode'].replace({'Off': 0,
                    'Transient': 1,
                    'Fan Only': 2,
                    'Compressor': 3}, inplace = True)

df = pd.merge(df, rt, on = 'Time', how = 'left')
df.to_excel(backslash_correct(r'C:\Life\5- Career & Business Development\Learning\Python Practice\Stata_Python_Booster\PhD - QFF\Processed\pm_source_clustering_ml_master.xlsx'), index = False)

