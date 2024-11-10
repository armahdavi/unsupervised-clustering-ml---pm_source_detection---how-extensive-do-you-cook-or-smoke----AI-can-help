# -*- coding: utf-8 -*-
"""
The purpose of this program is to cancel noises from time-series PM2.5 data using two techniques:
    1) Exponentially weighted mean (EMW)
    2) Savitzky-Golay filter

The goal is to get rid of any noise causing huge uncertainties in the value of concentrations and their derivatives over time.
This will help the k-means clustering work better when finding centeroids

@author: alima
"""

## Import essential libraries and program runnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
exec(open('C:/Life/5- Career & Business Development/Learning/Python Practice/Generic Codes/notion_corrections.py').read())

## Reading PM data
df = pd.read_excel(backslash_correct(r'C:\Life\5- Career & Business Development\Learning\Python Practice\Stata_Python_Booster\PhD - QFF\Processed\dt_drx.xlsx'))
df = df[df['visit'] != 5]
df.rename(columns = {'Date': 'Time'}, inplace = True)
df.drop('visit', axis = 1, inplace = True)
df.reset_index(drop = True, inplace = True)

## Making the data smooth by exponentially weighted mean and Savitzky-Golay fitler
df['PM2.5_smooth_ewm'] = df['PM2.5'].ewm(span = 20).mean()
df['PM2.5_smooth_sf'] = scipy.signal.savgol_filter(df['PM2.5'], 50, 3)


df = df[['Time', 'PM1', 'PM10', 'TSP', 'PM2.5', 'PM2.5_smooth_ewm', 'PM2.5_smooth_sf']]

## Sketching data 
for n in range(int(len(df)/1500)):
    ## EMW
    plt.plot(df.loc[n*1500 : (n+1)*1500 + 1,'PM2.5'], label = 'Original Data')
    plt.plot(df.loc[n*1500 : (n+1)*1500 + 1,'PM2.5_smooth_ewm'], label = 'Smoothed Data (EWM)')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    
    ## Savitzky-Golay filter
    plt.plot(df.loc[n*1500 : (n+1)*1500 + 1,'PM2.5'], label = 'Original Data')
    plt.plot(df.loc[n*1500 : (n+1)*1500 + 1,'PM2.5_smooth_sf'], label = 'Smoothed Data SF)')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
   

