# -*- coding: utf-8 -*-
"""
Program to read in and organize urban PM2.5 concentrations in Toronto

@author: alima
"""

import pandas as pd
exec(open('C:/Life/5- Career & Business Development/Learning/Python Practice/Generic Codes/notion_corrections.py').read())


### Reading PM2.5 data, reframing, and refining (downloaded from 'https://www.airqualityontario.com/history/pollutant.php?stationid=31103&pol_code=124' website)
df = pd.read_csv(backslash_correct(r'C:\Life\5- Career & Business Development\Learning\Python Practice\Stata_Python_Booster\PhD - QFF\Raw\pm25.csv'), skiprows = 10)
df = df[['Pollutant', 'Date', 'H01', 'H02', 'H03', 'H04', 'H05',
         'H06', 'H07', 'H08', 'H09', 'H10', 'H11', 'H12', 'H13', 'H14', 'H15',
         'H16', 'H17', 'H18', 'H19', 'H20', 'H21', 'H22', 'H23', 'H24']]

df = df.iloc[:44,:]

series_all = pd.Series([])

for i in range(len(df)):
    series_all = pd.concat([series_all, df.iloc[i,1:-1]])


### Getting hourly data from the project's start date
start_date = '2018-07-18 00:00'
num_hours = len(series_all)  # Number of hours
timestamps = pd.date_range(start=start_date, periods=num_hours, freq='H')

timestamps = pd.Series(pd.to_datetime(timestamps))
series_all.reset_index(inplace=True, drop = True)

df = pd.concat([timestamps, series_all], axis = 1)
df.columns = ['Time', 'PM2.5']


### Resampling to fill with hourly recorded to data to make min-by-min data
df.set_index('Time', inplace=True)
df_resampled = df.resample('1T').fillna(method = 'nearest')
df_resampled = df_resampled.reset_index()
df_resampled.to_excel(backslash_correct(r'C:\Life\5- Career & Business Development\Learning\Python Practice\Stata_Python_Booster\PhD - QFF\Processed\outdoor_pm25.xlsx'), index = False)
