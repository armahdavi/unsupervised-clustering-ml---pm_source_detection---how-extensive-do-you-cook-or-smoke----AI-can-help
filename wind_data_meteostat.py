# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 22:19:47 2024

@author: alima
"""

from datetime import datetime
from meteostat import Hourly
import pandas as pd
exec(open('C:\PhD Research\Generic Codes\notion_corrections.py').read())
exec(open('C:\PhD Research\Airborne\Code\source_clustering\indoor_pm_data_read_in.py').read())


### Reading climate data for wind data access
station_id = '72219' # Toronto Station ID

## Defining start and end date and reading data
start_date = datetime(year = 2018, month = 7, day = 18)
end_date = datetime(year = 2018, month = 8, day = 29)
data = Hourly(station_id, start_date, end_date)
data = data.fetch() # Fetch data from Meteostat API

## Obtaining wind direction (wdir) and wind speed (wspd) columns
wind = data[['wdir', 'wspd']]

## Re-arraning and merging with PM data
wind.reset_index(inplace = True)
wind.rename(columns = {'time':'Time'}, inplace = True)
df = pd.merge(df, wind, on ='Time', how = 'outer')
df.dropna(subset = ['PM2.5'], inplace = True)

## Filling the missing data for the beginning and end of timeframes
df['wdir'] = df['wdir'].interpolate(method='nearest')
df['wspd'] = df['wspd'].interpolate(method='nearest')

## Filling minute-wise data by the prior hourly data
df.loc[:len(df)/2,'wdir'].fillna(method = 'bfill', inplace = True)
df.loc[len(df)/2:,'wdir'].fillna(method = 'ffill', inplace = True)

df.loc[:len(df)/2,'wspd'].fillna(method = 'bfill', inplace = True)
df.loc[len(df)/2:,'wspd'].fillna(method = 'ffill', inplace = True)
