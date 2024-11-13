# -*- coding: utf-8 -*-
"""
Program to read in PM data recorded by DustTrak DRX and merge with outdoor PM2.5 data

@author: alima
"""

import pandas as pd
exec(open(r':/PhD Research/Generic Codes/notion_corrections.py').read())

### Reading DustTrak PM2.5 data and refining
df = pd.read_excel(backslash_correct(r'C:\PhD Research\Airborne\Processed\dt_drx.xlsx'))
df = df[df['visit'] != 5]
df.rename(columns = {'Date': 'Time'}, inplace = True)
df.drop('visit', axis = 1, inplace = True)
df.reset_index(drop = True, inplace = True)


### Reading previously saved Outdoor PM2.5 data
odpm = pd.read_excel(backslash_correct(r'C:\PhD Research\Airborne\Processed\outdoor_pm25.xlsx'))
df = pd.merge(df, odpm, on = 'Time', how = 'inner')
df.rename(columns = {'PM2.5_x': 'PM2.5', 'PM2.5_y': 'PM2.5_OD'}, inplace = True)

### Re-arranging and saving
df = df[['Time', 'PM1', 'PM10', 'TSP', 'PM2.5', 'PM2.5_OD']]
df.to_excel(backslash_correct(r'C:\PhD Research\Airborne\Processed\all_pm.xlsx'), index = False)
