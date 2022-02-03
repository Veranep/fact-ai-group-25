"""
Select data from the green taxi dataset based on chosen timeframe and area.
Raw data available at https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
Created by: Sabijn Perdijk 
Created at: jan 2022
"""

from datetime import datetime
import pandas as pd

def data_selection(df, min_latitude, max_latitude, min_longitude, max_longitude):
    """
    Select data occuring in specific area
    df: raw data
    min_latitude: value of smallest latitude
    max_latitude: value of highest latitude
    min_longitude: value of smallest longitude
    max_longitude: value of highest longitude
    """
    df = df[df['Pickup_latitude'] > min_latitude]
    df = df[df['Pickup_latitude'] < max_latitude]
    df = df[df['Pickup_longitude'] > min_longitude]
    df = df[df['Pickup_longitude'] < max_longitude]

    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
    df = df.sort_values(by="lpep_pickup_datetime")

    return df

# Lat and long coordinates of polygon defining area. Coordinates based on map NY.
longitude = [-73.962177,
            -73.913000,
            -73.917864,
            -73.897018,
            -73.956026,
            -74.034150,
            -73.998521,
            -73.973849]

lats =  [40.749373, 
        40.714213,
        40.710215, 
        40.677902, 
        40.644944, 
        40.674024, 
        40.705150, 
        40.708715]
min_latitude = min(lats)
max_latitude = max(lats)

min_longitude = min(longitude)
max_longitude = max(longitude)

# Loop through data of februari, march, and june. Select chosen area and concatenate files.
frames = []
for month in [2, 3, 6]:
    data = pd.read_csv(f'data/green_tripdata_2016-0{month}.csv')
    data = data_selection(data, min_latitude, max_latitude, min_longitude, max_longitude)
    frames.append(data)

result = pd.concat(frames)

# Save resulting dataframe
result.to_csv('data/green_tripdata_2016_modified.csv')