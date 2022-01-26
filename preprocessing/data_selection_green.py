# import osmnx as ox
# import networkx as nx
# import pandas as pd
# import numpy as np
# import csv
from datetime import datetime
# import sys
# import geopy.distance
# from get_graph_Manhattan import get_graph
# import time

import pandas as pd
# from shapely.geometry import Point
# from shapely.geometry.polygon import Polygon

def data_selection(df, min_latitude, max_latitude, min_longitude, max_longitude):
    df = df[df['Pickup_latitude'] > min_latitude]
    df = df[df['Pickup_latitude'] < max_latitude]
    df = df[df['Pickup_longitude'] > min_longitude]
    df = df[df['Pickup_longitude'] < max_longitude]

    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
    df = df.sort_values(by="lpep_pickup_datetime")

    return df

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

feb = pd.read_csv('data/green_tripdata_2016-02.csv')
feb = data_selection(feb, min_latitude, max_latitude, min_longitude, max_longitude)

maart = pd.read_csv('data/green_tripdata_2016-03.csv')
maart = data_selection(maart, min_latitude, max_latitude, min_longitude, max_longitude)

juni = pd.read_csv('data/green_tripdata_2016-06.csv')
juni = data_selection(juni, min_latitude, max_latitude, min_longitude, max_longitude)
frames = [feb, maart, juni]
result = pd.concat(frames)
result.to_csv('data/green_tripdata_2016-236.csv')