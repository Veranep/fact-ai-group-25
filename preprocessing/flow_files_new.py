import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import csv
from datetime import datetime
import sys
import geopy.distance
from collections import defaultdict
from get_graph_Manhattan import get_graph
import time

start_time = time.time()

def print_txt(data):
    print(len(data))
    for key in data:
        print(f'Flows: {key}-{key}')
        for flows in data[key]:
            print(f'{flows[0]}, {flows[1]}, {float(data[key][flows])}')
    return

G = get_graph()
nodes = ox.graph_to_gdfs(G, edges=False)
osmid_to_nodeid = dict(zip(list(nodes.index), list(range(nodes.shape[0]))))
formatstring = '%Y-%m-%d %H:%M:%S'

with open('data/yellow_tripdata_2016-03-04.csv', newline='') as csvfile:
    i = 0
    flow = 0
    day = 1
    # f = open(f"out/test_flow_5000_{day}.txt", 'w')
    # sys.stdout = f  
    reader = csv.DictReader(csvfile)
    print('1440')
    data_day = {flow: defaultdict(int)}

    # Loop through rows csv
    for row in reader:
        new_time = datetime.strptime(row['tpep_pickup_datetime'], formatstring)

        # Set begin time
        if i == 0:
            last_time = new_time
        
        # Start new day file
        if new_time.day != last_time.day:
            day += 1
            flow = 0
            last_time = new_time

            print_txt(data_day)
            # f.close()

            # f = open(f"out/test_flow_5000_{day}.txt", 'w')
            # sys.stdout = f  
            print('1440')
            data_day = {flow: defaultdict(int)}
            
            
        # Start new flow of 60 seconds
        if (new_time - last_time).total_seconds() > 59:
            last_time = new_time
            flow += 1
            data_day[flow] = defaultdict(int)
        
        # Retrieve pickup and destination from csv 
        pickup_osmid, dist_pickup = ox.distance.nearest_nodes(G, float(row['pickup_longitude']), float(row['pickup_latitude']), return_dist=True)
        dest_osmid, dist_dropoff = ox.distance.nearest_nodes(G, float(row['dropoff_longitude']), float(row['dropoff_latitude']), return_dist=True)

        if dist_pickup < 100 and dist_dropoff < 100:
            data_day[flow][(osmid_to_nodeid[pickup_osmid], osmid_to_nodeid[dest_osmid])] += 1

        i += 1
        if i % 1000 == 0:
            print(i)
            print("--- %s seconds ---" % (time.time() - start_time))

    print_txt(data_day)
    # f.close()