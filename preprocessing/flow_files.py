import osmnx as ox
import pandas as pd
import csv
from datetime import datetime
import sys
from get_graph_Manhattan import get_graph

# Get graph of streetnetwork Manhattan
G = get_graph()

# Create dataframe of nodes
nodes = ox.graph_to_gdfs(G, edges=False)

# Mapping from osm_id to integer
osmid_to_nodeid = dict(zip(list(nodes.index), list(range(nodes.shape[0]))))

# Formatstring for creating timestamp of date entry
formatstring = '%Y-%m-%d %H:%M:%S'

# Iterating trough dateset
with open('data/yellow_tripdata_2016-miniselection.csv', newline='') as csvfile:
    i = 0
    flow = 0
    day = 3
    # Writing output directly to txt file, comment this out to see stuff in your terminal
    f = open(f"out/test_flow_5000_{day}.txt", 'w')
    sys.stdout = f  
    reader = csv.DictReader(csvfile)
    print('1440')
    print(f'Flows:{flow}-{flow}')

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

            f.close()
            f = open(f"out/test_flow_5000_{day}.txt", 'w')
            sys.stdout = f  
            print('1440')
            print(f'Flows: {flow}-{flow}')
            
        # Start new flow of 60 seconds
        if (new_time - last_time).total_seconds() > 59:
            last_time = new_time
            flow += 1
            print(f'Flows:{flow}-{flow}')
        
        # Retrieve pickup and destination from csv 
        pickup_osmid, dist_pickup = ox.distance.nearest_nodes(G, float(row['pickup_longitude']), float(row['pickup_latitude']), return_dist=True)
        dest_osmid, dist_dropoff = ox.distance.nearest_nodes(G, float(row['dropoff_longitude']), float(row['dropoff_latitude']), return_dist=True)

        # Check if lat-long points are closer than 100 meter from an intersetion, otherwise ignore
        if dist_pickup < 100 and dist_dropoff < 100:
            print(f'{osmid_to_nodeid[pickup_osmid]}, {osmid_to_nodeid[dest_osmid]}, 1.0')

        i += 1

    f.close()
