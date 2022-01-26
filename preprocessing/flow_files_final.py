import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import csv
import datetime
import sys
from collections import defaultdict
from get_graph_Manhattan import get_graph
import time

def filter_list(list_to_filter, list_with_indices):
    """
    Returns a list with all entries list_to_filter[i] for all i in list_with_indices
    """
    return np.array(list_to_filter)[list_with_indices]


def print_txt(data, day):
    f = open(f"out/test_flow_5000_{day}.txt", 'w')
    sys.stdout = f

    # start the file with the number of flows
    print(len(data))

    for flow_number in data:
        print(f'Flows:{flow_number}-{flow_number}')

        for start_dest_pair, trip_count in data[flow_number].items():
            start_id = start_dest_pair[0]
            dest_id = start_dest_pair[1]
            print(f'{start_id}, {dest_id}, {float(trip_count)}')

    f.close()
    return

G = get_graph()
nodes = ox.graph_to_gdfs(G, edges=False)
osmid_to_nodeid = dict(zip(list(nodes.index), list(range(nodes.shape[0]))))
formatstring = '%Y-%m-%d %H:%M:%S'

with open('data/yellow_tripdata_2016-miniselection.csv', newline='') as csvfile:
    i = 0
    flow = 0
    day = 3
    amount = 0
    reader = csv.DictReader(csvfile)
    data_day = {flow: defaultdict(int)}
    start_time = time.time()

    # these lists respectively contain for each index i the flow number, the start latitude, the start longitude, the dest latitude and the dest longitude of one request
    flows_list = []
    start_lats = []
    start_lngs = []
    dest_lats = []
    dest_lngs = []

    # Loop through rows csv
    for row in reader:
        new_time = datetime.datetime.strptime(row['tpep_pickup_datetime'], formatstring)

        # Set begin time
        if i == 0:
            last_time = new_time

        # Start new day file
        if new_time.day != last_time.day:
            

            # Retrieve pickup and destination from csv
            start_osmids, start_dists = ox.distance.nearest_nodes(G, start_lngs, start_lats, return_dist=True)
            dest_osmids, dropoff_dists = ox.distance.nearest_nodes(G, dest_lngs, dest_lats, return_dist=True)

            filter = np.logical_and(np.array(start_dists) < 100, np.array(dropoff_dists) < 100)

            flows_list = filter_list(flows_list, filter)
            start_osmids = filter_list(start_osmids, filter)
            dest_osmids = filter_list(dest_osmids, filter)

            data_day = {}

            for j in range(len(flows_list)):
                
                flow_number = flows_list[j]

                if flow_number not in data_day:
                    data_day[flow_number] = defaultdict(int)

                start_id = osmid_to_nodeid[start_osmids[j]]
                dest_id = osmid_to_nodeid[dest_osmids[j]]
                data_day[flow_number][(start_id, dest_id)] += 1

            # should the day be incremented before printing?
            day += 1
            print_txt(data_day, day)
            
            # reset variables for next day
            flow = 0
            last_time = new_time
            data_day = {flow: defaultdict(int)}
            flows_list = []
            start_lats = []
            start_lngs = []
            dest_lats = []
            dest_lngs = []

        # Start new flow of 60 seconds
        if (new_time - last_time).total_seconds() > 59:
            last_time += datetime.timedelta(0,60)
            flow += 1

        flows_list.append(flow)
        start_lats.append(float(row['pickup_latitude']))
        start_lngs.append(float(row['pickup_longitude']))
        dest_lats.append(float(row['dropoff_latitude']))
        dest_lngs.append(float(row['dropoff_longitude']))

        i += 1

    print_txt(data_day, day)
