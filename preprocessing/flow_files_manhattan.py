"""
Create 'flow files' for Manhattan. 
Each file contains requests (in the from of (pickup, dest, counter)) of one day, ordered in 60sec flows.
Creator: Victor Prins & Sabijn Perdijk
Created at: jan 2022
"""

import osmnx as ox
import numpy as np
import csv
import datetime
from collections import defaultdict
from get_graph_Manhattan import get_graph

G = get_graph()
nodes = ox.graph_to_gdfs(G, edges=False)
GET_NODEID = dict(zip(list(nodes.index), list(range(nodes.shape[0]))))

def filter_list(list_to_filter, list_with_indices):
    """
    Returns a list with all entries list_to_filter[i] for all i in list_with_indices
    """
    return np.array(list_to_filter)[list_with_indices]

def print_txt(data, day):
    """
    Writes the dictonary of a day to a formatted txt
    """
    output = f'{len(data)}\n'
    for key in data:
        output += f'Flows:{key}-{key}\n'
        for flows in data[key]:
            output += f'{flows[0]},{flows[1]},{float(data[key][flows])}\n'
    f = open(f"out/test_flow_5000_{day}.txt", 'w')
    f.write(output)

    return

def process_requests(start_lngs, start_lats, dest_lngs, dest_lats, flows_list):
    """
    Maps coordinates of all requests of a day to nearest nodes in graph. 
    Omit requests more than 100 m from intersection (node). 
    Divide requets in flows of 60 seconds.
    Return dictionary of one day.
    """
    # Map coordinates to nearest nodes in graph
    start_osmids, start_dists = ox.distance.nearest_nodes(G, start_lngs, start_lats, return_dist=True)
    dest_osmids, dropoff_dists = ox.distance.nearest_nodes(G, dest_lngs, dest_lats, return_dist=True)

    # Omit requests more than 100m from intersection (node)
    filter = np.logical_and(np.array(start_dists) < 100, np.array(dropoff_dists) < 100)

    flows_list = filter_list(flows_list, filter)
    start_osmids = filter_list(start_osmids, filter)
    dest_osmids = filter_list(dest_osmids, filter)

    # Create flows of 60 sec
    data_day = {}
    for j in range(len(flows_list)):
        
        flow_number = flows_list[j]

        if flow_number not in data_day:
            data_day[flow_number] = defaultdict(int)

        # Map osmID to nodeID
        start_id = GET_NODEID[start_osmids[j]]
        dest_id = GET_NODEID[dest_osmids[j]]
        data_day[flow_number][(start_id, dest_id)] += 1
    
    return data_day 

with open('data/yellow_tripdata_2016-miniselection.csv', newline='') as csvfile:
    flow = 0
    # Original paper only uses day 3, 4, 5 for training
    day = 3
    reader = csv.DictReader(csvfile)
    formatstring = '%Y-%m-%d %H:%M:%S'

    # these lists respectively contain for each index i the flow number, the start latitude, the start longitude, the dest latitude and the dest longitude of one request
    flows_list = []
    start_lats = []
    start_lngs = []
    dest_lats = []
    dest_lngs = []

    # Loop through rows csv
    for i, row in enumerate(reader):
        new_time = datetime.datetime.strptime(row['tpep_pickup_datetime'], formatstring)

        # Set begin time
        if i == 0:
            last_time = new_time

        # Start of new day, process requests and restart variables
        if new_time.day != last_time.day:

            # Process the requets and write to txt file
            data_day = process_requests(start_lngs, start_lats, dest_lngs, dest_lats, flows_list)
            print_txt(data_day, day)
            day += 1
            
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

        # Add coordinates of one request to list
        flows_list.append(flow)
        start_lats.append(float(row['pickup_latitude']))
        start_lngs.append(float(row['pickup_longitude']))
        dest_lats.append(float(row['dropoff_latitude']))
        dest_lngs.append(float(row['dropoff_longitude']))

    # Process remaining day
    data_day = process_requests(start_lngs, start_lats, dest_lngs, dest_lats, flows_list)
    print_txt(data_day, day)

