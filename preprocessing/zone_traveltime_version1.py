import osmnx as ox
import networkx as nx
import pandas as pd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

G = nx.read_gpickle('Graph-Manhattan.pickle')

# Create geodataframes with the nodes and the edges
nodes = ox.graph_to_gdfs(G, edges=False)
edges = ox.graph_to_gdfs(G, nodes=False)

# create zone_traveltime
nodes_id = list(nodes.index)
csv_path = pd.read_csv('zone_path.csv', header=None)

for start_node_id in nodes_id:
    for dest_node_id in nodes_id:
        edge = G.get_edge_data(start_node_id, dest_node_id)
        travel_time = edge[0]['travel_time']
        csv_path[start_node_id][dest_node_id] = travel_time