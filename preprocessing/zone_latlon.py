import osmnx as ox
import networkx as nx
import pandas as pd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np

G = nx.read_gpickle('manhattan_graph.gpickle')

# Create geodataframes with the nodes and the edges
nodes = ox.graph_to_gdfs(G, edges=False)
edges = ox.graph_to_gdfs(G, nodes=False)
print(nodes['x'].iloc[0])
print(nodes['y'].iloc[0])

data = {'A': list(range(nodes.shape[0])), 'B': list(nodes['x']), 'C': list(nodes['y'])}
zone_path = pd.DataFrame(data)
zone_path.to_csv('zone_latlon_version1.csv')