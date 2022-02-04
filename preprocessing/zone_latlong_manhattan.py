"""
Create csv file which maps coordinates to nodeID's
"""

import osmnx as ox
import pandas as pd
from get_graph_Manhattan import get_graph

# Get graph of Manhattan
G = get_graph()

# Create geodataframes with the nodes and the edges
nodes = ox.graph_to_gdfs(G, edges=False)

# Create columns: A are nodeID's, B are latitudes, C are longitudes
data = {'A': list(range(nodes.shape[0])), 'B': list(nodes['x']), 'C': list(nodes['y'])}
zone_path = pd.DataFrame(data)
zone_path.to_csv('out/zone_latlon.csv', header=False, index=False)
