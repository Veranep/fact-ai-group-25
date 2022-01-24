"""
File to create drive graph
Base code credits: dimitris - https://gist.github.com/dimichai/2e17c7fd06c40459eb8dabae8283f074 
"""

import osmnx as ox
import networkx as nx
import pandas as pd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

def get_graph(city, plot):        
    # get the boundaries of Manhattan
    gdf = ox.geocode_to_gdf(city)

    # create the drive graph for the specified bounds
    G = ox.graph_from_bbox(
                    gdf.loc[0, 'bbox_north'],
                    gdf.loc[0, 'bbox_south'],
                    gdf.loc[0, 'bbox_east'],
                    gdf.loc[0, 'bbox_west'],
                    'drive')

    polygon = Polygon([
        [-73.925581, 40.877629], 
        [-73.910631, 40.871206],
        [-73.934415, 40.835740],
        [-73.915048, 40.794854],
        [-73.997688, 40.672853],
        [-73.977906, 40.710946],
        [-74.030911, 40.707082]
    ])

    G = ox.truncate.truncate_graph_polygon(G, polygon)

    # gets the largest connected component
    # this ensures that there is a way to get from every point A to every point B in the network
    G = ox.utils_graph.get_largest_component(G, strongly=True)

    # add the edge speeds and calculate the travel time between nodes
    G = ox.speed.add_edge_speeds(G, precision=1)
    G = ox.speed.add_edge_travel_times(G, precision=1)

    # Create geodataframes with the nodes and the edges
    nodes = ox.graph_to_gdfs(G, edges=False)
    edges = ox.graph_to_gdfs(G, nodes=False)
    # print(edges['osmid'])
    # print(edges)
    # print(f'Number of nodes: {nodes.shape[0]}')
    # print(f'Number of edges: {edges.shape[0]}')

    if plot:
        # create a figure of the network
        fig, ax = ox.plot_graph(G, node_size=3)
        fig.savefig('./test_drive_network.png')
    
    return G

G = get_graph('Manhattan', False)
nx.write_gpickle(G, 'manhattan_graph.gpickle')
