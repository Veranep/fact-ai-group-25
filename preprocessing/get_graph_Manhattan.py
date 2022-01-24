import osmnx as ox
import pickle
import networkx.algorithms.components as nx
from shapely.geometry import Polygon
import numpy as np
import os.path

def get_graph():
    file_name = 'Graph-Manhattan.pickle'

    if os.path.isfile(file_name):
        with open(file_name, 'rb') as f:
            G = pickle.load(f)
            return G

    else:

        # contains all the points of the polygon around Manhattan, in (lat, lng) format
        polygon = [
                [40.696366, -74.027805], 
                [40.708942, -73.974939],
                [40.749391, -73.964052],
                [40.779963, -73.938597],
                [40.801256, -73.927287],
                [40.808760, -73.933081],
                [40.834781, -73.933827],
                [40.843523, -73.929610],
                [40.849082, -73.953141],
                [40.758182, -74.015014]
            ]
        # osmnx expects the coordinates in the unconventional (lng,lat) format
        polygon = [(lng, lat) for lat, lng in polygon]

        # fetch the graph from OSM
        G = ox.graph.graph_from_polygon(Polygon(polygon), network_type='drive')

        # add the edge speeds and calculate the travel time between nodes
        G = ox.speed.add_edge_speeds(G, precision=1)
        G = ox.speed.add_edge_travel_times(G, precision=1)

        # only keep the largest strongly connected component
        largest_comp = None
        largest_comp_size = 0

        for comp in nx.strongly_connected_components(G):
            size = len(comp)
            if size > largest_comp_size:
                largest_comp_size = size
                largest_comp = comp

        G = G.subgraph(largest_comp).copy()

        with open(file_name, 'wb') as f:
            pickle.dump(G, f)

        return G

