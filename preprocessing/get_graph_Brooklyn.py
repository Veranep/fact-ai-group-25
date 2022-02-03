import osmnx as ox
import pickle
import networkx.algorithms.components as nx
from shapely.geometry import Polygon
import numpy as np
import os.path

def get_graph():
    file_name = 'out/Graph-Brooklyn.pickle'

    if os.path.isfile(file_name):
        with open(file_name, 'rb') as f:
            G = pickle.load(f)
        
        return G

    else:

        # contains all the points of the polygon around Brooklyn, in (lat, lng) format
        polygon = [
                [40.749373, -73.962177],
                [40.714213, -73.913000],
                [40.710215, -73.917864],
                [40.677902, -73.897018],
                [40.644944, -73.956026],
                [40.674024, -74.034150],
                [40.705150, -73.998521],
                [40.708715, -73.973849]
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

