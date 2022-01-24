import osmnx as ox
import networkx as nx
import pandas as pd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np

G = nx.read_gpickle('manhattan_graph.gpickle')

# Create geodataframes with the nodes and the edges
