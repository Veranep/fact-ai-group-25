import osmnx as ox
import random
import networkx as nx
import sys
from get_graph_Manhattan import get_graph

G = get_graph()
nodes = ox.graph_to_gdfs(G, edges=False)

f = open(f"out/taxi_3000_final.txt", 'w')
sys.stdout = f

for _ in range(3000):
    print(random.randint(0, len(nodes)))
f.close()
