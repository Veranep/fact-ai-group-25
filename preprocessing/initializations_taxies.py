import osmnx as ox
import random
import networkx as nx
import sys

G = nx.read_gpickle('manhattan_graph.gpickle')
nodes = ox.graph_to_gdfs(G, edges=False)

f = open(f"taxi_3000_final.txt", 'w')
sys.stdout = f

for _ in range(3000):
    print(random.randint(0, len(nodes)))
f.close()
