import pandas as pd
import numpy as np
import random

routes = pd.read_csv('out_authors/zone_path.csv', header=None).values
travel_times = pd.read_csv('out_authors/zone_traveltime.csv', header=None).values


def get_route(start_id, dest_id):
    route = [start_id]

    k = 0
    while route[-1] != dest_id:
        # append the next node on this route
        route.append(routes[route[-1]][dest_id])


        k += 1
        # infinite route
        if k > 5000:
            return -1

    return route

def get_traveltime(start_id, dest_id):
    return travel_times[start_id, dest_id]

def compute_traveltime(route):
    travel_time = 0
    
    for i in range(1, len(route)):
        edge_tt = get_traveltime(route[i-1], route[i])
        travel_time += edge_tt

    return travel_time

s = set()
N = len(routes)
total = 100000

for i in range(total):
    A = random.randrange(N)
    B = random.randrange(N)

    route = get_route(A,B)
    if route == -1:
        s.add((A,B))



print('Share of faulty routes:', len(s) / total)