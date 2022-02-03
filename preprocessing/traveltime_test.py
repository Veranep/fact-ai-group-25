import pandas as pd
import numpy as np
import random

# This file samples routes and checks if the travel time of a route is equal to the sum of the travel times of all
# edges of the route. If this is not the case, this traveltime file will make the route planner (i.e. the main script of the paper) crash

routes = pd.read_csv('out_Brooklyn/zone_path_SERVER.csv', header=None).values
travel_times = pd.read_csv('out_Brooklyn/zone_traveltime_SERVER.csv', header=None).values


def get_route(start_id, dest_id):
    route = [start_id]

    while route[-1] != dest_id:
        # append the next node on this route
        route.append(routes[route[-1]][dest_id])

    return route

def get_traveltime(start_id, dest_id):
    return travel_times[start_id, dest_id]

def compute_traveltime(route):
    travel_time = 0
    
    for i in range(1, len(route)):
        edge_tt = get_traveltime(route[i-1], route[i])
        travel_time += edge_tt

    return travel_time

N = len(routes)
not_equal = 0 
total = 10000

for i in range(total):
    A = random.randrange(N)
    B = random.randrange(N)

    route = get_route(A,B)
    tt_file = get_traveltime(A,B)
    tt_comp = compute_traveltime(route)

    if tt_file != tt_comp:
        not_equal += 1


print(not_equal/total)