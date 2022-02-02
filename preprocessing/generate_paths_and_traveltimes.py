# from get_graph_Manhattan import get_graph
from get_graph_Brooklyn import get_graph
from routing_functions import *
import numpy as np


# this if __name__ line is necessary for the multiprocessing to work
if __name__ == "__main__":

    # get the strongly connected graph of Manhattan/Brooklyn
    G = get_graph()

    csv_dict = generate_csv_dict(G)

    pairs_list, remaining_pairs_set = get_all_pairs(G)

    # STRATEGY 0: assign the route from a node to itself as itself
    for node_id in G.nodes:
        csv_dict[node_id][node_id] = node_id

    # STRATEGY 1: process all short routes around all nodes. That is,
    # compute all routes between pairs (n,m) where n is a predecessor of m

    # this will contain all the pairs (n,m) as described above
    predecessor_pairs = get_all_predecessor_pairs(G)
        
    print('Strategy 1 started')

    while len(predecessor_pairs) > 0:
        # for shorter routes, using greater batches is much faster
        compute_routes_for_batch(100000, predecessor_pairs, remaining_pairs_set, csv_dict, G)

        # save progress
        save_var(csv_dict, 'csv_strategy_1.pickle')

    print('Strategy 1 completed')

    # STRATEGY 2: randomly sample (n,m) pairs of nodes and compute the route between them

    print('Strategy 2 started')

    # after 25 iterations, the progress becomes very slow and there is enough coverage to transition to strategy 3
    for i in range(25):
        # because these routes are between random pairs, they tend to be longer so we reduce the batch size to 20000
        compute_routes_for_batch(20000, pairs_list, remaining_pairs_set, csv_dict, G)

        # save progress
        save_var(csv_dict, 'csv_strategy_2.pickle')

    print('Strategy 2 completed')


    # STRATEGY 3: Finish all routes using the following estimation technique: Take any pair (n,m) between which no 
    # route is presently known. Find all predecessors p of m up to the 4th degree. It is highly likely that there exists 
    # a known route (n,p) for at least some p, because strategy 2 has randomly covered the whole graph.
    # All routes (p,m) should be known, because they have been covered by strategy 1.
    # For all predecessors p for which a route (n,p) is known, we compute the total travel time (n,p) + (p,m). We then select
    # the single predecessor p for which this travel time is shortest and set the route (n,m) equal to the route (n,p) followed
    # by the route (p,m).

    print('Strategy 3 started')

    i = 0
    while len(remaining_pairs_set) > 0:
        
        # Get the next pair. We directly pop from the set of remaining pairs (rather than the 
        # randomised list) because we don't care about the random order anymore since we just need
        # to process every remaining pair and random order would provide little efficiency gain.
        start_id, dest_id = remaining_pairs_set.pop()

        # this set will contain the ids of all fourth degree predecessors of the destination
        predecessors = set()
        add_predecessors(dest_id, predecessors, G, 1, 4)

        # find the predecessor p such that start_id -> p -> dest_id has the lowest total travel time
        min_travel_time = 1e6
        best_pred_id = None

        for pred_id in predecessors:
            # compute the travel times (start,p) and (p,dest)
            A = compute_travel_time(start_id, pred_id, csv_dict, G)
            B = compute_travel_time(pred_id, dest_id, csv_dict, G)

            # if compute_travel_time returns -1, then no route could be found
            if A != -1 and B != -1:
                travel_time = A + B

                if travel_time < min_travel_time:
                    min_travel_time = travel_time
                    best_pred_id = pred_id

        # if best_pred_id is None, then no route was found. 
        # This is unlikely but may sometimes happen. We will cover these routes in strategy 4
        if best_pred_id is not None:
            # Get the routes (start,p) and (p,dest) 
            A = get_route(start_id, best_pred_id, csv_dict) 
            B = get_route(best_pred_id, dest_id, csv_dict)

            # concatenate the routes but remove the first element from B to prevent
            # a double entry of the best_pred_id node
            route = A + B[1:]

            update_shallow(route, csv_dict, remaining_pairs_set)

        i += 1
        if i % 20000 == 0:
            print_progress(remaining_pairs_set, G)
        
        if i % 100000 == 0:
            save_var(csv_dict, 'csv_strategy_3.pickle')

    save_var(csv_dict, 'csv_strategy_3.pickle')
    print('Strategy 3 completed')


    # STRATEGY 4: There may still be a handful (<100k) of uncomputed routes, for which strategy 3
    # didn't find a route. These routes we compute using Dijkstra (like in strategy 1 and 2), 
    # after which the csv is guaranteed to be complete

    print('Strategy 4 started')

    # get all remaining pairs by looping through the csv
    remaining_pairs_list = get_all_remaining_pairs(csv_dict)
    # copy it into a set. This is actually unnecessary but compute_routes_for_batch() expects
    # a list and a set, so to keep the code simple we just create the set by making a copy
    remaining_pairs_set = set(remaining_pairs_list)

    print('Number of uncomputed routes = ', len(remaining_pairs_list))

    while len(remaining_pairs_set) > 0:
        # Compute routes and update shallow (rather than recursive), because the vast majority of subroutes is already computed
        compute_routes_for_batch(20000, remaining_pairs_list, remaining_pairs_set, csv_dict, G, recursive_update=False)

        # save progress
        save_var(csv_dict, 'csv_strategy_4.pickle')

    print('Strategy 4 completed')

    # Print pairs remaing. If this is not 0, then there is a bug.
    print('Pairs remaining (must be 0!) = ', len(get_all_remaining_pairs(csv_dict)))

    # Save completed csv dict
    save_var(csv_dict, 'csv_final.pickle')

    print('Started generating paths and traveltime CSV files. This may take up to 20 minutes.')

    # LAST STEP: Save actual CSV file
    # and substitute node_ids by their index in list(G.nodes), such that they run from 0 to len(G.nodes)-1

    # get a dict that maps the node_id to its index in list(G.nodes)
    id_to_idx = nodeid_to_index(G)

    # we convert the dict to a np array (discarding all dict keys) for easy printing to CSV
    csv_paths_np = np.zeros(G.number_of_nodes()**2, dtype=np.int32)
    csv_traveltimes_np = np.zeros(G.number_of_nodes()**2, dtype=np.float32)
    
    # append each element of csv_dict to csv_paths_np, being careful to convert each node_id to its index in list(G.nodes)
    # Also compute travel times between each pair of node_ids and append to csv_traveltimes_np
    i = 0
    for start_id, row in csv_dict.items():
        for dest_id, next_node_id in row.items():
            csv_paths_np[i] = id_to_idx[next_node_id]
            csv_traveltimes_np[i] = compute_travel_time(start_id, dest_id, csv_dict, G) * 4 # multiply by 4 because osm estimates are too optimistic
            i += 1

    # reshape to a square matrix
    csv_paths_np = np.reshape(csv_paths_np, (G.number_of_nodes(), G.number_of_nodes()))
    csv_traveltimes_np = np.reshape(csv_traveltimes_np, (G.number_of_nodes(), G.number_of_nodes()))

    # Save as CSV
    np.savetxt('out/zone_path.csv', csv_paths_np, fmt='%i', delimiter=',')
    np.savetxt('out/zone_traveltime.csv', csv_traveltimes_np, fmt='%1.1f', delimiter=',')

    print('CSV files zone_paths.csv and zone_traveltime.csv generated successfully.')
    print('Remember to run traveltime_updater.py before using the traveltime CSV in the neural net.')



