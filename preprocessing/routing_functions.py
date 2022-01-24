import osmnx as ox
import random
import pickle

def save_var(var, file_name):
    with open('out/' + file_name, 'wb') as f:
        pickle.dump(var, f)

def print_progress(remaining_pairs, G):
    """
    Prints the percentage of all (start,dest) pairs that are already computed
    """
    total_pairs = G.number_of_nodes()**2
    uncomputed_pairs = len(remaining_pairs)
    print(1 - uncomputed_pairs/total_pairs)


def update_recursive(route, csv, remaining_pairs):
    """
    Updates the csv and the set of pairs based on the computed route
    route - Numpy array with all node_ids of the route, including start and destination. Should be a np array for optimal performance
    csv - the csv dict
    remainging_pairs - set with all pairs between which the route has yet to be computed
    """

    # exit this recursive function if the route contains only 1 node
    if len(route) < 2:
        return
    
    # the start node is the first node
    start_node_id = route[0]
    # the next node to visit is the second node on this route
    next_node_id = route[1]

    for dest_node_id in route[1:]:
        # update the csv
        csv[start_node_id][dest_node_id] = next_node_id

        # remove the pair from the set of pairs yet to be processed
        pair = (start_node_id, dest_node_id)
        if pair in remaining_pairs:
            remaining_pairs.remove(pair)

    update_recursive(route[1:], csv, remaining_pairs)


def update_shallow(route, csv, remaining_pairs):
    """
    Updates the csv and the set of pairs based on the computed route, but does not update all subroutes.
    This is more efficient if most subroutes have already been computed, as no recursive calls have to be made
    """

    dest_id = route[-1]
    for i in range(len(route) - 1):

        cur_node_id = route[i]

        # if this route was already computed, then we know the remainder of the route is already in the csv
        # so we can stop here.
        if csv[cur_node_id][dest_id] is not None:
            return

        csv[cur_node_id][dest_id] = route[i+1]

        # remove the pair from the set of pairs yet to be processed
        pair = (cur_node_id, dest_id)
        if pair in remaining_pairs:
            remaining_pairs.remove(pair)


def compute_travel_time(start_id, dest_id, csv, G):
    """
    Computes the travel time of the shortest route from start_id to dest_id using the
    route information stored in the csv dict
    """
    
    # route is not computed yet
    if csv[start_id][dest_id] is None:
        return -1

    travel_time = 0
    cur_node_id = start_id

    while cur_node_id != dest_id:
        
        # get the next node on this route
        next_node_id = csv[cur_node_id][dest_id]

        # update the travel time
        edge = G.get_edge_data(cur_node_id, next_node_id)
        travel_time += edge[0]['travel_time']

        cur_node_id = next_node_id

    return travel_time


def get_route(start_id, dest_id, csv):
    """
    Returns the optimal route from start_id to dest_id using the route
    information from csv. The route includes the start and end node.
    """

    # route is not computed yet
    if csv[start_id][dest_id] is None:
        return -1

    route = [start_id]

    while route[-1] != dest_id:
        # append the next node on this route
        route.append(csv[route[-1]][dest_id])

    return route


def add_predecessors(dest_id, predecessor_set, G, degree, max_degree):
    """
    Adds all predecessors that are no more than max_degree edges removed from id to the set.
    A predecessor of n is a node m such that there exists a directed edge from m to n.
    
    dest_id - id of the destination node whose predecessors to add
    predecessor_set - to which to add all predecessors. Should be a set to avoid double entries.
    degree - the degree of this recursive invocation
    max_degree - the maximum number of edges between the root node and the predecessors in the set
    """
    # add all neighbours up to max_degree degree
    if degree > max_degree:
        return
    
    for pred_id in G.predecessors(dest_id):

        # only recurse if this predecessor wasn't already added
        if pred_id not in predecessor_set:            
            predecessor_set.add(pred_id)

            # recursively add all predecessors of this node as well
            add_predecessors(pred_id, predecessor_set, G, degree + 1, max_degree)


def get_all_predecessor_pairs(G):
    """
    Returns a list that contains all pairs (n,m) where n is a predecessor of m
    of up to 12th degree
    """

    all_predecessor_pairs = []

    for dest_id in G.nodes:
        predecessors = set()
        add_predecessors(dest_id, predecessors, G, 1, 12)

        # create all pairs from the predecessors to the dest node and add to the list
        for predecessor_id in predecessors:
            pair = (predecessor_id, dest_id)
            all_predecessor_pairs.append(pair)


    # shuffle all pairs in-place. It would probably be more efficient to put the higher degree pairs last,
    # because then many shorter subroutes are automatically computed when these long routes are computed first,
    # but the efficiency improvement is only minor so to keep code simple we just shuffle all pairs.
    random.shuffle(all_predecessor_pairs)

    return all_predecessor_pairs


def compute_shortest_routes(start_ids, dest_ids, G):
    """
    Computes the shortest routes between all node pairs (start_ids[i], dest_ids[i]) for all i
    Returns a list of lists, where each inner list contitutes a route
    """
    return ox.distance.shortest_path(G, start_ids, dest_ids, weight='travel_time', cpus=6)


def get_next_batch(batch_size, pairs_list, remaining_pairs_set):
    """
    Returns a batch of pairs to compute the route between.
    pairs_list - a list of all pairs, from which the pairs will be popped
    remaining_pairs_set - a set that contains stricly only those pairs for which no route is yet computed.
    The batch will contain batch_size pairs all of which are checked to be present in remaining_pairs_set so as to ensure
    no routes are recomputed
    """
    start_ids = []
    dest_ids = []

    while len(start_ids) < batch_size and len(pairs_list) > 0:
        pair = pairs_list.pop()

        # only add the pair if it is present in the remaining_pairs_set.
        # if the pair is not contained in the set, then a route between it was already computed
        # and we can simply discard it
        if pair in remaining_pairs_set:
            start_ids.append(pair[0])
            dest_ids.append(pair[1])

    return start_ids, dest_ids


def get_all_pairs(G):
    """
    Generates a list and a set with (start_id, dest_id) tuples between all nodes in the graph. 
    The list is to allow random ordering (sets have no reliable order), and the set is to allow
    O(1) deletion and presence checking.
    """
    # list all (start,dest) pairs between which the route must be computed
    pairs_list = [(start, dest) for dest in G.nodes for start in G.nodes]

    # shuffle all elements in-place
    random.shuffle(pairs_list)

    # generate a set from the list
    pairs_set = set(pairs_list)

    return pairs_list, pairs_set


def generate_csv_dict(G):
    """
    Generates a dict containing on the first axis the start node id, on the second axis the destination node id, 
    and in each cell the id of the next node if you want to go from the start node to the destination node.
    Example: csv[1][50] contains the node_id of the next node on the quickest route from node 1 to node 50
    All cells will be initialised with None. Hence, a cell with None means that no route has been computed between the corresponding nodes.
    """
    csv_dict = { start_node_id: { dest_node_id: None for dest_node_id in G.nodes } for start_node_id in G.nodes }
    return csv_dict


def compute_routes_for_batch(batch_size, pairs_list, remaining_pairs_set, csv_dict, G, recursive_update=True):

    start_ids, dest_ids = get_next_batch(batch_size, pairs_list, remaining_pairs_set)

    # compute the routes
    routes = compute_shortest_routes(start_ids, dest_ids, G)

    for i, route in enumerate(routes):

        # route is None if no route was found. This should never happen since our
        # graph is strongly connected. We catch this case nonetheless for debugging purposes.
        if route is None:
            print('Could not compute route between (', start_ids[i], ',', dest_ids[i],').')
        else:
            if recursive_update:
                update_recursive(route, csv_dict, remaining_pairs_set)
            else:
                update_shallow(route, csv_dict, remaining_pairs_set)

    print_progress(remaining_pairs_set, G)


def get_all_remaining_pairs(csv_dict):
    """
    Loops through the csv to find all (start_id, dest_id) pairs for which
    no route has been computed.
    """
    pairs = []
    for start_id, row in csv_dict.items():
        for dest_id, el in row.items():
            if el is None:
                pair = (start_id, dest_id)
                pairs.append(pair)

    return pairs