# import osmnx as ox
# import networkx as nx
# import pandas as pd
# import numpy as np
# import csv
# from datetime import datetime
# import sys
# import geopy.distance
# from get_graph_Manhattan import get_graph
# import time

# G = get_graph()
# nodes = ox.graph_to_gdfs(G, edges=False)
# osmid_to_nodeid = dict(zip(list(nodes.index), list(range(nodes.shape[0]))))
# formatstring = '%Y-%m-%d %H:%M:%S'

# data = pd.read_csv('data/yellow_tripdata_2016-verkort.csv')
# data['tpep_pickup_datetime'] = pd.to_datetime(data["tpep_pickup_datetime"])

# for day in [25, 26, 27, 4]:
#     flow = 0
#     df = data[data['tpep_pickup_datetime'].dt.day.isin([day])]
#     # f = open(f"out/test_flow_5000_{day}.txt", 'w')
#     # sys.stdout = f  
#     print('1440')
#     print(f'Flows:{flow}-{flow}')
#     start_time = time.time()
#     for index, row in df.iterrows():
#         index += 1
#         new_time = row['tpep_pickup_datetime']
            
#         # Start new flow of 60 seconds
#         if new_time.second == 0:
#             flow += 1
#             print(f'Flows:{flow}-{flow}')
        
#         # Retrieve pickup and destination from csv 
#         pickup_osmid, dist_pickup = ox.distance.nearest_nodes(G, float(row['pickup_longitude']), float(row['pickup_latitude']), return_dist=True)
#         dest_osmid, dist_dropoff = ox.distance.nearest_nodes(G, float(row['dropoff_longitude']), float(row['dropoff_latitude']), return_dist=True)

#         if dist_pickup < 100 and dist_dropoff < 100:
#             print(f'{osmid_to_nodeid[pickup_osmid]}, {osmid_to_nodeid[dest_osmid]}, 1.0')
        
#         if index % 1000 == 0:
#             print("--- %s seconds ---" % (time.time() - start_time))
#             break
#     break
#     # f.close()
import pandas as pd

data = pd.read_csv('data/yellow_tripdata_2016-miniselection.csv')
print(data.head())
