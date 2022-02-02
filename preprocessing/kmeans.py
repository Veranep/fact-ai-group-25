import pickle
from sklearn.cluster import KMeans

def write_kmeans():

    zone_lat_long = open("../data/paper_replication/zone_latlong.csv").read().split("\n")
    d = {}
    coords = []
    for i in zone_lat_long:
        if i!='':
            a,b,c = i.split(",")
            d[a] = (float(b),float(c))
            coords.append((float(b),float(c)))

    regions = KMeans(n_clusters=10).fit(coords)
    labels = regions.labels_
    centers = regions.cluster_centers_

    pickle.dump(labels,open("../data/paper_replication/new_labels.pkl","wb"))

write_kmeans()
