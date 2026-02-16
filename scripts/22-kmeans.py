import matplotlib.pyplot as plt
import numpy as np


def show_cluster(dataset, cluster, centroids=None):
    colors = ["blue", "red", "green", "purple"]
    markers = ["o", "^", "s", "d"]
    K = len(np.unique(cluster))
    for i in range(K):
        plt.scatter(
            dataset[cluster == i, 0],
            dataset[cluster == i, 1],
            color=colors[i],
            marker=markers[i],
        )
    if centroids is not None:
        plt.scatter(
            centroids[:, 0], centroids[:, 1], color=colors[:K], marker="+", s=150
        )
    plt.show()


data = np.loadtxt("./data/kmeans_data.csv", delimiter=",")
print("Data size: ", len(data))
show_cluster(data, np.zeros(len(data), dtype=int))
