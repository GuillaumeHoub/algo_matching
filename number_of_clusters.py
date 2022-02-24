from cluster import kmeans, data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_cost(np_array, centroids, cluster):
    sum = 0
    for i, val in enumerate(np_array):
        sum += np.sqrt((centroids[int(cluster[i]), 0] - val[0]) ** 2 + (centroids[int(cluster[i]), 1] - val[1]) ** 2)
    return sum

def find_optimal_number_of_cluster(np_array):
    cost_list = []
    for k in range(1, 10):
        centroids, cluster = kmeans(np_array, k)
        # WCSS (Within cluster sum of square)
        cost = calculate_cost(np_array, centroids, cluster)
        cost_list.append(cost)
    return cost_list

if __name__ == '__main__':
    np_array = data.values
    cost_list = find_optimal_number_of_cluster(np_array)
    sns.lineplot(x=range(1, 10), y=cost_list, marker='o')
    plt.xlabel('k')
    plt.ylabel('WCSS')
    plt.show()