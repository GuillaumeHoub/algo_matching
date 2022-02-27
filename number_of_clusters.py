from cluster import kmeans, data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Permet de calculer le cout WCSS (Within Cluster Sum of Square)
def calculate_cost(np_array, centroids, cluster):
    sum = 0
    for i, val in enumerate(np_array):
        sum += np.sqrt((centroids[int(cluster[i]), 0] - val[0]) ** 2 + (centroids[int(cluster[i]), 1] - val[1]) ** 2)
    return sum

def find_optimal_number_of_cluster(np_array):
    cost_list = []
    best_k = 0
    # Calcul du cout WCSS pour un interval de 1 à 10 cluster (Methode Elbow)
    for k in range(1, 10):
        centroids, cluster = kmeans(np_array, k)
        cost = calculate_cost(np_array, centroids, cluster)
        cost_list.append(cost)
    for i in range(1, len(cost_list[1:])):
        # Si la diminution en cout WCSS entre un k et un k + 1 est inférieur à 20 % on considère que K est le nombre
        # optimal de cluster (peu fiable mais automatique)
        if (cost_list[i] / cost_list[i - 1]) >= 0.8:
            best_k = i
            break
    return best_k, cost_list

if __name__ == '__main__':
    np_array = data.values
    best_k, cost_list = find_optimal_number_of_cluster(np_array)
    sns.lineplot(x=range(1, 10), y=cost_list, marker='o')
    plt.xlabel('k')
    plt.ylabel('WCSS')
    plt.title("best number of cluster = " + str(best_k))
    plt.show()