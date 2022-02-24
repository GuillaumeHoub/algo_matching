from cluster import kmeans, calculate_cost
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


dtypes = {"Car": "str", "MPG": "float", "Cylinders": "int", "Displacement": "float", "Horsepower": "float",
          "Weight": "float", "Acceleration": "float", "Model": "int", "Origin": "str"}

df = pd.DataFrame(pd.read_csv("./cars.csv", header=0, dtype=dtypes))

# Choisir les colonnes qui serviront pour le clustering
data = df.loc[:, ['Acceleration', 'Horsepower']]


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