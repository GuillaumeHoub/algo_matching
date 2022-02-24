import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dtypes = {"Car": "str", "MPG": "float", "Cylinders": "int", "Displacement": "float", "Horsepower": "float",
          "Weight": "float", "Acceleration": "float", "Model": "int", "Origin": "str"}

df = pd.DataFrame(pd.read_csv("./cars.csv", header=0, dtype=dtypes))

# Choisir les colonnes qui serviront pour le clustering
colonnes = ["Acceleration", "Horsepower"]
data = df.loc[:, [colonnes[0], colonnes[1]]]

def kmeans(np_array, k):
    diff = 1
    cluster = np.zeros(np_array.shape[0])
    centroids = data.sample(n=k).values
    while diff:
    # for each observation
        for i, row in enumerate(np_array):
            mn_dist = float('inf')
            # dist of the point from all centroids
            for idx, centroid in enumerate(centroids):
                d = np.sqrt((centroid[0] - row[0])**2 + (centroid[1] - row[1])**2)
                # store closest centroid
                if mn_dist > d:
                    mn_dist = d
                    cluster[i] = idx
        new_centroids = pd.DataFrame(np_array).groupby(by=cluster).mean().values
    # if centroids are same then leave
        if np.count_nonzero(centroids-new_centroids) == 0:
            diff = 0
        else:
            centroids = new_centroids
    return centroids, cluster

if __name__ == '__main__':
    np_array = data.values
    print(np_array.shape)
    sns.scatterplot(x=np_array[:, 0], y=np_array[:, 1])
    plt.xlabel(colonnes[0])
    plt.ylabel(colonnes[1])
    plt.show()
    

    #Choix du nombre de cluster grace à number_of_cluster.py
    k = 5
    centroids, cluster = kmeans(np_array, k)
    sns.scatterplot(x=np_array[:, 0], y=np_array[:, 1], hue=cluster)
    sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], s=100, color='y')
    plt.xlabel(colonnes[0])
    plt.ylabel(colonnes[1])
    plt.show()