import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dtypes = {"Car": "str", "MPG": "float", "Cylinders": "int", "Displacement": "float", "Horsepower": "float",
          "Weight": "float", "Acceleration": "float", "Model": "int", "Origin": "str"}

df = pd.DataFrame(pd.read_csv("./cars.csv", header=0, dtype=dtypes))

# Choisir les colonnes qui serviront pour le clustering
colonnes = ["Weight", "MPG"]
data = df.loc[:, [colonnes[0], colonnes[1]]]

def kmeans(np_array, k):
    diff = 1
    cluster = np.zeros(np_array.shape[0])
    # Génération des centroids par défaut
    centroids = data.sample(n=k).values
    while diff:
        for i, row in enumerate(np_array):
            mn_dist = float('inf')
            # Calcul de la distance du point avec tout les centroids
            for idx, centroid in enumerate(centroids):
                d = np.sqrt((centroid[0] - row[0])**2 + (centroid[1] - row[1])**2)
                # Sauvegarde de la distance la plus courte
                if mn_dist > d:
                    mn_dist = d
                    cluster[i] = idx
        # Mise à jour des centroids avec les cluster générés
        new_centroids = pd.DataFrame(np_array).groupby(by=cluster).mean().values
        # Si les nouveaux centroids n'ont pas changés on quitte la boucle
        if np.count_nonzero(centroids - new_centroids) == 0:
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
    k = 4
    centroids, cluster = kmeans(np_array, k)
    # Sauvegarde des clusters et centroids dans le dossier data
    with open("data/" + colonnes[0] + "_" + colonnes[1] + "_clusters", "wb") as f:
        pickle.dump((cluster, centroids), f)
    sns.scatterplot(x=np_array[:, 0], y=np_array[:, 1], hue=cluster)
    sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], s=100, color='y')
    plt.xlabel(colonnes[0])
    plt.ylabel(colonnes[1])
    plt.show()