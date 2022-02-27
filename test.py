import pickle
import pandas as pd
import numpy as np

dtypes = {"Car": "str", "MPG": "float", "Cylinders": "int", "Displacement": "float", "Horsepower": "float",
          "Weight": "float", "Acceleration": "float", "Model": "int", "Origin": "str"}

df = pd.DataFrame(pd.read_csv("./cars.csv", header=0, dtype=dtypes))

# Choisir les colonnes qui serviront pour le test
colonnes = ["Weight", "MPG"]
data = df.loc[:, [colonnes[0], colonnes[1]]]
np_array = data.values

def test_clustering(file_to_test):
    with open(file_to_test, "rb") as f:
        cluster, centroids = pickle.load(f)
    cluster_test = np.zeros(np_array.shape[0])
    # Vérification des distances entre les points et les centroids
    for i, row in enumerate(np_array):
        mn_dist = float('inf')
        for idx, centroid in enumerate(centroids):
            d = np.sqrt((centroid[0] - row[0]) ** 2 + (centroid[1] - row[1]) ** 2)
            if mn_dist > d:
                mn_dist = d
                cluster_test[i] = idx
    # Si les deux clusters sont identiques alors le clustering est bon
    if np.count_nonzero(cluster - cluster_test) == 0:
        return True
    return False


if __name__ == '__main__':
    # Choix du fichier de cluster à tester
    file_to_test = "data/Weight_MPG_clusters"
    print(test_clustering(file_to_test))

