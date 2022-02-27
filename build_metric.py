import math
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import pandas as pd
from IPython import display
from graphviz import Graph

dataframe = pd.read_csv("./cars.csv")
nb_cars = len(dataframe.index)
print(nb_cars)
print(dataframe.info())
print(dataframe.head(10))
dataframe = dataframe.head(40)
nb_cars = len(dataframe.index)
car_id = 0
print(f"---\nall information on car {car_id}\n---")
print(dataframe.loc[car_id])

def compute_dissimilarity(car_1_id, car_2_id):
    """
        Compute  dissimilarity betwwen two cars
        based on their id.

        The meal is not a quantitative attribute.
        It is called a categorical variable.
        We must handle it differently than quantitative
        attributes.
    """
    car_1_MPG = dataframe.loc[car_1_id][1]
    car_2_MPG = dataframe.loc[car_2_id][1]

    car_1_weight = dataframe.loc[car_1_id][5]
    car_2_weight = dataframe.loc[car_2_id][5]
    # EDIT HERE
    dissimilarity = math.sqrt(
        (car_1_MPG-car_2_MPG)**2+(car_1_weight-car_2_weight)**2)

    print("----")
    car_1_name = dataframe.loc[car_1_id]["Car"]
    car_2_name = dataframe.loc[car_2_id]["Car"]
    print(f"plyr 1 {car_1_name}, plyr 2 {car_2_name}, dissimilarity: {dissimilarity}")
    return dissimilarity

dissimilarity_matrix = np.zeros((nb_cars, nb_cars))
print("compute dissimilarities")
for car_1_id in range(nb_cars):
    for car_2_id in range(nb_cars):
        dissimilarity = compute_dissimilarity(car_1_id, car_2_id)
        dissimilarity_matrix[car_1_id, car_2_id] = dissimilarity

print("dissimilarity matrix")
print(dissimilarity_matrix)
threshold = 1000
# build a graph from the dissimilarity
dot = Graph(comment='Graph created from complex data',
            strict=True)
for car_id in range(nb_cars):
    car_name = dataframe.loc[car_id][0]
    dot.node(car_name)

for car_1_id in range(nb_cars):
    # we use an undirected graph so we do not need
    # to take the potential reciprocal edge
    # into account
    for car_2_id in range(nb_cars):
        # no self loops
        if not car_1_id == car_2_id:
            car_1_name = dataframe.loc[car_1_id][0]
            car_2_name = dataframe.loc[car_2_id][0]
            # use the threshold condition
            if dissimilarity_matrix[car_1_id, car_2_id] > threshold:
                dot.edge(car_1_name,
                         car_2_name,
                         color='darkolivegreen4',
                         penwidth='1.1')

# visualize the graph
dot.attr(label=f"threshold {threshold}", fontsize='20')
graph_name = f"images/Weight_MPG_DiffGraph_threshold_{threshold}"
dot.render(graph_name)