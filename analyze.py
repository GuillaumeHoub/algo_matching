import pandas as pd

dtypes = {"Car": "str", "MPG": "float", "Cylinders": "int", "Displacement": "float", "Horsepower": "float", "Weight": "float", "Acceleration": "float", "Model": "int", "Origin": "str"}

df = pd.read_csv("./cars.csv", header=0, dtype=dtypes)

#Generate Histograms for MPG
print('---\nGenerate Histogram for MPG')
df.hist(column='MPG')

print('---\nGenerate Histogram for Cylindres')
df.hist(column='Cylinders')

print('---\nGenerate Histogram for Displacement')
df.hist(column='Displacement')

print('---\nGenerate Histogram for Horsepower')
df.hist(column='Horsepower')

print('---\nGenerate Histogram for Weight')
df.hist(column='Weight')

print('---\nGenerate Histogram for Acceleration')
df.hist(column='Acceleration')

print('---\nGenerate Histogram for Model')
df.hist(column='Model')