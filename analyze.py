import pandas as pd

dtypes = {"Car": "str", "MPG": "float", "Cylinders": "int", "Displacement": "float", "Horsepower": "float", "Weight": "float", "Acceleration": "float", "Model": "int", "Origin": "str"}

df = pd.DataFrame(pd.read_csv("./cars.csv", header=0, dtype=dtypes))

#Generate Histograms for MPG
print('---\nGenerate Histogram for MPG')
df.hist(column='MPG')
#Example to print only one column
#print(df.std()['MPG'])

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

print('---\nGenerate Histogram for Origin')
count_origin = df['Origin'].value_counts()
count_origin.plot(kind='bar')

#Print the standart Deviation of every Histograms
print('---\nGenerate Standart Deviation')
print(df.std())