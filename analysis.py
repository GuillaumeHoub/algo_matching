import pandas as pd

dtypes = {"Car": "str", "MPG": "float", "Cylinders": "int", "Displacement": "float", "Horsepower": "float", "Weight": "float", "Acceleration": "float", "Model": "int", "Origin": "str"}
df = pd.read_csv("./cars.csv", header=0, dtype=dtypes)

# general info on the dataframe
print('---\ngeneral info on the dataframe')
print(df.info())

# print the columns of the dataframe
print('---\ncolumns of the dataset')
print(df.columns)

# print the first 10 lines of the dataframe
print('---\nfirst lines')
print(df.head(10))

# print the correlation matrix of the dataset
print('---\nCorrelation matrix')
print(df.corr())

# print the standard deviation
print('---\nStandard deviation')
print(df.std())