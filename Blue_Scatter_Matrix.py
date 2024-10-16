import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix

df = pd.read_csv(r"Adjusted Dry Bean Dataset.csv")
columns = df[['Eccentricity', 'Solidity', 'Compactness', 'roundness', 'Area']]
scatter_matrix(columns, alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.savefig(fname = "Blue_Scatter_Matrix.png", transparent = True)
plt.show()