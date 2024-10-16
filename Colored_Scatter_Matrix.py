import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Adjusted Dry Bean Dataset.csv')
selected_columns = df[[ 'Eccentricity', 'Solidity', 'Compactness', 'roundness', 'Area', 'Class']]
sns.pairplot(selected_columns, hue='Class')
plt.savefig(fname = "Colored_Scatter_Matrix.png", transparent = True)
plt.show()