import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r"Adjusted Dry Bean Dataset.csv")

for column in df.columns[:-1]: 
    data = df[column].tolist()
    plt.figure() 
    plt.hist(data, zorder=3, rwidth=0.9)
    plt.grid(True, zorder=2)
    plt.title(column)
    
    plt.savefig(f"{column}_histogram.png", transparent=True)
    plt.show()