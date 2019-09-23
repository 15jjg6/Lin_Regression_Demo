import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #python library for plot and graphs
plt.style.use('xkcd')


# Read CSV files for housing data and drop columns with unnecessary info
data = pd.read_csv('housing_data', header=None)
data.columns =(['Size','Bedroom','Price'])
data.drop('Bedroom', axis=1, inplace=True)

print(data.head())

plt.plot(data.Size, data.Price, 'r.')

plt.show()

print(data.corr())
