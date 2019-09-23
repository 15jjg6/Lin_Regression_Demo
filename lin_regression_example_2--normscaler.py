import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LinearModel import normscaler
plt.style.use('fivethirtyeight')


# Read CSV files for housing data and drop columns with unnecessary info
data = pd.read_csv('housing_data', header=None)
data.columns =(['Size','Bedroom','Price'])
data.drop('Bedroom', axis=1, inplace=True)

# Turn pandas columns into numpy arrays
X = np.array(data.drop('Price',axis=1))
y = np.array(data.Price)
m = len(data)

print(X.shape)
print(y.shape)
print(m)

# Reshape y column from flattened array to n by 1 matrix
y = y.reshape((m,1))
print(y.shape)


Xn = normscaler(X, normal=True, scale='std')
yn = normscaler(y, normal=True, scale='std')

plt.plot(Xn, yn, 'r.')
plt.show()


