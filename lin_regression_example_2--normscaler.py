import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #python library for plot and graphs
from LinearModel import normscaler
plt.style.use('fivethirtyeight')


data = pd.read_csv('housing_data', header=None)
data.columns =(['Size','Bedroom','Price'])
data.drop('Bedroom', axis=1, inplace=True)

X = np.array(data.drop('Price',axis=1))
y = np.array(data.Price)
m = len(data)

print(X.shape)
print(y.shape)
print(m)


y = y.reshape((m,1))
print(y.shape)


Xn = normscaler(X, normal=True, scale='std')
yn = normscaler(y, normal=True, scale='std')

plt.plot(Xn, yn, 'r.')
plt.show()


