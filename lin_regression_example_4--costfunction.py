import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #python library for plot and graphs

from LinearModel import *
plt.style.use('fivethirtyeight')


data = pd.read_csv('housing_data', header=None)
data.columns =(['Size','Bedroom','Price'])
data.drop('Bedroom', axis=1, inplace=True)

X = np.array(data.drop('Price',axis=1))
y = np.array(data.Price)
m = len(data)


y = y.reshape((m,1))


Xn = normscaler(X, normal=True, scale='std')
yn = normscaler(y, normal=True, scale='std')


theta = np.array([0.9,-1])


lineX = np.linspace(Xn.min(), Xn.max(), 100)
liney = [theta[0] + theta[1]*xx for xx in lineX]


print(cost_function(Xn, yn, theta))
