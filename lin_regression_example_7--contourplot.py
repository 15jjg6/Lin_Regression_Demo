import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #python library for plot and graphs

from LinearModel import *
plt.style.use('fivethirtyeight')


# Read CSV files for housing data and drop columns with unnecessary info
data = pd.read_csv('housing_data', header=None)
data.columns =(['Size','Bedroom','Price'])
data.drop('Bedroom', axis=1, inplace=True)

# Turn pandas columns into numpy arrays
X = np.array(data.drop('Price',axis=1))
y = np.array(data.Price)
m = len(data)
y = y.reshape((m,1))

# Normalize and scale
Xn = normscaler(X, normal=True, scale='std')
yn = normscaler(y, normal=True, scale='std')

# Initial theta val guess
theta = np.array([0.9,-1])

alpha = 0.1
mul = 10
bat = 8
ch, th = GradDescent(Xn, yn, theta, alpha, mul, bat, log=False)


lineX = np.linspace(Xn.min(), Xn.max(), 100)
liney = [th[-1,0] + th[-1,1]*xx for xx in lineX]

#Grid over which we will calculate J
theta0_vals = np.linspace(-2, 2, 100)
theta1_vals = np.linspace(-2, 3, 100)

#initialize J_vals to a matrix of 0's
J_vals = np.zeros((theta0_vals.size, theta1_vals.size))

#Fill out J_vals
for t1, element in enumerate(theta0_vals):
    for t2, element2 in enumerate(theta1_vals):
        thetaT = np.zeros(shape=(2, 1))
        thetaT[0][0] = element
        thetaT[1][0] = element2
        J_vals[t1, t2] = cost_function(Xn, yn, thetaT.flatten())

#Contour plot
J_vals = J_vals.T
A, B = np.meshgrid(theta0_vals, theta1_vals)
C = J_vals

cp = plt.contourf(A, B, C)
plt.colorbar(cp)
plt.plot(th.T[0],th.T[1],'r--')
plt.show()
