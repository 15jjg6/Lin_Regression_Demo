import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #python library for plot and graphs
from LinearModel import LinearModel

def normscaler(Z, normal=False, scale='max'):
    Zn = np.zeros(Z.shape)
    for col in range(Zn.shape[1]):
        std = Z[:, col].std()
        clm = Z[:, col]
        mn = Z[:, col].mean()
        mx = Z[:, col].max()
        nrm = 0
        sclr = 1
        if normal:
            nrm = mn
        if scale == 'max':
            sclr = mx
        elif scale == 'std':
            sclr = std
        Zn[:, col] = (clm - nrm) / sclr

    return Zn


def cost_function(X, y, theta, deriv=False):
    z = np.ones((len(X), 1))
    X = np.append(z, X, axis=1)

    if deriv:
        loss = X.dot(theta) - y
        gradient = X.T.dot(loss) / len(X)
        return gradient, loss

    else:
        h = X.dot(theta)
        j = (h - y.flatten())
        J = j.dot(j) / 2 / (len(X))
        return J


X=1
y=0
a = LinearModel(5,4)
print(a.GradDesc(2,0.01,'MSE'))
print(a.Cost('RMSE'))

data = pd.read_csv('housing_data', header=None)
data.columns =(['Size','Bedroom','Price'])
data.drop('Bedroom', axis=1, inplace=True)
data = data.sample(frac=1)

print(data.head())

plt.plot(data.Size, data.Price, 'r.')
plt.show()

print(data.corr())

A = np.array([[1,2],
              [1,3],
              [1,4]])
B = np.array([[2],[3]])

print('A =')
print(A,'\nsize =',A.shape)
print('\nB =')
print(B,'\nsize =',B.shape)

# let's try it
H = A.dot(B)
print(H)


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


theta = np.array([0.9,-1])


lineX = np.linspace(Xn.min(), Xn.max(), 100)
liney = [theta[0] + theta[1]*xx for xx in lineX]

plt.plot(Xn,yn,'r.', label='Training data')
plt.plot(lineX,liney,'b--', label='Current hypothesis')
plt.legend()
plt.show()

