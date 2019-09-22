import numpy as np

class LinearModel():

    def __init__(self, features, target):
        self.X = features
        self.y = target

    def GradDesc(self, parameters, learningRate, cost):
        self.a = learningRate
        self.c = cost
        self.p = parameters
        return self.a, self.Cost(self.c), self.p

    def Cost(self, c):
        if c == 'RMSE':
            return self.y
        elif c == 'MSE':
            return self.X


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


def GradDescent(features, target, param, learnRate=0.01, multiple=1, batch=8, log=False):
    iterations = batch * len(features)
    epochs = iterations * multiple
    y = target.flatten()
    t = param
    b = batch
    a = learnRate

    theta_history = np.zeros((param.shape[0], epochs)).T
    cost_history = [0] * epochs

    for ix in range(epochs):

        i = epochs % 8
        cost = cost_function(features[i:i + b], y[i:i + b], t)

        cost_history[ix] = cost
        theta_history[ix] = t

        g, l = cost_function(features[i:i + b], y[i:i + b], t, deriv=True)
        t = t - a * g

        if log:
            if ix % 250 == 0:
                print("iteration :", ix + 1)
                # print("\tloss     = ", l)
                print("\tgradient = ", g)
                print("\trate     = ", a * g)
                print("\ttheta    = ", t)
                print("\tcost     = ", cost)

    return cost_history, theta_history