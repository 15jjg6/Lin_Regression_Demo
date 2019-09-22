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