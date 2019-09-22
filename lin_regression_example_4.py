import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #python library for plot and graphs


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


print(cost_function(Xn, yn, theta))

def GradDescent(features, target, param, learnRate=0.01, multiple=1, batch=len(X), log=False):
    iterations = batch * len(features)
    epochs = iterations * multiple
    y = target.flatten()
    t = param
    b = batch
    a = learnRate

    theta_history = np.zeros((param.shape[0], epochs)).T
    cost_history = [0] * epochs

    for ix in range(epochs):

        i = epochs % len(X)
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


alpha = 1.5
mul = 10
bat = 8
ch, th = GradDescent(Xn, yn, theta, alpha, mul, bat, log=False)


lineX = np.linspace(Xn.min(), Xn.max(), 100)
liney = [th[-1,0] + th[-1,1]*xx for xx in lineX]

plt.plot(Xn,yn,'r.', label='Training data')
plt.plot(lineX,liney,'b--', label='Current hypothesis')
plt.legend()
plt.show()


plt.plot(ch,'g--')
plt.show()


plt.plot(th[:,0],'r-.')
plt.plot(th[:,1],'b-.')
plt.show()


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




# =========== #
# Animation
import matplotlib.animation as animation

# Set the plot up,
fig = plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.plot(Xn, yn, 'ro', label='Training data')
plt.title('Housing Price Prediction')
plt.axis([Xn.min() - Xn.std(), Xn.max() + Xn.std(), yn.min() - yn.std(), yn.max() + yn.std()])
plt.grid(axis='both')
plt.xlabel("Size of house in ft^2 (X1) ")
plt.ylabel("Price in $1000s (Y)")
plt.legend(loc='lower right')

line, = plt.plot([], [], 'b-', label='Current Hypothesis')
annotation = plt.text(-2, 3, '', fontsize=20, color='green')
annotation.set_animated(True)

plt.subplot(122)
cp = plt.contourf(A, B, C)
plt.colorbar(cp)
plt.title('Filled Contours Plot')
plt.xlabel('theta 0')
plt.ylabel('theta 1')
track, = plt.plot([], [], 'r-')
point, = plt.plot([], [], 'ro')

plt.tight_layout()
plt.close()


# Generate the animation data,
def init():
    line.set_data([], [])
    track.set_data([], [])
    point.set_data([], [])
    annotation.set_text('')
    return line, track, point, annotation


# animation function.  This is called sequentially
def animate(i):
    fit1_X = np.linspace(Xn.min() - Xn.std(), Xn.max() + Xn.std(), 1000)
    fit1_y = th[i][0] + th[i][1] * fit1_X

    fit2_X = th.T[0][:i]
    fit2_y = th.T[1][:i]

    track.set_data(fit2_X, fit2_y)
    line.set_data(fit1_X, fit1_y)
    point.set_data(th.T[0][i], th.T[1][i])

    annotation.set_text('Cost = %.4f' % (ch[i]))
    return line, track, point, annotation


anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=800, interval=0, blit=True)

anim.save('animation.html', writer='imagemagick', fps=30)