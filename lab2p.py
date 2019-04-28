import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

k = 0
for line in open('D:\\sample_submission.csv'):
    x, y = line.split(',')
    X.append(k)
    Y.append(float(y))
    k += 1

X = np.array(X)
Y = np.array(Y)

X_mean = X.mean()
X2_mean = (X * X).mean()
Y_mean = Y.mean()
XY_mean = (X * Y).mean()

denom = X2_mean - X_mean**2
a = (XY_mean - X_mean * Y_mean) / denom
b = (Y_mean * X2_mean - X_mean * XY_mean) / denom

Y_predict = a*X+b

plt.scatter(X, Y, color="blue")
plt.plot(X, Y_predict, color="red")
plt.show()