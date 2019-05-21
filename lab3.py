import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

data = pd.read_csv("data.txt")

X = data.values[:,:4]
Y = data.values[:,4] * 2 - 1

X = np.concatenate([np.ones((X.shape[0],1)), X], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.45)

l = Y_test.shape[0]

print(X_train)
print(Y_train)

w = np.random.normal(0.0, 1.0, 5)

eps = 0.001
n = 0.01
prev_w = w

dw = eps
while(dw >= eps):
    prev_w = w.copy()
    for j in range(5):
        sum = 0
        for i in range(X_train.shape[0]):
            sum += X_train[i,j] * Y_train[i] * sigmoid(-Y_train[i]*w.dot(X_train[i]))
        w[j] = w[j] + n * (1 / l) * sum
        
    dw = math.sqrt(((prev_w - w)**2).sum())
    
print(w)

Y_predict = X_test.dot(w)

y_predict_b = []
for x in Y_predict:
  y_predict_b.append(1 if x > 0 else -1)

sum = 0
for i in range(l):
  sum += 1 if Y_test[i] == y_predict_b[i] else 0

A = sum / l
print(A)