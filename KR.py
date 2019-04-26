import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csvfile = "D:\\HousePrices_HalfMil.csv"
data = pd.read_csv(csvfile, index_col = None, header = None)

X = data.values[:,:15]
Y = data.values[:,15:]

X = np.array(X)
Y = np.array(Y)

X = np.concatenate([np.ones((X.shape[0],1)), X], axis=1)

w = np.linalg.solve((X.T.dot(X)), X.T.dot(Y))

y_predict = X.dot(w)

r_2 = 1 - ((Y - y_predict)**2).sum() / (((Y - y_predict.mean())**2).sum())

print(r_2)

plt.scatter(Y, y_predict)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], c='r')
plt.show()

