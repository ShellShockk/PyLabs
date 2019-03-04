import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.optimize as so

#1.1
A = np.matrix([[1, 2, 3], [-2, 3, 0], [5, 1, 4]])
B = np.matrix([[1, 2, 3], [2, 4, 6], [3, 7, 2]])
u = np.array([-4, 1, 1])
v = np.array([3, 2, 10])

#1.2
C = np.random.random((100, 100))
w = np.random.random((100, 1))
matrixC = np.matrix(C)


#1.3
print(A + B) 
print(np.dot(A, B)) 
print(A.dot(A).dot(A)) 
print(A.dot(B).dot(A)) 
print(np.dot(v.T, A.T).dot(u + 2*v)) 
print(u.dot(v)) 
print(C.dot(w)) 
print(w.T.dot(C)) 


#1.4
m = np.array(range(0, 20))
m = np.tile(m, (20, 1))
result = m * m.T
print(result)


#1.5
sio.savemat('matrixes', {'A': A, 'B': B, 'u': u, 'v': v})
checkMatrixA = sio.loadmat('matrixes')['A']
print(checkMatrixA)


#1.6
sumResult = A[A > 0].sum() + B[B > 0].sum()
print(sumResult)


#1.7
oneRow = A.reshape((1,A.shape[0]*A.shape[1]))
print(oneRow)
oneRow2 = oneRow[0, range(1, oneRow.shape[1], 2)]
print(oneRow2)


#1.8
invA = np.linalg.inv(A)
pinvA = np.linalg.pinv(A)
pinvB = np.linalg.pinv(B)
pinvC = np.linalg.pinv(C)
print(A.dot(invA), '\n')
print(B.dot(pinvB), '\n')
print(C.dot(pinvC), '\n')


#1.9
A = np.array([[32, 7, -6], [-5, -20, 3], [0, 1, -3]])
b = np.array([12, 3, 7])
x = np.linalg.solve(A, b)
print(x)
print('Test: ', A.dot(x))


#1.10
eigenvalues, eigenvectors = np.linalg.eig(A)
print('\n', eigenvectors)
print('\n', eigenvalues)


#1.11
def f1(x):
    return 5 * (x - 2)**4 - 1 / (x**2 + 8)
def f2(x):
    return 4 * (x[0] - 3*x[1])**2 + 7 * x[0] ** 4

min = so.minimize(f1, 0.0, method='BFGS')
print('\nmin f1(x) =', min.x)
print('Проверка: f1(x) =', f1(min.x))

min = so.minimize(f2, [5.0, 2.0], method='BFGS')
print('min f2(x) =', min.x)
print('Проверка: f2(x) =', f2(min.x))


#1.12
def g1(x):
    return x**5 - 2*x**4 + 3*x - 7
def g2(x):
    return x**5 + 2*x**4 - 3*x - 7

x = np.linspace(-5, 5, 50)
g1y = g1(x)
g2y = g2(x)

plt.plot(x, g1y)
plt.plot(x, g2y)

plt.xlabel('x')
plt.ylabel('g1(x), g2(x)')
plt.legend(['g1', 'g2'])
plt.title('Plot')

plt.show()


#1.13
def g1(x):
    return x**5 - 2*x**4 + 3*x - 7
def g2(x):
    return x**5 + 2*x**4 - 3*x - 7

plt.subplot(1, 3, 1)
plt.plot(x, g1y)
plt.xlabel('x')
plt.ylabel('g1(x)')
plt.legend(['g1'])
plt.title('Plot 1')

plt.subplot(1, 3, 3)
plt.plot(x, g2y)
plt.xlabel('x')
plt.ylabel('g2(x)')
plt.legend(['g2'])
plt.title('Plot 2')

plt.show()


#1.14
g1x0 = so.brentq(g1, -5, 5)
g2x0 = so.brentq(g2, -5, 5)

print(g1x0)
print(g2x0)