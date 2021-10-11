import matplotlib.pyplot as plt
import numpy as np

import matrixstitcher as mats
from matrixstitcher.method import LeastSquareTech

x = np.array(list(range(-5, 6)))
y = np.array([2, 7, 9, 12, 13, 14, 14, 13, 10, 8, 4])

plt.plot(x, y, 'kx', markersize=10)

X1 = np.stack([np.ones(x.shape), x], axis=1)
X1 = mats.Matrix(X1)
X2 = np.stack([np.ones(x.shape), x, x*x], axis=1)
X2 = mats.Matrix(X2)
y = mats.Matrix(y)


tech1 = LeastSquareTech()
alpha1 = tech1(X1, y)
print(alpha1[0], alpha1[1])
plt.plot(x, tech1.predict(X1).numpy(), c='r')


tech2 = LeastSquareTech()
alpha2 = tech2(X2, y)
print(alpha2[0], alpha2[1])
plt.plot(x, tech2.predict(X2).numpy(), c='b')

plt.grid(True)
plt.show()
