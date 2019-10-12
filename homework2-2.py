import numpy as np
import matrixstitcher as mats 
from matrixstitcher.transform import *



A = [[2, 0, -1], [-1, 1, 1], [-1, 0, 1]]
inv_A = [[1, 0, 1], [0, 1, -1], [1, 0, 2]]
C = [[0, 0, 0], [0, 0, 0], [2, 0, 1]]
D = [[0, 0, 0], [1, 0, 0], [0, 0, 1]]

A = Matrix(A)
inv_A = Matrix(inv_A)
C = Matrix(C)
D = Matrix(D)
ones = Matrix(np.eye(A.rows))

print(C*D.T)
print(A + C*D.T)
det = ones + D.T*inv_A*C
print(Rank()(det))

print(inv_A - inv_A*C*Inverse()(det)*D.T*inv_A)
print(Inverse()(A + C*D.T))

