import numpy as np
import matrixstitcher as mats
from matrixstitcher.method import HouseHolder, Givens


v = mats.Matrix([[0, -20, -14], [3, 27, -4], [4, 11, -2]], dtype=np.float)
print('-> Origin Matrix:')
print(v)

print('*' * 80)
Q1, R1 = HouseHolder()(v)
print('-> HouseHolder Reduction:')
print('-> Q:')
print(Q1)
print('-> R:')
print(R1)
print('-> Q * R:')
print(Q1 * R1)

print('*' * 80)
Q2, R2 = Givens()(v)
print('-> Givens Reduction:')
print('-> Q:')
print(Q2)
print('-> R:')
print(R2)
print('-> Q * R:')
print(Q2 * R2)