import numpy as np
import matrixstitcher as mats
from matrixstitcher.method import Rotator, Reflector, HouseHold, Givens, GramSchmidt


v = mats.Matrix([[0, -20, -14], [3, 27, -4], [4, 11, -2]], dtype=np.float64)
Q1, R1 = GramSchmidt()(v)
Q2, R2 = HouseHold()(v)
Q3, R3 = Givens()(v)
print(Q1 * R1)
print(Q2 * R2)
print(Q3 * R3)
print(v)