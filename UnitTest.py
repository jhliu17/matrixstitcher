import numpy as np
import matrixstitcher as mats
from matrixstitcher.method import Rotator, Reflector, HouseHold

# v = mats.Matrix([-1, 2, 0, -2], dtype=np.float) / 3
# p = Rotator(1, 2)(v)
# v = p * v
# p = Rotator(1, 4)(v)
# v = p * v
# print(p, v)

# r = Reflector()(v)
# v = r * v
# print(r, v)

v = mats.Matrix([[0, -20, -14], [3, 27, -4], [4, 11, -2]], dtype=np.float)
R, v = HouseHold()(v)
print(R)