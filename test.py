import numpy as np
import matrixstitcher as mats
import matrixstitcher.backend as B
from matrixstitcher.transform import *


m = mats.Matrix([[2, 3], [1, 4]])
result = m.apply([RowSwap(1, 2), ColumnTransform(1, 2, 2)], display=True)
result = result.apply([RowSwap(1, 2), ColumnTransform(2, 1.3, 2)], display=True)
result = result.apply(RowTransform(1, 0.2, 2))
print('eager result:\n', result)
print('lazy result:\n', m.forward())

m.refresh()
print('rank\n', Rank()(result))


# basic operations
print(m)
print(m + 1)
print(1 + m)
print(m * 2)
print(2 * m)
print(m - 2)
print(2 - m)
print(m / 2)