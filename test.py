import numpy as np
import matrixstitcher as mats
import matrixstitcher.backend as B
from matrixstitcher.transform import *


m = mats.Matrix([[2, 3], [1, 4]])
# print(m)
# m.column_swap(1, 2)
# print(m)
# m.row_swap(1, 2)
# print(m)
# m.row_transform(1, 0.5, 2)
# print(m)
# m.column_transform(1, 0.5, 2)
# print(m)
# m.refresh()
# print(m)
# m.get_origin()
# print(m)
# print(m[1, :])
# print(m[1, 1])
# m[1, :] = [0.4, 0.9]
# print(m)
# m.row_mul(1, 3.4)
# print(m)
# m.row_transform(1, 3.4, 1)
# print(m)
# m.T
# print(m)
# print(ColumnTransform(1, 2, 2)(m))
# print(m)
# print(m.forward())

m.refresh()
result = m.apply([RowSwap(1, 2), ColumnTransform(1, 2, 2)], display=True)
result = result.apply([RowSwap(1, 2), ColumnTransform(2, 1.3, 2)], display=True)
result = result.apply([RowTransform(1, 0.2, 2)])
print('eager result:', result)
# print(result.get_elementary())
print('lazy eval', m.forward(display=True))

# result = RowTransform(1, 0.5, 2)(result)
# print('eager result:', result._elementary_tape)
# result = Inverse()(m)
# print('lazy eval', result)
print('rank', Rank()(result))