import numpy as np
import matrix as mat
import matrix.backend as B
from matrix.transform import *


m = mat.Matrix([[2, 3], [1, 4]])
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
result = mat.apply(m, [RowSwap(1, 2), ColumnTransform(1, 2, 2)], display=True)
result = mat.apply(result, [RowSwap(1, 2), ColumnTransform(1, 2, 2)], display=True)
print(m.forward())
print(result)