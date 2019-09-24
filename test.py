import numpy as np
import matrix as mat


m = mat.Matrix([[2, 3], [1, 4]])
print(m)
m.column_swap(1, 2)
print(m)
m.row_swap(1, 2)
print(m)
m.row_transform(1, 0.5, 2)
print(m)
m.column_transform(1, 0.5, 2)
print(m)
m.refresh()
print(m)
m.get_origin()
print(m)
print(m[1, :])
print(m[1, 1])
m[1, :] = np.array([0.4, 0.9])
print(m)
m.row_time(1, 3.4)
print(m)
m.row_transform(1, 3.4, 1)
print(m)
m.T
print(m)
print(m*4)