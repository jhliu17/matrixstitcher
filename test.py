import matrixstitcher as mats
import matrixstitcher.transform as T


data = [[1, 2], [3, 5]]
data1 = [[3, 2], [3, 5]]
a = mats.Matrix(data)
b = mats.Matrix(data1)

with mats.TransformTape():
    c = a * b

print(c, c.transforms())
print(a.transforms())
print(b.transforms())