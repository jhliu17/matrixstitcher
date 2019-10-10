# Matrix Stitcher (under development)

A matrix wrapper of numpy for educational purposes in which combine the advantage of eager and lazy execution. It run dynamically and automatically track your elementary transformations applied on the origin matrix.

- Intuitive implementation for educational purposes 
- Index from 1 which is consistent with the mathematical declaration
- Dynamicaly track the elementary transformations applied on the orgin matrix
- Easily extend your operation

## Quick Guide
```python
import matrixstitcher as mats 
from matrixstitcher.transform import LUFactorization

# you can define matrix using int, float, list, 
# tuple and numpy array
A = [[1, 2, -3, 4], 
     [4, 8, 12, -8], 
     [2, 3, 2, 1], 
     [-3, -1, 1, -4]] 
     
A = mats.Matrix(A) # get the matrix object from MatrixStitcher
P, L, U = LUFactorization()(A) # apply LU Factorization on matrix A and get the results

# show the transformations applied on the matrix A
print('Execution Path:')
result = A.forward(display=True)
```
Here is the execution result:
```
Execution Path:

-> Origin matrix:
array([[ 1.,  2., -3.,  4.],
       [ 4.,  8., 12., -8.],
       [ 2.,  3.,  2.,  1.],
       [-3., -1.,  1., -4.]])

-> Stage 1, RowTransform(1, -4.0, 2):
array([[  1.,   2.,  -3.,   4.],     
       [  0.,   0.,  24., -24.],     
       [  2.,   3.,   2.,   1.],     
       [ -3.,  -1.,   1.,  -4.]])    

-> Stage 2, RowTransform(1, -2.0, 3):
array([[  1.,   2.,  -3.,   4.],     
       [  0.,   0.,  24., -24.],     
       [  0.,  -1.,   8.,  -7.],     
       [ -3.,  -1.,   1.,  -4.]])    

-> Stage 3, RowTransform(1, 3.0, 4): 
array([[  1.,   2.,  -3.,   4.],     
       [  0.,   0.,  24., -24.],     
       [  0.,  -1.,   8.,  -7.],     
       [  0.,   5.,  -8.,   8.]])    

-> Stage 4, RowSwap(2, 3):      
array([[  1.,   2.,  -3.,   4.],
       [  0.,  -1.,   8.,  -7.],
       [  0.,   0.,  24., -24.],
       [  0.,   5.,  -8.,   8.]])

-> Stage 5, RowTransform(2, 5.0, 4):
array([[  1.,   2.,  -3.,   4.],
       [  0.,  -1.,   8.,  -7.],
       [  0.,   0.,  24., -24.],
       [  0.,   0.,  32., -27.]])

-> Stage 6, RowTransform(3, -1.3333333333333333, 4):
array([[  1.,   2.,  -3.,   4.],
       [  0.,  -1.,   8.,  -7.],
       [  0.,   0.,  24., -24.],
       [  0.,   0.,   0.,   5.]])
```
