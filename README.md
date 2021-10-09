# Matrix Stitcher

An elementary-transform tape system for matrix analysis education purpose which combines the advantage of eager and lazy execution. It runs dynamically and automatically tracks your transformations applied to the original matrix. It is a naive final project for Matrix Analysis and Applications course in UCAS including `PLU Factorization` (Gaussian elimination), `QR Factorization` (GramSchmidt, HouseHolder, and Givens).


## Features
- Intuitive implementation for education purpose
- Index from 1 which is consistent with the mathematical declaration
- Dynamicaly track the transformations applied to the orgin matrix
- Easily extend your transforms and methods


## Quick Guide

```python
import numpy as np
import matrixstitcher as mats 
from matrixstitcher.method import LUFactorization

# you can define matrix using int, float, list, 
# tuple and numpy array
A = [[1, 2, -3], 
     [4, 8, 12], 
     [2, 3, 2]] 
     
A = mats.Matrix(A, dtype=np.float) # get the matrix object from MatrixStitcher

with mats.TransformTape(): # tape all the transforms
    P, L, U = LUFactorization()(A) # apply LU Factorization on matrix A and get the factorization results

# show the execution state of the matrix A
print('Execution Path:')
result = A.forward(display=True)
```

Here is the execution result:
```
Execution Path:
-> Origin matrix:
array([[ 1.,  2., -3.],
       [ 4.,  8., 12.],
       [ 2.,  3.,  2.]])

-> Stage 1, RowTransform(1, -4.0, 2):
array([[ 1.,  2., -3.],
       [ 0.,  0., 24.],
       [ 2.,  3.,  2.]])

-> Stage 2, RowTransform(1, -2.0, 3):
array([[ 1.,  2., -3.],
       [ 0.,  0., 24.],
       [ 0., -1.,  8.]])

-> Stage 3, RowSwap('i=2', 'j=3'):
array([[ 1.,  2., -3.],
       [ 0., -1.,  8.],
       [ 0.,  0., 24.]])
```

If you don't want to track the transformations, applying your method outside the context of `TransformTape()`.
```python
import matrixstitcher as mats 

# construct data ...

P, L, U = LUFactorization()(A)

A.forward(display=True)

# Output
# Execution Path:
# -> Origin matrix:
# array([[ 1.,  2., -3.],
#        [ 4.,  8., 12.],
#        [ 2.,  3.,  2.]])
```
You can get the same factorization results P, L, U. But the transformations would not be recorded into A's tape history.


## Extend Transform and Method

You can esaily extend your transform and method. Here are two simple examples.
```python
import numpy as np
import matrixstitcher.transform as T
from matrixstitcher.method import Method
from matrixstitcher.transform import Tranform


class Rank(Transform):
    def __init__(self, *args, **kwargs):
        '''
        In order to represent your transform correctly, you must 
        tell the base object the parameters you have used.
        '''
        super().__init__(*args, **kwargs)

    def perform(self, matrix): # you must finish `perform` function
        return np.linalg.matrix_rank(matrix.matrix)


class LeastSquareTech(Method):
    def __init__(self):
        '''
        A `Method` can be seen as a container of several
        `Transform`s
        '''
        super().__init__()
        self.tape = False # setting whether the transformations 
                          # used under this method would be tracked 
        self.parameter = None
        self.error = None
    
    def perform(self, X, y): # you must finish `perform` function
        self.parameter = T.Inverse()(X.T * X) * X.T * y
        self.error = (self.predict(X) - y).T * (self.predict(X) - y)
        return self.parameter, self.error

    def predict(self, X):
        return X * self.parameter
```
