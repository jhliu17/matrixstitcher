# Matrix Stitcher (under development)

A matrix wrapper of numpy for educational purposes in which combine the advantage of Pytorch and Keras. It run dynamically and automatically trace your elementary transformations on the origin matrix.

- Intuitive implementation for educational purposes 
- Index from 1 corresponding to mathematical declaration
- Dynamicaly trace the elementary transformations on the orgin matrix
- Easily entend operation

## Quick Guide
```python
import matrixstitcher as mats 
from matrixstitcher.transform import LUFactorization

A = [[1, 2, -3, 4], 
     [4, 8, 12, -8], 
     [2, 3, 2, 1], 
     [-3, -1, 1, -4]]

A = mats.Matrix(A)
P, L, U = LUFactorization()(A)

print('Execution Path:')
result = A.forward(display=display_execution)
```