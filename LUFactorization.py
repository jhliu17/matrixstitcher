import numpy as np
import matrixstitcher as mats 
from matrixstitcher.method import LUFactorization


# Edit your matrix to be factorized, here are two examples
# A = [[0, 0, 9, 1], [3, 0, 5, 2], [6, 1, 0, 2], [1, 1, 1, 1]]
A = [[1, 2, -3, 4], [4, 8, 12, -8], [2, 3, 2, 1], [-3, -1, 1, -4]]
display_execution = True


if __name__ == '__main__':
    A = mats.Matrix(A)

    with mats.TransformTape():
        P, L, U = LUFactorization()(A)
    
    print('The second column of A:\n', A[:, 2], sep='')
    print('\n', '*' * 80, sep='')
    print('Execution Path:')
    result = A.forward(display=display_execution)
    
    print('\n', '*'*80, sep='')
    print('The LU Factorization result:')
    print('P:\n', P, sep='')
    print('L:\n', L, sep='')
    print('U:\n', U, sep='')

    print('\n', '*'*80, sep='')
    print('Proof P * A = L * U:')
    print('P * A:\n', P * A, sep='')
    print('L * U:\n', L * U, sep='')