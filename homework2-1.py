import numpy as np
import matrixstitcher as mats 
from matrixstitcher.transform import *


# Edit your matrix to be factorized, here are two examples
# A = [[0, 0, 9, 1], [3, 0, 5, 2], [6, 1, 0, 2], [1, 1, 1, 1]]
A = [[1, 4, 5], [4, 18, 26], [3, 16, 30]]
display_execution = True


if __name__ == '__main__':
    A = mats.Matrix(A, dtype=np.float)
    P, L, U = LUFactorization()(A)

    print('\n', '*'*80, sep='')
    print('Execution Path:\n')
    reduced_row_echelon_form = A.forward(display=display_execution)
    
    print('\n', '*'*80, sep='')
    print('The LU Factorization result:\n')
    print('P:\n', P, sep='')
    print('L:\n', L, sep='')
    print('U:\n', U, sep='')

    print('\n', '*'*80, sep='')
    print('Proof P * A = L * U:\n')
    print('P * A:\n', P * A, sep='')
    print('L * U:\n', L * U, sep='')