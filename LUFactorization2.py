import numpy as np
import matrixstitcher as mats 
from matrixstitcher.transform import *


# Edit your matrix to be factorized, here are two examples
# A = [[0, 0, 9, 1], [3, 0, 5, 2], [6, 1, 0, 2], [1, 1, 1, 1]]
A = [[1, 2, -3, 4], [4, 8, 12, -8], [2, 3, 2, 1], [-3, -1, 1, -4]]
display_execution = True


if __name__ == '__main__':
    A = mats.Matrix(A)
    P, L, U = LUFactorization()(A)

    print('\n', '*'*80, sep='')
    print('Execution Path:')
    reduced_row_echelon_form = A.forward(display=display_execution)
    
    print('\n', '*'*80, sep='')
    print('The LU Factorization result:')
    print('P:\n', P)
    print('L:\n', L)
    print('U:\n', U)

    print('\n', '*'*80, sep='')
    print('Proof P * A = L * U:')
    print('P * A:\n', P * A)
    print('L * U:\n', L * U)