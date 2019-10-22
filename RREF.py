import numpy as np
import matrixstitcher as mats 
from matrixstitcher.transform import Rank
from matrixstitcher.method import RREF


# Edit your matrix to be factorized, here are two examples
# A = [[0, 0, 9, 1], [3, 0, 5, 2], [6, 1, 0, 2], [1, 1, 1, 1]]
A = [[0, 0, 0, 0], [4, 8, 0, -8], [0, 0, 0, 1], [0, 0, 0, 0]]
display_execution = True


if __name__ == '__main__':
    A = mats.Matrix(A)
    P = RREF()(A)
    with mats.no_tape():
        R = Rank()(A)

    print('The second column of A:\n', A[:, 2], sep='')
    print('\n', '*'*80, sep='')
    print('Execution Path:')
    result = A.forward(display=display_execution)
    
    print('\n', '*'*80, sep='')
    print('The RREF result:')
    print('P:\n', P, sep='')

    print('The Rank result:')
    print('R:', R)