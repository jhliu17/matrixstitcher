import numpy as np
import matrixstitcher as mats 
from matrixstitcher.transform import *


# Edit your matrix to be factorized, here are two examples
# A = [[0, 0, 9, 1], [3, 0, 5, 2], [6, 1, 0, 2], [1, 1, 1, 1]]
A = [[1, 2, -3, 4], [4, 8, 12, -8], [2, 3, 2, 1], [-3, -1, 1, -4]]
display_execution = True


def lu_factorization(input_matrix):
    if not input_matrix.square:
        raise Exception('Plz input a square matrix')

    row_num = input_matrix.rows
    L = mats.Matrix(np.zeros(input_matrix.shape))
    P = mats.Matrix(np.eye(row_num))

    for i in range(1, row_num):
        
        # check and change the pivot
        non_zero = i
        while input_matrix[non_zero, i].to_scalar() == 0.0:
            non_zero += 1
            if non_zero > row_num:
                raise Exception('cannot be fatorized')

        if non_zero != i:
            row_swap = RowSwap(i, non_zero)
            input_matrix = row_swap(input_matrix)
            P = row_swap(P)
            L = row_swap(L)
        
        # reduce row echelon form
        for j in range(i+1, row_num+1):
            k = input_matrix[j, i] / input_matrix[i, i]
            k = k.to_scalar()
            if k != 0.0:
                input_matrix = input_matrix.apply(RowTransform(i, -k, j))
                L[j, i] = k # generate L's element
    
    L = L + np.eye(L.rows)
    U = input_matrix
    return P, L, U


if __name__ == '__main__':
    A = mats.Matrix(A, dtype=np.float)
    P, L, U = lu_factorization(A)

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