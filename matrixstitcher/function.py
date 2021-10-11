import numpy as np
import matrixstitcher.backend as B


def row_transform(matrix, i: int, k: float, j: int):
    '''
    apply basic row transform:
    i-th row times k and add to j-th row
    '''
    if i > matrix.rows or j > matrix.rows:
        raise Exception('{}-th row or {}-th row out of matrix'.format(i, j))

    i, j = B.index_mechanism(i, j)
    matrix.matrix[j, :] = matrix.matrix[i, :] * k + matrix.matrix[j, :]
    matrix.update()
    return matrix


def column_transform(matrix, i: int, k: float, j: int):
    '''
    apply basic column transform:
    i-th column times k and add to j-th column
    '''
    if i > matrix.columns or j > matrix.columns:
        raise Exception('{}-th row or {}-th row out of matrix'.format(i, j))

    i, j = B.index_mechanism(i, j)
    matrix.matrix[:, j] = matrix.matrix[:, i] * k + matrix.matrix[:, j]
    matrix.update()
    return matrix


def row_swap(matrix, i: int, j: int):
    '''
    exchange i-th row and j-th row
    '''
    if i > matrix.rows or j > matrix.rows:
        raise Exception('{}-th row or {}-th row out of matrix'.format(i, j))

    i, j = B.index_mechanism(i, j)
    matrix.matrix[[i, j], :] = matrix.matrix[[j, i], :]
    matrix.update()
    return matrix


def column_swap(matrix, i: int, j: int):
    '''
    exchange i-th column and j-th column
    '''
    if i > matrix.columns or j > matrix.columns:
        raise Exception('{}-th column or {}-th column out of matrix'.format(i, j))

    i, j = B.index_mechanism(i, j)
    matrix.matrix[:, [i, j]] = matrix.matrix[:, [j, i]]
    matrix.update()
    return matrix


def row_mul(matrix, i: int, k: float):
    if i > matrix.rows:
        raise Exception('{}-th row row out of matrix'.format(i))

    i = B.index_mechanism(i)
    matrix.matrix[i, :] = matrix.matrix[i, :] * k
    matrix.update()
    return matrix


def column_mul(matrix, i: int, k: float):
    if i > matrix.columns:
        raise Exception('{}-th column out of matrix'.format(i))

    i = B.index_mechanism(i)
    matrix.matrix[:, i] = matrix.matrix[:, i] * k
    matrix.update()
    return matrix


def transpose(matrix):
    matrix.matrix = matrix.matrix.transpose()
    matrix.update()
    return matrix


def inverse(matrix):
    if matrix.square:
        try:
            inv = np.linalg.inv(matrix.matrix)
        except:
            raise Exception('Matrix is noninvertible')
        else:
            return inv
    else:
        raise Exception('Unsquare matrix is noninvertible')


def rank(matrix):
    return np.linalg.matrix_rank(matrix.matrix)
