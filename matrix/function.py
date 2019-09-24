import numpy as np 
import matrix.backend as B 


def row_transform(matrix, i: int, k: float, j: int):
    '''
    apply basic row transform: 
    i-th row times k and add to j-th row
    '''
    if i > matrix.rows or j > matrix.rows:
        raise Exception('{}-th row or {}-th row out of matrix'.format(i, j))
    
    i, j = B.index_mechanism(i, j)
    matrix.matrix[j, :] = matrix.matrix[i, :] * k + matrix.matrix[j, :]
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
    return matrix


def row_swap(matrix, i: int, j: int):
    '''
    exchange i-th row and j-th row
    '''
    if i > matrix.rows or j > matrix.rows:
        raise Exception('{}-th row or {}-th row out of matrix'.format(i, j))
    
    i, j = B.index_mechanism(i, j)
    matrix.matrix[[i, j], :] = matrix.matrix[[j, i], :]
    return matrix


def column_swap(matrix, i: int, j: int):
    '''
    exchange i-th column and j-th column
    '''
    if i > matrix.columns or j > matrix.columns:
        raise Exception('{}-th column or {}-th column out of matrix'.format(i, j))
    
    i, j = B.index_mechanism(i, j)
    matrix.matrix[:, [i, j]] = matrix.matrix[:, [j, i]]
    return matrix


def row_time(matrix, i: int, k: float):
    if i > matrix.rows:
        raise Exception('{}-th row row out of matrix'.format(i))
    
    i = B.index_mechanism(i)
    matrix.matrix[i, :] = matrix.matrix[i, :] * k
    return matrix


def column_time(matrix, i: int, k: float):
    if i > matrix.columns:
        raise Exception('{}-th column out of matrix'.format(i))
    
    i = B.index_mechanism(i)
    matrix.matrix[:, i] = matrix.matrix[:, i] * k
    return matrix


def transpose(matrix):
    matrix.matrix = matrix.matrix.transpose()
    return matrix
