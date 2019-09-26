import numpy as np
import matrixstitcher.backend as B
import matrixstitcher.function as F
from matrixstitcher.backend import Matrix


__support_tape__ = [
    'row_transform', 'column_transform', 'row_swap', 'column_swap',
    'row_mul', 'column_mul', 'lu_factorization'
]


class Transform:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
    
    def build(self, matrix):
        if self._method in __support_tape__:
            new_matrix = B.copy(matrix)
            self.add_tape(matrix)
        else:
            new_matrix = B.copy(matrix, causal=False)
        return new_matrix

    def add_tape(self, matrix: Matrix):
        matrix.update_tape(self._method, *self._args, **self._kwargs)


class RowTransform(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._method = 'row_transform'
    
    def __call__(self, matrix):
        matrix = super().build(matrix)
        return getattr(F, self._method)(matrix, *self._args, **self._kwargs)
    

class ColumnTransform(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._method = 'column_transform'
    
    def __call__(self, matrix):
        matrix = super().build(matrix)
        return getattr(F, self._method)(matrix, *self._args, **self._kwargs)


class RowSwap(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._method = 'row_swap'

    def __call__(self, matrix):
        matrix = super().build(matrix)
        return getattr(F, self._method)(matrix, *self._args, **self._kwargs)


class ColumnSwap(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._method = 'column_swap'

    def __call__(self, matrix):
        matrix = super().build(matrix)
        return getattr(F, self._method)(matrix, *self._args, **self._kwargs)


class RowMul(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._method = 'row_mul'

    def __call__(self, matrix):
        matrix = super().build(matrix)
        return getattr(F, self._method)(matrix, *self._args, **self._kwargs)


class ColumnMul(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._method = 'column_mul'

    def __call__(self, matrix):
        matrix = super().build(matrix)
        return getattr(F, self._method)(matrix, *self._args, **self._kwargs)


class Transpose(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._method = 'transpose'

    def __call__(self, matrix):
        matrix = super().build(matrix)
        return getattr(F, self._method)(matrix, *self._args, **self._kwargs)


class Inverse(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._method = 'inverse'

    def __call__(self, matrix):
        matrix = super().build(matrix)
        return getattr(F, self._method)(matrix, *self._args, **self._kwargs)


class Rank(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._method = 'rank'

    def __call__(self, matrix):
        matrix = super().build(matrix)
        return getattr(F, self._method)(matrix, *self._args, **self._kwargs)


class LUFactorization(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._method = 'lu_factorization'
    
    def __call__(self, matrix):
        matrix = super().build(matrix)
        
        if not matrix.square:
            raise Exception('Please input a square matrix')

        row_num = matrix.rows
        L = Matrix(np.zeros(matrix.shape))
        P = Matrix(np.eye(row_num))

        for i in range(1, row_num):
            
            # check and change the pivot
            non_zero = i
            while matrix[non_zero, i].to_scalar() == 0.0:
                non_zero += 1
                if non_zero > row_num:
                    raise Exception('this matrix cannot be fatorized')

            if non_zero != i:
                row_swap = RowSwap(i, non_zero)
                matrix = row_swap(matrix)
                P = row_swap(P)
                L = row_swap(L)
            
            # reduce row echelon form
            for j in range(i+1, row_num+1):
                k = matrix[j, i] / matrix[i, i]
                k = k.to_scalar()
                matrix = matrix.apply(RowTransform(i, -k, j))
                L[j, i] = k # generate L's element
        
        L = L + np.eye(L.rows)
        U = matrix
        return P, L, U