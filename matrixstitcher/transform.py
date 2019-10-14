import numpy as np
from functools import wraps
import matrixstitcher.backend as B
import matrixstitcher.function as F
from matrixstitcher.backend import Matrix


__support_tape__ = [
    'row_transform', 'column_transform', 'row_swap', 'column_swap',
    'row_mul', 'column_mul'
] # the provided elementary transformations


class Transform:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self.method = None
        self.tape = True
    
    def build(self, *matrixs):
        return_matrix = []
        for matrix in matrixs:
            if self.tape:
                new_matrix = B.copy(matrix)
                # self.add_tape(matrix)
                self.add_tape(new_matrix)
            else:
                # new_matrix = B.copy(matrix, causal=False)
                new_matrix = B.copy(matrix)
            return_matrix.append(new_matrix)
        if len(matrixs) > 1:
            return return_matrix
        else: 
            return return_matrix[0]

    def add_tape(self, matrix: Matrix):
        if self.method is not None:
            matrix.update_tape(self.method, *self._args, **self._kwargs)


class RowTransform(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, matrix):
        matrix = super().build(matrix)
        return F.row_transform(matrix, *self._args, **self._kwargs)
    

class ColumnTransform(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, matrix):
        matrix = super().build(matrix)
        return F.column_transform(matrix, *self._args, **self._kwargs)


class RowSwap(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, matrix):
        matrix = super().build(matrix)
        return F.row_swap(matrix, *self._args, **self._kwargs)


class ColumnSwap(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, matrix):
        matrix = super().build(matrix)
        return F.column_swap(matrix, *self._args, **self._kwargs)


class RowMul(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, matrix):
        matrix = super().build(matrix)
        return F.row_mul(matrix, *self._args, **self._kwargs)


class ColumnMul(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, matrix):
        matrix = super().build(matrix)
        return F.column_mul(matrix, *self._args, **self._kwargs)


class Transpose(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, matrix):
        matrix = super().build(matrix)
        return F.transpose(matrix, *self._args, **self._kwargs)


class Inverse(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, matrix):
        matrix = super().build(matrix)
        return F.inverse(matrix, *self._args, **self._kwargs)


class Rank(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, matrix):
        matrix = super().build(matrix)
        return F.rank(matrix, *self._args, **self._kwargs)


class LUFactorization(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tape = False
    
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
                    raise Exception('This matrix cannot be fatorized')

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


class LeastSquareTech(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tape = False
        self.parameter = None
        self.error = None
    
    def __call__(self, X, y):
        X, y = super().build(X, y)
        
        self.parameter = Inverse()(X.T * X) * X.T * y
        self.error = (self.predict(X) - y).T * (self.predict(X) - y)
        return self.parameter, self.error

    def predict(self, X):
        X = super().build(X)
        return X * self.parameter