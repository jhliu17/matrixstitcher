import numpy as np
import matrix.backend as B
import matrix.function as F
from matrix.backend import Matrix


__support_tape__ = [
    'row_transform', 'column_transform', 'row_swap', 'column_swap',
    'row_mul', 'column_mul'
]


class Transform:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
    
    def __call__(self, matrix):
        newmatrix = B.copy_matrix(matrix)
        if self._method in __support_tape__:
            self.add_tape(matrix)
        return newmatrix

    def add_tape(self, matrix: Matrix):
        matrix.update_tape(self._method, *self._args, **self._kwargs)


class RowTransform(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._method = 'row_transform'
    
    def __call__(self, matrix):
        new_matrix = super().__call__(matrix)
        
        return getattr(F, self._method)(new_matrix, *self._args, **self._kwargs)


class ColumnTransform(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._method = 'column_transform'
    
    def __call__(self, matrix):
        new_matrix = super().__call__(matrix)
        
        return getattr(F, self._method)(new_matrix, *self._args, **self._kwargs)


class RowSwap(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._method = 'row_swap'
    
    def __call__(self, matrix):
        new_matrix = super().__call__(matrix)
        
        return getattr(F, self._method)(new_matrix, *self._args, **self._kwargs)


class ColumnSwap(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._method = 'column_swap'
    
    def __call__(self, matrix):
        new_matrix = super().__call__(matrix)
        
        return getattr(F, self._method)(new_matrix, *self._args, **self._kwargs)


class RowMul(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._method = 'row_mul'
    
    def __call__(self, matrix):
        new_matrix = super().__call__(matrix)
        
        return getattr(F, self._method)(new_matrix, *self._args, **self._kwargs)


class ColumnMul(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._method = 'column_mul'
    
    def __call__(self, matrix):
        new_matrix = super().__call__(matrix)
        
        return getattr(F, self._method)(new_matrix, *self._args, **self._kwargs)


class Transpose(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._method = 'transpose'
    
    def __call__(self, matrix):
        new_matrix = super().__call__(matrix)
        return getattr(F, self._method)(new_matrix, *self._args, **self._kwargs)