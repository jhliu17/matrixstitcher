import numpy as np
import matrixstitcher.backend as B
import matrixstitcher.function as F
from matrixstitcher.backend import Matrix


__support_tape__ = [
    'row_transform', 'column_transform', 'row_swap', 'column_swap',
    'row_mul', 'column_mul'
]


class Transform:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
    
    def __call__(self, matrix):
        if self._method in __support_tape__:
            new_matrix = B.copy(matrix)
            self.add_tape(matrix)
        else:
            new_matrix = B.copy(matrix, causal=False)
        return getattr(F, self._method)(new_matrix, *self._args, **self._kwargs)

    def add_tape(self, matrix: Matrix):
        matrix.update_tape(self._method, *self._args, **self._kwargs)


class RowTransform(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._method = 'row_transform'
    

class ColumnTransform(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._method = 'column_transform'
    

class RowSwap(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._method = 'row_swap'


class ColumnSwap(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._method = 'column_swap'


class RowMul(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._method = 'row_mul'


class ColumnMul(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._method = 'column_mul'


class Transpose(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._method = 'transpose'


class Inverse(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._method = 'inverse'


class Rank(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._method = 'rank'