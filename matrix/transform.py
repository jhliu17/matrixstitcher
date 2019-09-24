import numpy as np
import matrix.backend as B
import matrix.function as F
from matrix.backend import Matrix


class Transform(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self):
        raise NotImplementedError


class RowTransform(Transform):
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
    
    def __call__(self, matrix):
        new_matrix = B.copy_matrix(matrix)
        return F.row_transform(new_matrix, *self._args, **self._kwargs)


class ColumnTransform(Transform):
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
    
    def __call__(self, matrix):
        new_matrix = B.copy_matrix(matrix)
        return F.column_transform(new_matrix, *self._args, **self._kwargs)


class RowSwap(Transform):
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
    
    def __call__(self, matrix):
        new_matrix = B.copy_matrix(matrix)
        return F.row_swap(new_matrix, *self._args, **self._kwargs)


class ColumnSwap(Transform):
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
    
    def __call__(self, matrix):
        new_matrix = B.copy_matrix(matrix)
        return F.column_swap(new_matrix, *self._args, **self._kwargs)


class RowMul(Transform):
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
    
    def __call__(self, matrix):
        new_matrix = B.copy_matrix(matrix)
        return F.row_mul(new_matrix, *self._args, **self._kwargs)


class ColumnMul(Transform):
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
    
    def __call__(self, matrix):
        new_matrix = B.copy_matrix(matrix)
        return F.column_mul(new_matrix, *self._args, **self._kwargs)


class Transpose(Transform):
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
    
    def __call__(self, matrix):
        new_matrix = B.copy_matrix(matrix)
        return F.transpose(new_matrix, *self._args, **self._kwargs)