import numpy as np
from functools import wraps
import matrixstitcher.backend as B
import matrixstitcher.function as F
from matrixstitcher.backend import Matrix


class Transform:
    # enabled tape
    tape_enabled = True

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._is_elementary = False
        self.tape = True

    def __call__(self, *matrix):
        matrix = self.__build(*matrix)
        result = self.perform(*matrix)
        return result

    def __build(self, *matrixs):
        return_matrix = []
        for matrix in matrixs:
            if self.tape_enabled and self.tape:
                new_matrix = B.copy(matrix)
                self.add_tape(new_matrix)
            else:
                new_matrix = B.copy(matrix)
            return_matrix.append(new_matrix)
        return return_matrix

    def add_tape(self, matrix: Matrix):
        matrix.update_tape(self, *self._args, **self._kwargs)

    def is_elementary(self):
        return self._is_elementary

    @classmethod
    def set_tape_enabled(cls, mode):
        cls.tape_enabled = mode

    @classmethod
    def is_tape_enabled(cls):
        return cls.tape_enabled

    def __repr__(self):
        string = B.get_transform_template(
            self.__class__.__name__, *self._args, **self._kwargs)
        return string


class RowTransform(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_elementary = True

    def perform(self, matrix):
        return F.row_transform(matrix, *self._args, **self._kwargs)


class ColumnTransform(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_elementary = True

    def perform(self, matrix):
        return F.column_transform(matrix, *self._args, **self._kwargs)


class RowSwap(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_elementary = True

    def perform(self, matrix):
        return F.row_swap(matrix, *self._args, **self._kwargs)


class ColumnSwap(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_elementary = True

    def perform(self, matrix):
        return F.column_swap(matrix, *self._args, **self._kwargs)


class RowMul(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_elementary = True

    def perform(self, matrix):
        return F.row_mul(matrix, *self._args, **self._kwargs)


class ColumnMul(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_elementary = True

    def perform(self, matrix):
        return F.column_mul(matrix, *self._args, **self._kwargs)


class Transpose(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def perform(self, matrix):
        return F.transpose(matrix, *self._args, **self._kwargs)


class Inverse(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def perform(self, matrix):
        return F.inverse(matrix, *self._args, **self._kwargs)


class Rank(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def perform(self, matrix):
        return F.rank(matrix, *self._args, **self._kwargs)