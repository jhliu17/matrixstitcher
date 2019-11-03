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
        self.eager = False
        self.causal = True
        
    def __call__(self, *matrix):
        matrix = self.__build(*matrix)
        result = self.perform(*matrix)
        return result

    def __build(self, *matrixs):
        return_matrix = []
        for matrix in matrixs:
            if not self.eager:
                new_matrix = B.copy(matrix, causal=self.causal)
            else:
                new_matrix = matrix
            if self.tape_enabled and self.tape:
                self.add_tape(new_matrix)
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


class Add(Transform):
    def __init__(self, other):
        super().__init__(other)
        self.other = other
    
    def perform(self, matrix):
        if isinstance(self.other, Matrix):
            result = matrix.matrix + self.other.matrix
            result = B.copy(matrix, new_value=matrix.matrix + self.other.matrix, causal=True)
        else:
            with B.no_tape():
                result = matrix + Matrix(self.other, matrix._dtype)
        return result


class Mul(Transform):
    def __init__(self, other):
        super().__init__(other)
        self.other = other
    
    def perform(self, matrix):
        if isinstance(self.other, Matrix):
            result = matrix.matrix @ self.other.matrix
        elif isinstance(self.other, (int, float)):
            result = matrix.matrix * self.other
        else:
            raise Exception('no defination')

        result = B.copy(matrix, new_value=result, causal=True)
        return result


class Sub(Transform):
    def __init__(self, other):
        super().__init__(other)
        self.other = other
    
    def perform(self, matrix):
        if isinstance(self.other, Matrix):
            result = matrix.matrix - self.other.matrix
            return B.copy(matrix, result, causal=True)
        else:
            with B.no_tape():
                result = matrix - Matrix(other, dtype=matrix._dtype)
            return result


class Div(Transform):
    def __init__(self, other):
        super().__init__(other)
        self.other = other
    
    def perform(self, matrix):
        if isinstance(self.other, Matrix):
            result = matrix.matrix / self.other.matrix
            return B.copy(matrix, result, causal=True)
        else:
            with B.no_tape():
                result = matrix / Matrix(other)
            return result


class SetItem(Transform):
    def __init__(self, target, key, value):
        super().__init__(target, key, value)
        self.key = key
        self.value = value
        self.eager = True

    def perform(self, matrix):
        if isinstance(self.key, (list, tuple)):
            key = B.index_mechanism(*self.key)
        else:
            key = B.index_mechanism(*[self.key])
        matrix.matrix[key] = self.value
        matrix = Matrix(matrix.matrix, dtype=matrix._dtype)


class GetItem(Transform):
    def __init__(self, target, key):
        super().__init__(target, key)
        self.key = key
        self.causal = False

    def perform(self, matrix):
        if isinstance(self.key, (list, tuple)):
            key = B.index_mechanism(*self.key)
        else:
            key = B.index_mechanism(*[self.key])
        result = matrix.matrix.__getitem__(key)
        return B.copy(matrix, new_value=result, causal=True)


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