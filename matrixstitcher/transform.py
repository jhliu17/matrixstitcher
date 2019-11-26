import numpy as np
from functools import wraps
import matrixstitcher.backend as B
import matrixstitcher.function as F
from matrixstitcher.backend import Matrix


class Transform:
    # enabled tape
    tape_enabled = False
    lazy_perform = False

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._is_elementary = False
        self.tape = True
        self.eager = False
        self.causal = True
        
    def __call__(self, *matrix):
        matrix = self.__build(*matrix)
        if not self.lazy_perform:
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

    @classmethod
    def set_lazy_perform(cls, mode):
        cls.lazy_perform = mode

    @classmethod
    def is_lazy_perform(cls):
        return cls.lazy_perform

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
            
            # binary operation tape
            with B.LazyPerform():
                self.other + matrix
        else:
            with B.NoTape():
                result = matrix + Matrix(self.other, matrix.dtype)
        return result


class Mul(Transform):
    def __init__(self, other):
        super().__init__(other)
        self.other = other
    
    def perform(self, matrix):
        if isinstance(self.other, Matrix):
            if self.other.shape == (1, 1):
                with B.NoTape():
                    result = matrix * self.other.to_scalar()
                    result = result.matrix
            else:
                result = matrix.matrix @ self.other.matrix
            
            # binary operation tape
            with B.LazyPerform():
                self.other * matrix
        elif isinstance(self.other, (int, float)):
            result = matrix.matrix * self.other
        else:
            with B.NoTape():
                result = matrix * Matrix(self.other, dtype=matrix.dtype)

        result = B.copy(matrix, new_value=result, causal=True)
        return result


class Sub(Transform):
    def __init__(self, other):
        super().__init__(other)
        self.other = other
    
    def perform(self, matrix):
        if isinstance(self.other, Matrix):
            result = matrix.matrix - self.other.matrix
            result = B.copy(matrix, result, causal=True)
            
            # binary operation tape
            with B.LazyPerform():
                -1 * self.other + matrix
        else:
            with B.NoTape():
                result = matrix - Matrix(self.other, dtype=matrix.dtype)
        return result


class Div(Transform):
    def __init__(self, other):
        super().__init__(other)
        self.other = other
    
    def perform(self, matrix):
        if isinstance(self.other, Matrix):
            result = matrix.matrix / self.other.matrix
            result = B.copy(matrix, result, causal=True)
        else:
            with B.NoTape():
                result = matrix / Matrix(self.other, dtype=matrix.dtype)
        return result


class SetItem(Transform):
    def __init__(self, key, value):
        super().__init__(key, value)
        self.key = key
        self.value = value
        self.eager = True

    def perform(self, matrix):
        if isinstance(self.value, Matrix):
            matrix.matrix[self.key] = self.value.matrix
        else:
            matrix.matrix[self.key] = self.value
        matrix = Matrix(matrix.matrix, dtype=matrix.dtype)


class GetItem(Transform):
    def __init__(self, key):
        super().__init__(key)
        self.key = key
        self.causal = False

    def perform(self, matrix):
        result = matrix.matrix[self.key]
        return B.copy(matrix, new_value=result, causal=True)


class AsType(Transform):
    def __init__(self, new_type):
        super().__init__(new_type)
        self.new_type = new_type
    
    def perform(self, matrix):
        return B.copy(matrix, new_type=self.new_type, causal=True)


class Reshape(Transform):
    def __init__(self, shape):
        super().__init__(shape)
        assert isinstance(shape, (list, tuple))
        assert len(shape) == 2
        self.shape = shape

    def perform(self, matrix):
        result = matrix.matrix.reshape(self.shape)
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
        return Matrix(F.inverse(matrix, *self._args, **self._kwargs), dtype=matrix.dtype)


class Rank(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def perform(self, matrix):
        return Matrix(F.rank(matrix, *self._args, **self._kwargs), dtype=matrix.dtype)


class FrobeniusNorm(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def perform(self, matrix):
        return Matrix(np.linalg.norm(matrix.matrix, ord='fro'), dtype=matrix.dtype)


class Cat(Transform):
    def __init__(self, axis: int = -1):
        super().__init__(axis=axis)
        self.axis = axis
    
    def perform(self, *matrix):
        matrix = np.concatenate([m.matrix for m in matrix], axis=self.axis)
        new_matrix = Matrix(matrix, dtype=matrix.dtype)
        return new_matrix


class L2Norm(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def perform(self, matrix):
        return Matrix(np.linalg.norm(matrix.matrix), dtype=matrix.dtype)