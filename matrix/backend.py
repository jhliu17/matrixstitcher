import numpy as np
import matrix.function as F 
from functools import reduce


class Matrix:
    '''
    A base object of matrix
    '''
    def __init__(self, data, dtype=np.float):
        try:
            assert isinstance(data, (list, tuple, np.ndarray))
        except:
            raise Exception('data must be a list or tuple')
        try:
            if isinstance(data, np.ndarray):
                self.matrix = data
            else:
                self.matrix = np.array(data, dtype=dtype)
        except:
            raise Exception('data can not be converted to matrix')
        try:
            assert len(self.matrix.shape) in (1, 2)
        except:
            raise Exception('only support 1 dimensional vector or 2-dimensional matrix not support tensor')

        # auto-determined
        self._origin_data = data
        self._dtype = dtype
        if len(self.matrix.shape) == 2:
            self.rows, self.columns = self.matrix.shape
        else:
            self.rows, self.columns = self.matrix.shape[0], 1
            self.matrix = np.reshape(self.matrix, [self.rows, self.columns])
        self.square = True if self.rows == self.columns else False
        self.shape = (self.rows, self.columns)
        self._direction = {'row': 0, 'column': 1}

        # handy-copy
        self._elementary_tape = [[], []]

    def refresh(self):
        self.matrix = np.array(self._origin_data, dtype=self._dtype)
        self._elementary_tape = [[], []]

    def get_origin(self):
        return np.array(self._origin_data, dtype=self._dtype)
    
    def reshape(self, shape):
        assert isinstance(shape, (list, tuple))
        assert len(shape) == 2

        self.matrix = self.matrix.reshape(shape)
        self.rows, self.columns = self.matrix.shape
        return self.matrix

    def row_transform(*args):
        return F.row_transform(*args)

    def column_transform(*args):
        return F.column_transform(*args)

    def row_swap(*args):
        return F.row_swap(*args)

    def column_swap(*args):
        return F.column_swap(*args)

    def row_mul(*args):
        return F.row_mul(*args)

    def column_mul(*args):
        return F.column_mul(*args)

    def __repr__(self):
        return self.matrix.__repr__()

    def __getitem__(self, key):
        key = index_mechanism(*key)
        data = self.matrix.__getitem__(key)
        if not isinstance(data, np.ndarray):
            data = np.array(data).reshape(1)

        return Matrix(data, dtype=self.matrix.dtype)

    def __setitem__(self, key, value):
        key = index_mechanism(*key)
        self.matrix.__setitem__(key, value)

    def __add__(self, other):
        if isinstance(other, Matrix):
            result = self.matrix + other.matrix
            return Matrix(result, dtype=result.dtype)
        else:
            result = self.matrix + Matrix(other)
            return Matrix(result, dtype=result.dtype)
    
    def __mul__(self, other):
        if isinstance(other, Matrix):
            result = self.matrix @ other.matrix
            return Matrix(result, dtype=result.dtype)
        elif isinstance(other, (int, float)):
            result =  self.matrix * other
            return Matrix(result, dtype=result.dtype)
        else:
            raise Exception('no defination')
    
    def __sub__(self, other):
        if isinstance(other, Matrix):
            result = self.matrix - other.matrix
            return Matrix(result, dtype=result.dtype)
        else:
            result = self.matrix - Matrix(other)
            return Matrix(result, dtype=result.dtype)

    @property
    def T(*args):
        return F.transpose(*args)

    def update_tape(self, transform_method, *args, **kwargs):
        from matrix.transform import __support_tape__
        assert transform_method in __support_tape__

        direction = 'row' if 'row' in transform_method.lower() else 'column'
        size = self.shape[self._direction[direction]]

        elementary = Matrix(np.eye(size), dtype=self._dtype)
        elementary = getattr(F, transform_method)(elementary, *args, **kwargs)
        self._elementary_tape[self._direction[direction]].append(elementary)

    def get_elementary(self):
        return self._elementary_tape[0][::-1], self._elementary_tape[1]

    def forward(self):
        foward_tape = self._elementary_tape[0][::-1] + [self] + self._elementary_tape[1]
        return reduce(lambda x, y: x * y, foward_tape)

        


def index_mechanism(*key):
    key = tuple(i - 1 if not isinstance(i, slice) else slice_mechanism(i) for i in key)
    return key


def slice_mechanism(key: slice):
    start = key.start - 1 if key.start is not None else None
    stop = key.stop - 1 if key.stop is not None else None
    step = key.step
    return slice(start, stop, step)


def apply_pipeline(matrix: Matrix, pipeline, display=False):
    '''
    A list or tuple of tranforms to apply on the input matrix.
    '''
    assert isinstance(pipeline, (list, tuple))
    from matrix.transform import Transform
    if display:
        print('-> Origin matrix:\n{}'.format(matrix))
        for idx, p in enumerate(pipeline, 1):
            assert isinstance(p, Transform)
            matrix = p(matrix)
            transform_template = '{}{}'.format(p.__class__.__name__, 
                                        p._args + tuple('{}={}'.format(i, p._kwargs[i]) for i in p._kwargs))
            print('-> Stage {}, {}:\n{}'.format(idx, transform_template, matrix))
    else:
        for p in pipeline:
            assert isinstance(p, Transform)
            matrix = p(matrix)
    return matrix


def copy_matrix(matrix: Matrix):
    new_matrix = Matrix(np.copy(matrix.matrix), dtype=matrix._dtype)
    new_matrix._elementary_tape = matrix._elementary_tape
    return new_matrix