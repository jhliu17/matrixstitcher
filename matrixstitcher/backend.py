import numpy as np
import matrixstitcher.function as F 
from functools import reduce


class Matrix:
    '''
    A base object of matrix
    '''
    def __init__(self, data, dtype=np.float):
        try:
            assert isinstance(data, (list, tuple, np.ndarray, int, float))
        except:
            raise Exception('data must be a list or tuple')
        try:
            if isinstance(data, np.ndarray):
                self.matrix = data
            else:
                if isinstance(data, (float, int)):
                    data = [data]
                    
                if dtype is not None:
                    self.matrix = np.array(data, dtype=dtype)
                else:
                    self.matrix = np.array(data)
        except:
            raise Exception('data can not be converted to matrix')
        try:
            assert len(self.matrix.shape) in (1, 2)
        except:
            raise Exception('only support 1 dimensional vector or 2-dimensional matrix not support tensor')

        # Auto determined
        self._origin_data = data
        self._dtype = self.matrix.dtype
        if len(self.matrix.shape) == 2:
            self.rows, self.columns = self.matrix.shape
        else:
            self.rows, self.columns = self.matrix.shape[0], 1
            self.matrix = np.reshape(self.matrix, [self.rows, self.columns])
        self.square = True if self.rows == self.columns else False
        self.shape = (self.rows, self.columns)
        self._direction = {'row': 0, 'column': 1}

        # Manual operation
        '''
        The elementary tape records the whole path of applied elementary transformations on this matrix where
        two lists include applied row and column transform in time ordering respectively. All the sequence matrix
        derive from this matrix will share the elementary tape and hist and this behavior can be controled by causal
        parameter.
        '''
        self._elementary_tape = [[], []]
        self._elementary_hist = []

    def refresh(self):
        self.matrix = np.array(self._origin_data, dtype=self._dtype)
        self._elementary_tape = [[], []]
        self._elementary_hist = []

    def get_origin(self):
        return np.array(self._origin_data, dtype=self._dtype)
    
    def reshape(self, shape):
        assert isinstance(shape, (list, tuple))
        assert len(shape) == 2

        self.matrix = self.matrix.reshape(shape)
        self.rows, self.columns = self.matrix.shape
        return self.matrix

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
            result = self + Matrix(other)
            return result

    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        if isinstance(other, Matrix):
            result = self.matrix @ other.matrix
            return Matrix(result, dtype=result.dtype)
        elif isinstance(other, (int, float)):
            result =  self.matrix * other
            return Matrix(result, dtype=result.dtype)
        else:
            raise Exception('no defination')

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __sub__(self, other):
        if isinstance(other, Matrix):
            result = self.matrix - other.matrix
            return Matrix(result, dtype=result.dtype)
        else:
            result = self - Matrix(other)
            return result
    
    def __rsub__(self, other):
        return -1 * (self.__sub__(other))

    def __truediv__(self, other):
        if isinstance(other, Matrix):
            result = self.matrix / other.matrix
            return Matrix(result, dtype=result.dtype)
        else:
            result = self / Matrix(other)
            return result

    def to_scalar(self):
        if self.rows * self.columns == 1:
            return float(self.matrix)
        else:
            raise Exception('({}, {}) matrix can not be converted to a scalar'.format(self.rows, self.columns))

    @property
    def T(self):
        matrix = copy(self, causal=False)
        return F.transpose(matrix)

    def update_tape(self, transform_method, *args, **kwargs):
        from matrixstitcher.transform import __support_tape__
        
        if transform_method in __support_tape__:
            determined = transform_method.split('_')[0].lower()
            if 'row' in determined or 'column' in determined:
                direction = 'row' if 'row' in determined else 'column'
                size = self.shape[self._direction[direction]]

                elementary = Matrix(np.eye(size), dtype=self._dtype)
                elementary = getattr(F, transform_method)(elementary, *args, **kwargs)
                self._elementary_tape[self._direction[direction]].append(elementary)
                method_name = ''.join([i[0].upper() + i[1:] for i in transform_method.split('_')])
                self._elementary_hist.append(transform_template(method_name, args, kwargs))

    def get_elementary(self):
        return self._elementary_tape[0][::-1], self._elementary_tape[1]

    def forward(self, causal=True, display=False):
        left_tape, right_tape = self.get_elementary()
        
        if not display:
            foward_tape = left_tape + [self] + right_tape
            if len(foward_tape) > 1:
                result = reduce(lambda x, y: x * y, foward_tape)
            else:
                result = foward_tape[0]
        else:
            i, j = 0, 0
            result = self
            print('-> Origin matrix:\n{}\n'.format(result))
            for idx, method in enumerate(self._elementary_hist, 1):
                if 'row' in method.lower():
                    result = self._elementary_tape[self._direction['row']][i] * result
                    i += 1
                elif 'column' in method.lower():
                    result = result * self._elementary_tape[self._direction['column']][j]
                    j += 1
                else:
                    raise Exception('An illegal method in the elementary tape history')
                print('-> Stage {}, {}:\n{}\n'.format(idx, method, result))
        
        # Manual operation
        if causal:
            new_matrix = copy(self)
            new_matrix.matrix = result.matrix
            result = new_matrix

        return result
        
    def apply(self, *args, **kwargs):
        return apply_pipeline(self, *args, **kwargs)

    # have been deprecated
    # def row_transform(*args):
    #     return F.row_transform(*args)

    # def column_transform(*args):
    #     return F.column_transform(*args)

    # def row_swap(*args):
    #     return F.row_swap(*args)

    # def column_swap(*args):
    #     return F.column_swap(*args)

    # def row_mul(*args):
    #     return F.row_mul(*args)

    # def column_mul(*args):
    #     return F.column_mul(*args)

        
def index_mechanism(*key):
    key = tuple(i - 1 if not isinstance(i, slice) else slice_mechanism(i) for i in key)
    return key


def slice_mechanism(key: slice):
    start = key.start - 1 if key.start is not None else None
    stop = key.stop - 1 if key.stop is not None else None
    step = key.step
    return slice(start, stop, step)


def transform_template(p, _args, _kwargs):
    template = '{}{}'.format(p, _args + tuple('{}={}'.format(i, _kwargs[i]) for i in _kwargs))
    return template


def apply_pipeline(matrix: Matrix, pipeline, display=False):
    '''
    A list or tuple of tranforms to apply on the input matrix.
    '''
    from matrixstitcher.transform import Transform
    assert isinstance(pipeline, (list, tuple, Transform))
    done_pipeline = len(matrix._elementary_tape[0] + matrix._elementary_tape[1])
    
    if isinstance(pipeline, Transform):
        pipeline = [pipeline]

    if display:
        if done_pipeline == 0:
            print('-> Origin matrix:\n{}\n'.format(matrix))
        for idx, p in enumerate(pipeline, done_pipeline+1):
            assert isinstance(p, Transform)
            matrix = p(matrix)
            transform_template_ = transform_template(p.__class__.__name__, p._args, p._kwargs)
            print('-> Stage {}, {}:\n{}\n'.format(idx, transform_template_, matrix))
    else:
        for p in pipeline:
            assert isinstance(p, Transform)
            matrix = p(matrix)
    return matrix


def copy(matrix: Matrix, causal=True):
    new_matrix = Matrix(np.copy(matrix.matrix), dtype=matrix._dtype)

    # Manual operation
    if causal:
        new_matrix._elementary_tape = matrix._elementary_tape
        new_matrix._elementary_hist = matrix._elementary_hist
    return new_matrix