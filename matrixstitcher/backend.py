import numpy as np
import matrixstitcher.function as F 
from functools import reduce
from copy import deepcopy


class Matrix:
    '''
    A base object of matrix
    '''
    def __init__(self, data, dtype=np.float):
        try:
            if isinstance(data, np.ndarray):
                self.matrix = data.astype(dtype)
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
            assert len(self.matrix.shape) in (0, 1, 2)
        except:
            raise Exception('only support 1 dimensional vector or 2-dimensional matrix not support tensor')

        # Auto determined
        self._origin_data = data
        self._dtype = self.matrix.dtype
        if len(self.matrix.shape) == 2:
            self.rows, self.columns = self.matrix.shape
        elif len(self.matrix.shape) == 0:
            self.rows, self.columns = 1, 1
            self.matrix = np.reshape(self.matrix, [self.rows, self.columns])
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
        self.__elementary_tape = [[], []]
        self.__tape = []
        self.__tape_hist = []

    def refresh(self):
        self.matrix = np.array(self._origin_data, dtype=self._dtype)
        self.__elementary_tape = [[], []]
        self.__tape = []
        self.__tape_hist = []

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
        from matrixstitcher.transform import GetItem
        return GetItem(self, key)(self)   

    def __setitem__(self, key, value):
        from matrixstitcher.transform import SetItem
        return SetItem(key, value)(self)        

    def __add__(self, other):
        from matrixstitcher.transform import Add
        return Add(other)(self)

    def __radd__(self, other):
        from matrixstitcher.transform import Add
        return Add(other)(self)
    
    def __mul__(self, other):
        from matrixstitcher.transform import Mul
        return Mul(other)(self)

    def __rmul__(self, other):
        from matrixstitcher.transform import Mul
        return Mul(other)(self)
    
    def __sub__(self, other):
        from matrixstitcher.transform import Sub
        return Sub(other)(self)
    
    def __rsub__(self, other):
        from matrixstitcher.transform import Sub
        return Mul(-1)(Sub(other)(self))

    def __truediv__(self, other):
        from matrixstitcher.transform import Div
        return Div(other)(self)

    def to_scalar(self):
        if self.rows * self.columns == 1:
            return float(self.matrix)
        else:
            raise Exception('({}, {}) matrix can not be converted to a scalar'.format(self.rows, self.columns))

    @property
    def T(self):
        from matrixstitcher.transform import Transpose
        return Transpose()(self) 

    def update_tape(self, transform, *args, **kwargs):
        transform_name = transform.__class__.__name__

        if transform.is_elementary():
            determined = transform_name.lower()
            if 'row' in determined or 'column' in determined:
                direction = 'row' if 'row' in determined else 'column'
                size = self.shape[self._direction[direction]]

                elementary = Matrix(np.eye(size), dtype=self._dtype)
                with no_tape():
                    elementary = transform(elementary)
                self.__elementary_tape[self._direction[direction]].append(elementary)
        
        self.__tape.append(transform)
        self.__tape_hist.append(get_transform_template(transform_name, *args, **kwargs))

    def elementary(self):
        return self.__elementary_tape[0][::-1], self.__elementary_tape[1]
    
    def get_elementary(self):
        return self.__elementary_tape

    def get_transform_tape(self):
        return self.__tape, self.__tape_hist

    def set_elementary(self, args):
        self.__elementary_tape = args
        
    def set_transform_tape(self, *args):
        self.__tape = args[0]
        self.__tape_hist = args[1]

    def forward(self, causal=True, display=False):
        pipeline, _ = self.get_transform_tape()

        with no_tape():
            result = self.apply(pipeline, display=display, forward=True)
        
        # Manual operation
        if causal:
            new_matrix = copy(self)
            result = copy(result)
            new_matrix.matrix = result.matrix
            result = new_matrix

        return result
        
    def apply(self, *args, **kwargs):
        return apply_pipeline(self, *args, **kwargs)

    def numpy(self):
        if 1 in (self.rows, self.columns):
            return self.matrix.reshape(-1)
        else:
            return self.matrix

    def transforms(self):
        transforms, _ = self.get_transform_tape()
        return transforms

    def as_type(self, dtype):
        return Matrix(self.matrix, dtype)

        
def index_mechanism(*key):
    new_key = []
    # key = tuple(i - 1 if not isinstance(i, slice) else slice_mechanism(i) for i in key)
    for i in key:
        if isinstance(i, slice):
            new_key.append(slice_mechanism(i))
        else:
            if i > 0:
                new_key.append(i - 1)
            elif i == 0:
                raise Exception('Index from 0 is not vaild')
            else:
                new_key.append(i)
    return tuple(new_key)


def slice_mechanism(key: slice):
    start = index_mechanism(*[key.start])[0] if key.start is not None else None
    stop = index_mechanism(*[key.stop])[0] if key.stop is not None else None
    step = key.step
    return slice(start, stop, step)


def get_transform_template(transform_name, *_args, **_kwargs):
    template = '{}{}'.format(transform_name, _args + tuple('{}={}'.format(k, _kwargs[k]) for k in _kwargs))
    return template


def apply_pipeline(matrix: Matrix, pipeline, display=False, forward=False):
    '''
    A list or tuple of tranforms to apply on the input matrix.
    '''
    from matrixstitcher.transform import Transform

    assert isinstance(pipeline, (list, tuple, Transform))
    if forward:
        done_pipeline = 0
    else:
        done_pipeline = len(matrix.get_transform_tape()[0])
    
    if isinstance(pipeline, Transform):
        pipeline = [pipeline]

    if display:
        if done_pipeline == 0:
            print('-> Origin matrix:\n{}\n'.format(matrix))
        for idx, p in enumerate(pipeline, done_pipeline + 1):
            assert isinstance(p, Transform)
            matrix = p(matrix)
            transform_template = repr(p)
            print('-> Stage {}, {}:\n{}\n'.format(idx, transform_template, matrix))
    else:
        for p in pipeline:
            assert isinstance(p, Transform)
            matrix = p(matrix)
    return matrix


def copy(matrix: Matrix, new_value=None, causal=True, eager_copy=False):
    if isinstance(matrix, Matrix):
        if new_value is None:
            if not eager_copy:
                new_matrix = Matrix(np.copy(matrix.matrix), dtype=matrix._dtype)
            else:
                new_matrix = Matrix(matrix.matrix, dtype=matrix._dtype)
        else:
            new_matrix = Matrix(new_value, dtype=matrix._dtype)
        
        # Manual operation
        if causal:
            new_matrix.set_elementary(matrix.get_elementary())
            new_matrix.set_transform_tape(*matrix.get_transform_tape())
    elif isinstance(matrix, (list, tuple)):
        new_matrix = Matrix(deepcopy(matrix))
    else:
        new_matrix = Matrix(matrix)
    return new_matrix


class no_tape:
    def __enter__(self):
        from matrixstitcher.transform import Transform
        self.prev = Transform.is_tape_enabled()
        Transform.set_tape_enabled(False)
    
    def __exit__(self, *args):
        from matrixstitcher.transform import Transform
        Transform.set_tape_enabled(self.prev)


def cat(matrix_list: list, axis: int):
    matrix = np.concatenate([m.matrix for m in matrix_list], axis=axis)
    new_matrix = Matrix(matrix)
    return new_matrix