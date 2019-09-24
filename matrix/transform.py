import numpy as np
import matrix.backend as B
import matrix.function as F
from matrix.backend import Matrix


class BaseTransform(object):
    def __init__(self, *args, **kwargs):
        self.args = args
    
    def __call__(self, matrix):
        new_matrix = B.copy_matrix(matrix)
        args = new_matrix + self.args
        return F.column_swap(*args)
