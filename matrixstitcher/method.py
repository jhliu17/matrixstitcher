import numpy as np
import matrixstitcher.backend as B
import matrixstitcher.transform as T
from matrixstitcher.backend import Matrix


class Method:
    def __init__(self):
        self.tape = True
    
    def __call__(self, *matrix):
        matrix = self.__build(*matrix)

        if self.tape:
            result = self.perform(*matrix)
        else:
            with B.no_tape():
                result = self.perform(*matrix)
        return result
    
    def __build(self, *matrixs):
        return_matrix = []
        for matrix in matrixs:
            new_matrix = B.copy(matrix)
            return_matrix.append(new_matrix)
        return return_matrix


class LUFactorization(Method):
    def __init__(self, tape=True):
        super().__init__()
        self.tape = tape
    
    def perform(self, matrix):
        
        if not matrix.square:
            raise Exception('Please input a square matrix')

        row_num = matrix.rows
        L = Matrix(np.zeros(matrix.shape))
        P = Matrix(np.eye(row_num))

        for i in range(1, row_num):
            
            # check and change the pivot
            non_zero = i
            while matrix[non_zero, i].to_scalar() == 0.0:
                non_zero += 1
                if non_zero > row_num:
                    raise Exception('This matrix cannot be fatorized')

            if non_zero != i:
                row_swap = T.RowSwap(i, non_zero)
                matrix = row_swap(matrix)
                P = row_swap(P)
                L = row_swap(L)
            
            # reduce row echelon form
            for j in range(i+1, row_num+1):
                k = matrix[j, i] / matrix[i, i]
                k = k.to_scalar()
                matrix = matrix.apply(T.RowTransform(i, -k, j))
                L[j, i] = k # generate L's element
        
        L = L + np.eye(L.rows)
        U = matrix
        return P, L, U


class LeastSquareTech(Method):
    def __init__(self, tape=True):
        super().__init__()
        self.tape = tape
        self.parameter = None
        self.error = None
    
    def perform(self, X, y):
        self.parameter = T.Inverse()(X.T * X) * X.T * y
        self.error = (self.predict(X) - y).T * (self.predict(X) - y)
        return self.parameter, self.error

    def predict(self, X):
        return X * self.parameter