import numpy as np
import matrixstitcher.backend as B
import matrixstitcher.transform as T
from matrixstitcher.backend import Matrix


class Method:
    def __init__(self):
        self.tape = False
    
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
            pivot = matrix[i, i].to_scalar()
            for j in range(i+1, row_num+1):
                k = matrix[j, i].to_scalar()
                if k != 0.0:
                    k /= pivot
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


class RREF(Method):
    def __init__(self, tape=True):
        super().__init__()
        self.tape = tape

    def perform(self, matrix):
        row_num = matrix.rows
        col_num = matrix.columns
        i = 1
        j = 1

        while True:
            if i > row_num or j > col_num:
                break

            # check and choice the pivot
            non_zero = i
            while matrix[non_zero, j].to_scalar() == 0.0:
                non_zero += 1
                if non_zero > row_num:
                    break
            if non_zero > row_num:
                j += 1
                continue
            else:
                if non_zero != i:
                    matrix = matrix.apply(T.RowSwap(i, non_zero))
            
            # reduce row echelon form
            pivot = matrix[i, j].to_scalar()
            for ii in range(i + 1, row_num + 1):
                k = matrix[ii, j].to_scalar()
                if k != 0.0:
                    k /= pivot
                    matrix = matrix.apply(T.RowTransform(i, -k, ii))
            i += 1
            j += 1

        return matrix


class SolveLinear(Method):
    def __init__(self, tape=False):
        super().__init__()
        self.tape = tape
    
    def perform(self, A, b):
        A_rref = RREF()(A)
        argument = B.cat([A, b], axis=-1)
        argument_rref = RREF()(argument)
        argument_rank = np.sum(np.sum(argument_rref.matrix, axis=-1) > 0)
        A_rank = np.sum(np.sum(A_rref.matrix, axis=-1) > 0)

        if A_rank != argument_rank:
            print('No solution')
        else:
            raise NotImplementedError
