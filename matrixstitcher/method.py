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

    def no_tape(self):
        self.tape = False
    
    def tape(self):
        self.tape = True


class LUFactorization(Method):
    def __init__(self):
        super().__init__()
    
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
    def __init__(self):
        super().__init__()
        self.parameter = None
        self.error = None
    
    def perform(self, X, y):
        self.parameter = T.Inverse()(X.T * X) * X.T * y
        self.error = (self.predict(X) - y).T * (self.predict(X) - y)
        return self.parameter, self.error

    def predict(self, X):
        return X * self.parameter


class RREF(Method):
    def __init__(self):
        super().__init__()

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
    def __init__(self):
        super().__init__()
    
    def perform(self, A, b):
        A_rref = RREF()(A)
        Arg = B.cat([A, b], axis=-1)
        Arg_rref = RREF()(Arg)

        A_rank = np.sum(np.sum(A_rref.matrix, axis=-1) > 0)
        Arg_rank = np.sum(np.sum(Arg_rref.matrix, axis=-1) > 0)

        if A_rank != argument_rank:
            print('No solution')
            return None
        elif A_rank == A.columns:
            pass
        else:
            return VectorSpace('df')


class VectorSpace(Method):
    def __init__(self, basis: Matrix, bias: Matrix = None, strict=False):
        super().__init__()
        if strict:
            rank = T.Rank()(basis)
            if rank.to_scalar() < basis.columns:
                raise Exception('Can not construct a vector space based on this basis')
        self.basis = basis
        self.bias = Matrix(np.zeros(basis.shape), dtype=basis._dtype) if bias is None else bias
    
    @property
    def dim(self):
        rref = RREF()(self.basis)
        rank = np.sum(np.sum(rref.matrix, axis=-1) > 0)
        return rank

    def perform(self, x):
        return self.basis * x + self.bias