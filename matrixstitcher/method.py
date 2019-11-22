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


class REF(Method):
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


class GramSchmidt(Method):
    def __init__(self):
        super().__init__()

    def perform(self, matrix):
        # linear independent checking
        rank = T.Rank()(matrix)
        if rank != matrix.columns:
            return None
        
        U = []
        u_k = matrix[:, 1]
        u_k_norm = T.L2Norm()(matrix[:, 1])
        U.append(u_k / u_k_norm)
        R = Matrix(np.zeros((matrix.columns,) * 2))
        R[1, 1] = u_k_norm

        for col in range(2, matrix.columns + 1):
            x_k = matrix[:, col]
            u_k = B.copy(x_k)

            for row, u in enumerate(U, 1):
                c_k = (x_k.T * u).to_scalar()
                R[row, col] = c_k
                u_k -= c_k * u
            
            u_k_norm = T.L2Norm()(u_k)
            U.append(u_k / u_k_norm)
            R[row + 1, col] = u_k_norm
            
        return T.Cat(axis=-1)(*U), R


class Reflector(Method):
    def __init__(self, ref2dim=1):
        super().__init__()
        self.dim = ref2dim

    def perform(self, matrix):
        row = matrix.rows
        u = Matrix(np.zeros((row, 1)))
        u[self.dim] = 1

        u = matrix - T.L2Norm()(matrix).to_scalar() * u
        reflector = Matrix(np.eye(row)) - 2 * u * u.T / (u.T * u)
        return reflector


class Rotator(Method):
    def __init__(self, i, j):
        super().__init__()
        self.i = i
        self.j = j

    def perform(self, matrix):
        row = matrix.rows
        x_i = matrix[self.i].to_scalar()
        x_j = matrix[self.j].to_scalar()
        x = (x_i ** 2 + x_j ** 2) ** 0.5
        
        c = x_i / x
        s = x_j / x

        rotator = Matrix(np.eye(row))
        rotator[self.i, self.i] = c
        rotator[self.i, self.j] = s
        rotator[self.j, self.i] = -s
        rotator[self.j, self.j] = c

        return rotator


class HouseHolder(Method):
    def __init__(self):
        super().__init__()
    
    def perform(self, matrix):
        # linear independent checking
        rank = T.Rank()(matrix)
        if rank != matrix.columns:
            return None

        R = []
        for i in range(1, matrix.columns):
            sub_matrix = matrix[i:, i:]
            r = Matrix(np.eye(matrix.rows))
            r[i:, i:] = Reflector()(sub_matrix[:, 1])
            matrix = r * matrix
            R.append(r)

        from functools import reduce
        R = reduce(lambda x, y: x * y, R[::-1])
        return R.T, matrix


class Givens(Method):
    def __init__(self):
        super().__init__()
    
    def perform(self, matrix):
        # linear independent checking
        rank = T.Rank()(matrix)
        if rank != matrix.columns:
            return None

        P = []
        for i in range(1, matrix.columns):
            sub_matrix = matrix[i:, i:]

            for j in range(2, sub_matrix.rows + 1):
                p = Matrix(np.eye(matrix.rows))
                p[i:, i:] = Rotator(1, j)(sub_matrix[:, 1])
                matrix = p * matrix
                P.append(p)
        
        from functools import reduce
        P = reduce(lambda x, y: x * y, P[::-1])
        
        return P.T, matrix


if __name__ == '__main__':
    '''
    used for unit test
    '''

    v = Matrix([-1, 2, 0, -2]) / 3
    p = Rotator(1, 2)(v)
    print(p)