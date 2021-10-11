import matrixstitcher as mats
from matrixstitcher.method import REF


# Edit your matrix to be factorized, here are two examples
# A = [[0, 0, 9, 1], [3, 0, 5, 2], [6, 1, 0, 2], [1, 1, 1, 1]]
A = [[1, 19, -34], [-2, 5, 20], [2, 8, 37]]
B = [[2, -4, 4], [-4, 2, 4], [4, 4, 2]]
display_execution = True


if __name__ == '__main__':
    with mats.TransformTape():
        A = mats.Matrix(A)
        res = REF()(A)

    A.forward(display=True)
