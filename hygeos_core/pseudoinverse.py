import numpy as np
from numpy.linalg import inv


def pseudoinverse(A):
    """
    Calculate the pseudoinverse of array A over the last 2 axes
    (broadcasting the first axes)
    A* = ((A'.A)^(-1)).A'
    where X' is the transpose of X and X^-1 is the inverse of X

    shapes: A:  [...,i,j]
            A*: [...,j,i]
    """

    # B = A'.A (with broadcasting)
    B = np.einsum("...ji,...jk->...ik", A, A)

    # (B^-1).A' (with broadcasting)
    pA = np.einsum("...ij,...kj->...ik", inv(B), A)

    return pA


def weighted_pseudoinverse(A, W):
    """
    Calculate the weighted pseudoinverse of array A over the last 2 axes
    (broadcasting the first axes)
    W is the weight matrix (diagonal)
    A* = ((A'.W.A)^(-1)).A'.W
    """
    assert W.dtype == "float32"

    # A'.W.A
    B = np.einsum("...ji,...jk,...kl->...il", A, W, A)

    # (B^-1).A'.W
    pA = np.einsum("...ij,...kj,...kl->...il", inv(B), A, W)

    return pA
