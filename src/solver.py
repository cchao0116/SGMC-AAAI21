"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import collections

import numpy as np
import scipy.sparse as sp

# from dataclasses import dataclass
# @dataclass
# class Solution:
#     """incompatible with pickle"""
#     U: np.ndarray = np.empty(0)  # Eigenvector
#     V: np.ndarray = np.empty(0)  # Eigenvalue
#     H: np.ndarray = np.empty(0)  # Filter
# Declare the solution, U: Eigenvector, V: Eigenvalue, H: Filter
Solution = collections.namedtuple("Solution", ("U", "V", "H"))


class Solvoer:
    """
        Solver for matrix decomposition or inverse
    """

    @classmethod
    def decompose(cls, kernel, **kwargs) -> Solution:
        """

        Parameters
        ----------
        Kernel : Dense matrix or Sparse matrix, the kernel matrix
        kwargs : dict, parameters

        Returns
        -------
        out : Solution, the approximate matrix satisfying the regularizations
        """
        if isinstance(kernel, sp.csc_matrix) or isinstance(kernel, sp.csr_matrix):
            kernel = kernel.astype(np.float32).toarray()

        if kwargs.get("rank", 0) != 0:
            import scipy.sparse.linalg as la
            V, U = la.eigsh(kernel, k=kwargs["rank"], which="LA")
            return Solution(V=V, U=U, H=np.empty(0))
        else:
            import scipy.linalg as la
            la.inv(kernel, overwrite_a=True)
            return Solution(U=np.empty(0), V=np.empty(0), H=kernel)

    @classmethod
    def build(cls, name: str):
        solvers = {"Nystrom": Nystrom(), "Fowlkes": Fowlkes()}
        return solvers.get(name, Solvoer())


class Nystrom(Solvoer):
    """
        Sampling-based approach, where the approximate eigenvectors are not orthonormal.

        see,
        On the Nyström Method for Approximating a Gram Matrix for Improved Kernel-Based Learning. JMLR
    """

    @classmethod
    def decompose(cls, Kernel: sp.csc_matrix, **kwargs) -> Solution:
        assert "columns_to_sample" in kwargs, "Argument columns_to_sample is required"
        assert "rank" in kwargs, "Argument rank is required"

        s: np.ndarray = kwargs["columns_to_sample"]
        rank: int = kwargs["rank"]

        import scipy.sparse.linalg as la
        C: np.ndarray = Kernel[:, s].toarray()
        W: np.ndarray = C[s, :]
        nV, nU = la.eigsh(W, k=rank, which="LA")

        n, l = C.shape[0], C.shape[1]
        np.multiply(np.sqrt(l * 1. / n), C, out=C)
        np.multiply(nU, 1. / nV, out=nU)
        U = np.matmul(C, nU)

        V = np.multiply(n / l, nV, out=nV)
        return Solution(V=V, U=U, H=np.empty(0))


class Fowlkes(Solvoer):
    """
        Sampling-based approach, where the approximate eigenvectors are orthonormal.

        see,
        Spectral Grouping Using the Nystro¨m Method. TPAMI
    """

    @classmethod
    def decompose(cls, Kernel: sp.csc_matrix, **kwargs) -> Solution:
        assert "columns_to_sample" in kwargs, "Argument columns_to_sample is required"
        assert "rank" in kwargs, "Argument rank is required"

        s: np.ndarray = kwargs["columns_to_sample"]
        rank: int = kwargs["rank"]

        import scipy.sparse.linalg as la
        C: np.ndarray = Kernel[:, s].toarray()
        W: np.ndarray = C[s, :]
        wV, wU = la.eigsh(W, k=rank, which="LA")

        R = np.matmul(C.T, C)
        np.multiply(wU, np.power(wV, -.25), wU)  # sqrtm(W) = nU * nU.T
        R = np.matmul(np.matmul(wU.T, R), wU)
        R = np.matmul(np.matmul(wU, R), wU.T)

        # U = C * sqrtm(W) * rU * sqrtm(rV)
        rV, rU = la.eigsh(R, k=kwargs["rank"], which="LA")
        np.multiply(rU, np.power(rV, -.5), out=rU)
        U = np.matmul(np.matmul(C, wU), np.matmul(wU.T, rU))
        V = rV
        return Solution(V=V, U=U, H=np.empty(0))
