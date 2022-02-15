"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import numpy as np
import scipy.sparse as sp

from solver import Solution

topk = 50


class Kernel:
    """
        Kernel matrix, which measures the similarities
        between vertices in the kernel space
    """

    @classmethod
    def sparsify(cls, K, epsilon: float, eta: int):
        """
        Regarding to sparsifying the Laplacian matrix, more details refered to,
            "Parallel Spectral Clustering in Distributed Systems, TPAMI 2011".

        Parameters
        ----------
        K: sparse matrix, the Kernel matrix
        epsilon : the prespecified threshold to zero-out kernel matrix
        eta : the number of nearest neighbor to keep in kernnel matrix

        Returns
        -------
        K: sparse matrix, the sparsified version
        """
        assert K is not None, "K is None"
        assert eta < K.shape[1], "eta is larger than the dimension"
        if epsilon == 0. and eta == 0.:
            return K
        if not isinstance(K, np.ndarray):
            K = K.toarray()

        # drop values lower than epsilon
        if epsilon > 0.:
            assert eta > 0, "epsilon must collaborate with eta"
            K[K < epsilon] = 0.

        # keep top-eta nearest neighbors
        if eta > 0:
            kk = np.asarray(np.argpartition(np.abs(K), -eta)[:, -eta:])
            nn = np.asarray(K[np.arange(K.shape[0])[:, np.newaxis], kk])

            ix = nn.nonzero()  # index for non-zero entries
            vals, rows, cols = nn[ix].flatten(), ix[0].flatten(), kk[ix].flatten()
            K = sp.csc_matrix((vals, (rows, cols)), shape=K.shape, dtype=K.dtype)

        return K

    @classmethod
    def kernel(cls, u, R, scale: float, epsilon: float, eta: int):
        raise NotImplementedError("kernel method is not implemented.")

    @classmethod
    def shift(cls, R, shift: float):
        raise NotImplementedError("shift method is not implemented.")

    @classmethod
    def build(cls, name: str):
        kernel = {"Laplacian": Laplacian(), "Covariance": Covariance()}
        return kernel.get(name, Laplacian())


class Laplacian(Kernel):
    """
        Graph Topology-based Kernel Matrix,
        where the similary is determined by the Affinity matrix of the graph
    """

    @classmethod
    def kernel(cls, u, R: sp.csr_matrix, scale: float, epsilon: float, eta: int):
        """
        Compute the weighted kernel matrix

        Parameters
        ----------
        u : int or list, the column(s) in a chunk
        R : the Affinity matrix, or in RecSys user-item rating matrix
        scale : the scaling parameter of the kernel matrix, the default value is 0.75
        epsilon : the prespecified threshold to zero-out kernel matrix
        eta : the number of nearest neighbor to keep in kernnel matrix

        Returns
        -------
        L : the kernel matrix, a measure for each edge over graphs
        """
        # R: affinity matrix, De: degree matrix of edges, Dv: degree matrix of vertices
        R = R.tocsc().astype(np.float32)
        Dv = np.asarray(R.sum(axis=0)).flatten()
        De = np.asarray(R.sum(axis=1)).flatten()

        # sR: scaled affinity matrix
        sDv = 1. / np.maximum(np.power(Dv, scale / 2.), 1.)
        sDe = 1. / np.maximum(np.power(De, scale / 2.), 1.)
        sR = sp.diags(sDe) * R * sp.diags(sDv)

        # K: dense kernelized matrix
        K: sp.csc_matrix = sR[:, u].T * sR

        # K: sparse kernelized matrix
        K = cls.sparsify(K, epsilon, eta)
        return K

    @classmethod
    def shift(cls, K: sp.csr_matrix, shift: float):
        K = K - sp.diags(K.diagonal()) if shift == 0. else K + sp.diags(shift - K.diagonal())
        return K


class Covariance(Kernel):
    """
        Covariance-based Kernel Matrix,
        where the similary is determined by the correlations between vertices
    """

    @classmethod
    def kernel(cls, u, R: sp.csr_matrix, scale: float, epsilon: float, eta: int):
        """
        Compute the weighted kernel matrix

        Parameters
        ----------
        u : int or list, the column(s) in a chunk
        R : the Affinity matrix, or in RecSys user-item rating matrix
        scale : the scaling parameter of the kernel matrix, the default value is 0.75
        epsilon : the prespecified threshold to zero-out kernel matrix
        eta : the number of nearest neighbor to keep in kernnel matrix

        Returns
        -------
        L : the kernel matrix, a measure for each edge over graphs
        """
        # R: affinity matrix, De: degree matrix of edges, Dv: degree matrix of vertices
        R = R.tocsc().astype(np.float32)
        Dv = np.asarray(R.sum(axis=0)).flatten()
        De = np.asarray(R.sum(axis=1)).flatten()

        # sR: scaled affinity matrix
        sDv = 1. / np.maximum(np.power(Dv, scale / 2.), 1.)
        sDe = 1. / np.maximum(np.power(De, scale / 2.), 1.)
        sR = sp.diags(sDe) * R * sp.diags(sDv)

        # K: dense kernelized matrix
        K: sp.csc_matrix = sR[:, u].T * sR

        # K: sparse kernelized matrix
        K = cls.sparsify(K, epsilon, eta)
        return K

    @classmethod
    def shift(cls, K: sp.csr_matrix, shift: float):
        if shift == 0.:
            return K

        K = K + sp.diags(shift * np.ones_like(K.diagonal()))
        return K


class Filter:
    """
        Spectral Graph Matrix Completion
    """

    @classmethod
    def filter(cls, signal: np.ndarray, solut: Solution, sDv: np.ndarray):
        """
        Graph filter

        Parameters
        ----------
        signal : ndarray, i.e., graph signal for a user or rating vetor
        solut : Solution, including eigenvalues(V), eigenvectors(U), and filter matrix(H)
        sDv : scaled diagonal degree vector

        Returns
        -------
        out : list of integer, top-N recommendations
        """
        H = solut.H
        r = signal / sDv[np.newaxis, :]

        # filter the signals on the graph vertex domain
        ans = np.matmul(r, H) * sDv
        ans = np.where(signal == 1., -np.inf, ans)

        # topk recommendation
        out = np.argsort(ans)[:, -topk:][:, ::-1]
        return out

    @classmethod
    def build(cls, name: str):
        filters = {"TRfilter": TRfilter(), "LPfilter": LPfilter(),
                   "LRwfilter": LRwfilter(), "LICfilter": LICfilter(),
                   "LDPfilter": LDPfilter()}
        return filters.get(name, Filter())


class TRfilter(Filter):
    """
        Tikhonov regularization filter, the inverse of the kernel matrix
    """

    @classmethod
    def filter(cls, signal: np.ndarray, solut: Solution, sDv: np.ndarray):
        # scale down by its degree, in RecSys, to address the popularity bias
        H = solut.H
        r = signal / sDv[np.newaxis, :]

        # filter low-band signals, followed by scaling up, to address the popularity bias
        ans = np.matmul(r, -H) / (np.diag(H) / sDv)
        ans = np.where(signal == 1., -np.inf, ans)

        # topk recommendation
        out = np.argsort(ans)[:, -topk:][:, ::-1]
        return out


class LPfilter(Filter):
    """
        Low-pass filter
    """

    @classmethod
    def filter(cls, signal: np.ndarray, solut: Solution, sDv: np.ndarray):
        # scale down by its degree, in RecSys, to address the popularity bias
        U, V = solut.U, solut.V
        r = signal / sDv[np.newaxis, :]

        # filter low-band signals, followed by scaling up, to address the popularity bias
        U: np.ndarray = U
        ans = np.matmul(np.matmul(r, U), U.T) * sDv
        ans = np.where(signal == 1., -np.inf, ans)

        # topk recommendation
        out = np.argsort(ans)[:, -topk:][:, ::-1]
        return out


class LRwfilter(Filter):
    """
        Low-pass One-step Random-walk filter
    """

    @classmethod
    def filter(cls, signal: np.ndarray, solut: Solution, sDv: np.ndarray):
        # scale down by its degree, in RecSys, to address the popularity bias
        U, V = solut.U, solut.V
        r = signal / sDv[np.newaxis, :]

        # filter low-band signals, followed by scaling up, to address the popularity bias
        V: np.ndarray = np.minimum(V, 4.)  # maximum eigenvalues for hypergraph

        # h(lambda) = (aI - lambda) where a is a hyper-parameter
        # hereby, a - 1 = 1. / 1e-1 for random walk
        ans = np.matmul(np.matmul(r, U) * (1. + 1e-1 * V), U.T) * sDv
        ans = np.where(signal == 1., -np.inf, ans)

        # topk recommendation
        out = np.argsort(ans)[:, -topk:][:, ::-1]
        return out


class LICfilter(Filter):
    """
        Low-pass Inverse Cosine filter
    """

    @classmethod
    def filter(cls, signal: np.ndarray, solut: Solution, sDv: np.ndarray):
        # scale down by its degree, in RecSys, to address the popularity bias
        U, V = solut.U, solut.V
        r = signal / sDv[np.newaxis, :]

        # filter low-band signals, followed by scaling up, to address the popularity bias
        V = np.minimum(V, 4.)  # maximum eigenvalues for hypergraph
        # h(lambda) = 1. / cos( lambda * pi / 4 )
        # hereby, the denomenator is 8, as the maximum eigenvalue is 4.
        ans = np.matmul(np.matmul(r, U) * np.cos(np.pi * (4 - V) / 8.), U.T) * sDv
        ans = np.where(signal == 1., -np.inf, ans)

        # topk recommendation
        out = np.argsort(ans)[:, -topk:][:, ::-1]
        return out


class LDPfilter(Filter):
    """
        Low-pass Diffussion Process filter
    """

    @classmethod
    def filter(cls, signal: np.ndarray, solut: Solution, sDv: np.ndarray):
        # scale down by its degree, in RecSys, to address the popularity bias
        U, V = solut.U, solut.V
        r = signal / sDv[np.newaxis, :]

        # filter low-band signals, followed by scaling up, to address the popularity bias
        # h(lambda) = exp( -a / 2lambda )
        # hereby, a = 1. which can be tuned
        k = np.where(V < 4, np.exp(1. / (V - 4.)), 1.)
        ans = np.matmul(np.matmul(r, U) * k, U.T) * sDv
        ans = np.where(signal == 1., -np.inf, ans)

        # topk recommendation
        out = np.argsort(ans)[:, -topk:][:, ::-1]
        return out
