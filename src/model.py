"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import logging
import pickle
from os import path

import numpy as np
import scipy.sparse as sp

from module import Kernel, Filter
from solver import Solution, Solvoer


class Model:
    """
        The base model for kernel regularized method toward 1-bit matrix completion
    """

    def __init__(self, sc, FLAGS):
        self.sc = sc
        self.FLAGS = FLAGS

    def kernel(self, tr_data: sp.csr_matrix, kernel: Kernel):
        sc = self.sc
        FLAGS = self.FLAGS

        fKernel: str = "kernel[%05d,%05d,%.5f,%.5f]" % (FLAGS.rank, FLAGS.eta, FLAGS.epsilon, FLAGS.scale)
        if FLAGS.reuse and path.exists(fKernel):
            logging.info("Step 2. ...load kernel matrix")
            outs = pickle.load(open(fKernel, "rb"))
        else:
            logging.info("Step 2. ...build kernel matrix")

            # V: the vertex set on the graph
            tasks = np.arange(tr_data.shape[1])
            chunks: list = np.array_split(tasks, FLAGS.partitions)

            # prepare broadcast variables
            tr_data = sc.broadcast(tr_data)

            def func(u):
                return kernel.kernel(u, tr_data.value, FLAGS.scale, FLAGS.epsilon, FLAGS.eta)

            # calculate kernel matrix
            outs = sc.parallelize(chunks).map(func).collect()
            tr_data.unpersist()
            outs = sp.vstack(outs, format="csc")
            pickle.dump(outs, open(fKernel, "wb"))
        logging.info("Step 2. ...density %.4f" % (outs.nnz / np.prod(outs.shape)))

        logging.info("Step 2. ...shift diagonal entries to %.1f" % FLAGS.shift)
        outs = kernel.shift(outs, FLAGS.shift)
        return outs

    def solve(self, kernel: sp.csr_matrix, solver: Solvoer) -> Solution:
        logging.info("Step 2. ...decompose kernel")
        solut = solver.decompose(kernel, rank=self.FLAGS.rank, columns_to_sample=self.FLAGS.columns_to_sample)
        logging.info("Step 2. V=%s, U=%s, H=%s" % (solut.V.shape, solut.U.shape, solut.H.shape))
        return solut

    def train(self, tr_data: sp.csr_matrix, kernel: Kernel, solver: Solvoer) -> Solution:
        outs = self.kernel(tr_data, kernel)
        solut = self.solve(outs, solver)
        return solut

    def evaluate(self, test_tr: sp.csr_matrix, test_te: sp.csr_matrix,
                 solut: Solution, rFilter: Filter, sDv: np.ndarray):
        sc = self.sc
        FLAGS = self.FLAGS

        # V: the vertex set on the graph
        tasks = np.asarray(test_te.sum(1)).flatten().nonzero()[0]
        chunks: list = np.array_split(tasks, FLAGS.partitions)
        logging.info("Step 4. ... %d users for testing" % tasks.size)

        # prepare broadcast variables
        tr_data_b, te_data_b = sc.broadcast(test_tr), sc.broadcast(test_te)
        solut_b, sDv_b = sc.broadcast(solut), sc.broadcast(sDv)

        def func(u):
            signal = tr_data_b.value[u].toarray()  # graph signal
            pred = rFilter.filter(signal, solut_b.value, sDv_b.value)
            outs = te_data_b.value[u[:, np.newaxis], pred]
            return outs.toarray()

        # calculate prediction results
        res = sc.parallelize(chunks).map(func).collect()
        sc.stop()

        # calculate topk performance
        res = np.vstack(res).astype(np.float32)
        Pu = np.sum(res, 1).astype(np.float32)
        Tu = np.asarray(test_te[tasks].sum(1)).flatten().astype(np.float32)

        HR = np.mean(np.sign(Pu))  # Hit-Rate

        k = res.shape[1]  # topK
        vP = np.divide(Pu, k)
        PR = np.mean(vP)  # Precision

        vR = np.divide(Pu, np.minimum(Tu, k))  # satisfy, sup(RC) = 1.
        RC = np.mean(vR)  # Recall

        vF = PR + RC
        F1 = np.where(vF == 0., 0., 2. * PR * RC / vF)

        score = 1. / np.log2(np.arange(2., 2. + k))
        DCG = np.sum(res * score[np.newaxis, :], 1)
        pDCG = np.cumsum(score)[np.minimum(Tu, k).astype(np.int32) - 1]
        NDCG = np.mean(DCG / pDCG)
        logging.info("=====HR: {0:.5f}, PR: {1:.5f}, RC: {2:.5f}, F1: {3:.5f}, NDCG: {4:.5f}====="
                     .format(HR, PR, RC, F1, NDCG))
