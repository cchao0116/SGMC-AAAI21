"""
@version: 1.0
@author: Chao Chen
@contact: chao.chen@sjtu.edu.cn
"""
import argparse
import logging.config
import pickle

logging.config.fileConfig('./conf/logging.conf')

import pandas as pd
import numpy as np
import scipy.sparse as sp

from os import path
from pyspark import SparkContext, SparkConf

from model import Model
from module import Kernel, Filter
from solver import Solvoer

parser = argparse.ArgumentParser(description='SGMC experiment.')
parser.add_argument('--train', action="store", dest="train",
                    help="training data file name", required=True)
parser.add_argument('--test_tr', action="store", dest="test_tr",
                    help="the fold-in data used for testing", required=True)
parser.add_argument('--test_te', action="store", dest="test_te",
                    help="the ground-truth data used for testing", required=True)
parser.add_argument('--eta', action="store", dest="eta", default=100, type=int,
                    help="the number of nearest neighbor to keep in kernnel matrix")
parser.add_argument('--rank', action="store", dest="rank", default=50, type=int,
                    help="the number of eigenvectors/eigenvalues to keep")
parser.add_argument('--columns_to_sample', action="store", dest="columns_to_sample", default=0, type=int,
                    help="the name of the solver to decompose or inverse the Kernel matrix")
parser.add_argument('--epsilon', action="store", dest="epsilon", default=0., type=float,
                    help="the prespecified threshold to zero-out kernel matrix")
parser.add_argument('--shift', action="store", dest="shift", type=float,
                    help="the diagonal shift to enjoy definiteness")
parser.add_argument('--scale', action="store", dest="scale", type=float,
                    help="the scaling parameter to control the significance of degree matrix")
parser.add_argument('--reuse', action="store_true", dest="reuse",
                    help="the indicator whether to reuse intermediate result")
parser.add_argument('--partitions', action="store", dest="partitions", type=int,
                    help="the number of the partitions of data to split ")
parser.add_argument('--sugar', action="store", dest="sugar", type=float,
                    help="the temperature to smooth the vertex degree for prediction")
parser.add_argument('--kernel', action="store", dest="kernel", type=str, default=None,
                    help="the name of the kernel matrix to depict the similarities between vertcies")
parser.add_argument('--solver', action="store", dest="solver", type=str, default=None,
                    help="the name of the solver to decompose or inverse the Kernel matrix")
parser.add_argument('--filter', action="store", dest="filter", type=str, default=None,
                    help="the name of the filter to denoise/recover the graph signal")
parser.set_defaults(reuse=False, shift=0., scale=0.75, partitions=16, sugar=0.5)
FLAGS = parser.parse_args()


def before_run(Dv):
    if FLAGS.columns_to_sample > 0:
        FLAGS.__setattr__("columns_to_sample",
                          np.argpartition(Dv, -FLAGS.columns_to_sample)[-FLAGS.columns_to_sample:])


def data_loader(fname: str, align=np.ones(2, dtype=np.int32)) -> sp.csr_matrix:
    """
    load data into a sparse matrix of bytes
    Parameters
    ----------
    fname : file name to load consisting of two elements concatenated with comma
    align : the matrix shape to align, the resulting shape must greater than this

    Returns
    -------
    outs : a sparse byte user-item matrix

    """
    df = pd.read_csv(fname)

    rows = df["uid"].values
    cols = df["sid"].values
    data = np.ones_like(rows, dtype=np.bool_)

    shape = np.maximum([np.max(rows) + 1, np.max(cols) + 1], align)
    outs = sp.csr_matrix((data, (rows, cols)), shape=shape, dtype=np.bool_)
    logging.info(".... data shape %s, density %.4f" % (outs.shape, outs.nnz / np.prod(outs.shape)))
    return outs


def main():
    logging.info("Step 1. loading train data")
    tr_data = data_loader(FLAGS.train)
    Dv = np.asarray(tr_data.astype(np.float32).sum(0)).flatten()
    sDv = np.power(np.maximum(Dv, 1.), FLAGS.sugar / 2.)
    before_run(Dv)

    m = Model(sc, FLAGS)

    fVU: str = "VU[%05d,%.5f,%.5f,%05d]" % (FLAGS.eta, FLAGS.epsilon, FLAGS.scale, FLAGS.rank)
    if FLAGS.reuse and path.exists(fVU):
        solut = pickle.load(open(fVU, "rb"))
    else:
        logging.info("Step 2. training model")
        solut = m.train(tr_data, Kernel.build(FLAGS.kernel), Solvoer.build(FLAGS.solver))
        pickle.dump(solut, open(fVU, "wb"))

    logging.info("Step 3. loading fold-in data and test data")
    test_tr = data_loader(FLAGS.test_tr, align=tr_data.shape)
    test_te = data_loader(FLAGS.test_te, align=tr_data.shape)

    logging.info("Step 4. evaluating model")
    m.evaluate(test_tr, test_te, solut, Filter.build(FLAGS.filter), sDv)


if __name__ == "__main__":
    sc = SparkContext(conf=SparkConf())
    logging.info("================")
    logging.info(vars(FLAGS))
    logging.info("================")
    main()
