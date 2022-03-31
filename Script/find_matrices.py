from lib.neuronlib import fread_sparse_matrix, fread_dense_matrix, fwrite_dense_matrix
from scipy import linalg
import numpy as np
import argparse
import os


def find_matrices(tau: float, network: str, io_path: str):
    # find exp(tau*W)
    print('Finding exp(tau W)')
    W = fread_sparse_matrix(os.path.join(io_path, network+'_gji'), 4095)
    exp_tauW = linalg.expm(tau*W)
    fwrite_dense_matrix(exp_tauW, os.path.join(io_path, 'exp_tauW'))

    # find inverse[K(0)]
    print('Finding inverse[K(0)]')
    K_0 = fread_dense_matrix(os.path.join(io_path, 'K_0'))
    try:
        invK0 = np.linalg.inv(K_0)
        fwrite_dense_matrix(invK0, os.path.join(io_path, 'invK0'))
    except np.linalg.LinAlgError:
        print('<error> K(0) is singular, cannot find its inverse')
        exit(1)

    # find M = 1/tau*log(K_tau*inv(K_0))
    print('Finding M = log(K(tau) inverse[K(0)])/tau')
    K_tau = fread_dense_matrix(os.path.join(io_path, 'K_tau'))
    M = 1/tau * linalg.logm(np.matmul(K_tau, invK0))
    fwrite_dense_matrix(M, os.path.join(io_path, 'M'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        # prog='FindMatrices',
        description='calculate exp(tauW), inv[K(0)], M from K(0), K(tau)'
    )
    parser.add_argument('tau', help='value of tau', type=float)
    parser.add_argument('network', help='network name', type=str)
    parser.add_argument('-io', help='input/output directory', type=str)
    args = parser.parse_args()
    if args.io == None: args.io = ''

    find_matrices(args.tau, args.network, args.io)