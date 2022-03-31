from lib.neuronlib import fread_sparse_matrix, fread_dense_matrix, GraphMatrixRelation
import numpy as np
import argparse
import os



takeArg = True
def menu():
    tau = 0.1
    network = 'DIV66'
    io_path = ''

    plot_exp_tauW_vs_Id_add_tauW(tau, network, io_path)
    plot_Ktau_vs_exp_tauW_K0(tau, network, io_path)
    plot_Ktau_invK0_vs_exp_tauW(tau, network, io_path)
    plot_M_vs_W(tau, network, io_path)



def plot_matrices(tau: float, network: str, io_path: str, flag):
    if flag[0]: plot_exp_tauW_vs_Id_add_tauW(tau, network, io_path)
    if flag[1]: plot_Ktau_vs_exp_tauW_K0(tau, network, io_path)
    if flag[2]: plot_Ktau_invK0_vs_exp_tauW(tau, network, io_path)
    if flag[3]: plot_M_vs_W(tau, network, io_path)


def plot_exp_tauW_vs_Id_add_tauW(tau: float, network: str, io_path: str):
    print('Drawing graphs: exp(tauW) vs Id + tauW for diag and off-diag')
    y = fread_dense_matrix(os.path.join(io_path, 'exp_tauW'))
    x = tau * fread_sparse_matrix(os.path.join(io_path, network+'_gji'), 4095) + np.eye(len(y))

    g = GraphMatrixRelation()
    axislabel=[r'exp($\tau$W)', r'Id + $\tau$W']

    g.create_fig()
    g.add_data(x, y, diag=False)
    c = g.fit_linear()
    g.draw_xyline()
    g.label_plot(axislabel=axislabel, textbox='Network: {}, $\\tau$: {}, linear fit: y=[{:.5f}]x+[{:.5f}]'.format(network, tau, c[0], c[1]))
    g.plot()
    g.save_fig('exp_tauW_vs_Id_add_tauW_offdiag', network)

    g.create_fig()
    g.add_data(x, y, offdiag=False)
    g.draw_xyline()
    g.label_plot(axislabel=axislabel, textbox='Network: {}, $\\tau$: {}'.format(network, tau))
    g.plot()
    g.save_fig('exp_tauW_vs_Id_add_tauW_diag', network)

def plot_Ktau_vs_exp_tauW_K0(tau: float, network: str, io_path: str):
    print('Drawing graphs: K(tau) vs exp(tauW) K(0) for diag and off-diag')
    x = np.matmul(fread_dense_matrix(os.path.join(io_path, 'exp_tauW')), fread_dense_matrix(os.path.join(io_path, 'K_0')))
    y = fread_dense_matrix(os.path.join(io_path, 'K_tau'))

    g = GraphMatrixRelation()
    axislabel=[r'exp($\tau$W) K(0)', r'K($\tau$)']

    g.create_fig()
    g.add_data(x, y, diag=False)
    c = g.fit_linear()
    g.draw_xyline()
    g.label_plot(axislabel=axislabel, textbox='Network: {}, $\\tau$: {}, linear fit: y=[{:.5f}]x+[{:.5f}]'.format(network, tau, c[0], c[1]))
    g.plot()
    g.save_fig('Ktau_vs_exp_tauW_K0_offdiag', network)

    g.create_fig()
    g.add_data(x, y, offdiag=False)
    g.draw_xyline()
    g.label_plot(axislabel=axislabel, textbox='Network: {}, $\\tau$: {}'.format(network, tau))
    g.plot()
    g.save_fig('Ktau_vs_exp_tauW_K0_diag', network)

def plot_Ktau_invK0_vs_exp_tauW(tau: float, network: str, io_path: str):
    print('Drawing graphs: K(tau) inv[K(0)] vs exp(tau W) for diag and off-diag')
    x = tau * fread_sparse_matrix(os.path.join(io_path, network+'_gji'), 4095)
    y = np.matmul(fread_dense_matrix(os.path.join(io_path, 'K_tau')), fread_dense_matrix(os.path.join(io_path, 'invK0')))

    g = GraphMatrixRelation()
    axislabel = [r'exp($\tau$W)', r'K($\tau$) inverse[K(0)]']

    g.create_fig()
    g.add_data(x, y, diag=False)
    c = g.fit_linear()
    g.draw_xyline()
    g.label_plot(axislabel=axislabel, textbox='Network: {}, $\\tau$: {}, linear fit: y=[{:.5f}]x+[{:.5f}]'.format(network, tau, c[0], c[1]))
    g.plot()
    g.save_fig('Ktau_invK0_vs_exp_tauW_offdiag', network)

    g.create_fig()
    g.add_data(x, y, offdiag=False)
    g.draw_xyline()
    g.label_plot(axislabel=axislabel, textbox='Network: {}, $\\tau$: {}'.format(network, tau))
    g.plot()
    g.save_fig('Ktau_invK0_vs_exp_tauW_diag', network)

def plot_M_vs_W(tau: float, network: str, io_path: str):
    print('Drawing graphs: M = log(K(tau) inv[K(0)])/tau vs W for diag and off-diag')
    x = tau * fread_sparse_matrix(os.path.join(io_path, network+'_gji'), 4095)
    y = fread_dense_matrix(os.path.join(io_path, 'M'))

    g = GraphMatrixRelation()
    axislabel=[r'W', r'M']

    g.create_fig()
    g.add_data(x, y, diag=False)
    c = g.fit_linear()
    g.draw_xyline()
    g.label_plot(axislabel=axislabel, textbox='Network: {}, $\\tau$: {}, linear fit: y=[{:.5f}]x+[{:.5f}]'.format(network, tau, c[0], c[1]))
    g.plot()
    g.save_fig('M_vs_W_offdiag', network)

    g.create_fig()
    g.add_data(x, y, offdiag=False)
    g.draw_xyline()
    g.label_plot(axislabel=axislabel, textbox='Network: {}, $\\tau$: {}'.format(network, tau))
    g.plot()
    g.save_fig('M_vs_W_diag', network)


if __name__ == '__main__':
    if takeArg:
        parser = argparse.ArgumentParser(
            # prog='PlotMatrices',
            description='plot the following graphs from matrices calculated by find_matrices.py'
        )
        parser.add_argument('tau',     help='value of tau', type=float)
        parser.add_argument('network', help='network name', type=str)
        parser.add_argument('-io', help='input/output directory', type=str)
        parser.add_argument('-pA', help='plot all graphs avaliable below', action='store_true')
        parser.add_argument('-p0', help='plot exp(tauW) vs Id + tauW', action='store_true')
        parser.add_argument('-p1', help='plot K(tau) vs exp(tauW) K(0)', action='store_true')
        parser.add_argument('-p2', help='plot K(tau) inv[K(0)] vs exp(tau W)', action='store_true')
        parser.add_argument('-p3', help='plot M = log(K(tau) inv[K(0)])/tau vs W', action='store_true')
        args = parser.parse_args()
        flag = [
            args.p0,
            args.p1,
            args.p2,
            args.p3
        ]
        if args.pA: flag = [True] * len(flag)
        if args.io == None: args.io = ''
        plot_matrices(args.tau, args.network, args.io, flag)

    else:
        menu()