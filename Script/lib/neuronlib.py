"""
NeuronLib
=========

A library containing useful tools for analyzing data from neural network simulations.

Provides:
  1. `class Network` for extracting information about neural networks, such as connection probability, degree
  2. `class NeualDynamics` for graph plotting of neural dynamics, such as firing rate and ISI distribution
  3. `class Grapher` for drawing density distribution, matrix relation plots
  4. Other useful tools, such as reading/writing matrix from/into files, matrix and array operations

How to use the library
----------------------
Place the folder "lib" in your working directory. Do not modify its name or content.\n
Import this library as follow:
>>> import lib.neuronlib as nlib

To import specific class:
>>> from lib.neuronlib import NeuralDynamics

See class docstrings for help or further instructions, or you can use the following\n
to view all instructions such as additional arguments for graph plotting.
>>> nlib.ask_for_help()
>>> nlib.help_graph_formatting()

"""


import os
import csv
import math
import numpy as np
from scipy import linalg, stats
from matplotlib import pyplot as plt, ticker as tck, collections as mcol
from lib.cppvector import Vector_fl


class Network:

    def __init__(self, adjacency_matrix, input_folder=None, output_folder=None, delimiter='\t', number_of_neurons=4095, full_matrix_format=False):
        """Tools for analyzing networks.

        Parameters
        ----------
        adjacency_matrix : str or numpy.ndarray
            a string storing the path to a local matrix file or an adjacency matrix in form of a numpy.ndarray
        input_folder : str, optional
            name or path of the input data folder, by default None
        output_folder : str, optional
            name or path of the output folder, by default None
        delimiter : str or chr, optional
            delimiter of the input matrix file, this argument only applies if `full_matrix_format` is enabled, by default `'\\t'`
        network_size : int, optional
            size of the network or number of neuron nodes, by default 4095
        full_matrix_format : bool, optional
            enable if the input matrix file stores the full matrix with each row element separated by `delimiter`;
            disable if the file stores only the nonzero elements with format: j i w_ji, by default False
        """
        if type(adjacency_matrix).__module__ == np.__name__:
            _, self.output_path = self._init_input_output_path(None, output_folder)
            self.adjacency_matrix = adjacency_matrix
            print('Adjacency matrix is input as a numpy.ndarray.')
        elif type(adjacency_matrix) == str:
            input_path, self.output_path = self._init_input_output_path(input_folder, output_folder)
            self.adjacency_matrix = self._init_adjacency_matrix_from_file(adjacency_matrix, input_path, delimiter, number_of_neurons, full_matrix_format)
            print('Adjacency matrix is input from a local file: {}.'.format(adjacency_matrix))
        else:
            err = 'ArgumentTypeError: the input argument \'adjacency matrix\' must either be a numpy.ndarray or a string storing the path to a local matrix file.'
            print(err); exit(1)
        np.fill_diagonal(self.adjacency_matrix, 0) # assume no self-linkage
        self.number_of_neurons = number_of_neurons
        self.isAdjMatModified = False

    def __del__(self):
        pass

    def _init_input_output_path(self, input_folder: str, output_folder: str):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        index = this_dir.rfind('\\')
        this_dir = this_dir[:index]
        if input_folder == None: input_folder = ''
        else: input_folder = input_folder + '\\'
        input_path = this_dir + '\\' + input_folder
        if output_folder == None: output_folder = ''
        else: output_folder = output_folder + '\\'
        output_path = this_dir + '\\' + output_folder
        try: os.mkdir(output_path)
        except FileExistsError: pass
        return input_path, output_path

    def _init_adjacency_matrix_from_file(self, adjacency_matrix_file: str, input_path: str, delimiter: str, network_size: int, full_matrix_format: bool):
        try:
            with open(input_path+adjacency_matrix_file, 'r', newline='') as fp:
                if full_matrix_format == False:
                    content = list(csv.reader(fp, delimiter=' '))
                    for i in range(len(content)):
                        content[i] = remove_all_occurrences('', content[i])
                        content[i][0] = int(content[i][0])-1 # j
                        content[i][1] = int(content[i][1])-1 # i
                        content[i][2] = float(content[i][2]) # w_ij
                    adjacency_matrix = np.zeros((network_size, network_size))
                    for item in content:
                        adjacency_matrix[item[1]][item[0]] = item[2]
                    del content
                    adjacency_matrix = np.array(adjacency_matrix).astype(float)
                else:
                    reader = csv.reader(fp, delimiter=delimiter)
                    adjacency_matrix = np.array(list(reader)).astype(float)
        except FileNotFoundError:
            err = 'FileNotFoundError: adjacency matrix file [{}] cannot be found.'.format(input_path+adjacency_matrix_file)
            print(err); exit(1)
        return adjacency_matrix

    def _init_synaptic_weight(self, _adjacency_matrix=None, _custom_matrix=False):
        if _custom_matrix == False: _adjacency_matrix = self.adjacency_matrix
        negative_weights = np.array(_adjacency_matrix.flatten()[_adjacency_matrix.flatten() < 0])
        positive_weights = np.array(_adjacency_matrix.flatten()[_adjacency_matrix.flatten() > 0])
        nonzero_weights = np.array(_adjacency_matrix.flatten()[_adjacency_matrix.flatten() != 0])
        return negative_weights, positive_weights, nonzero_weights

    def _plot_distribution(self, data, bin_size=0.15, color='b', marker_style='^', line_style='-',
                           plot_type='line', marker_size=8, marker_fill=True, line_width=2,
                           figsize=(9, 6), dpi=150, plot_label='', xlabel='', textbox='',
                           xlim=(None, None), ylim=(None, None), show_norm=False,
                           x_logscale=False, y_logscale=False, remove_zero_density=False,
                           file_name='plot_dist', file_label='', file_type=['svg','png'],
                           save_fig=True, show_fig=False, mpl_ax=None,
                           return_plot_data=False, return_area_under_graph=False):

        # Return ax
        def plot_subroutine(x, y, ax, ptype: str, plabel: str, plot_norm=False):
            if plot_norm: c='r'; marker='None'; ls='--'
            else: c=color; marker=marker_style; ls=line_style
            if marker_fill == True:
                if ptype == 'line': ax.plot(x, y, c=c, marker=marker, ms=marker_size, ls=ls, lw=line_width,
                                                label=plabel)
                else: ax.scatter(x, y, c=c, marker=marker, s=marker_size, label=plabel)
            else:
                if ptype == 'line': ax.plot(x, y, c=c, marker=marker, ms=marker_size, ls=ls, lw=line_width,
                                                mfc='none', label=plabel)
                else: ax.scatter(x, y, c=c, marker=marker, s=marker_size, facecolors='none', label=plabel)
            return ax
        
        # Return ax, plot_data, area_under_graph
        def make_plot(ax):
            min_datum = np.amin(data); max_datum = np.amax(data)
            if show_norm == True:
                mu = np.mean(data); sigma = np.std(data)
                x_value_norm = np.linspace(mu - 4*sigma, mu + 4*sigma, 150)
                ax = plot_subroutine(x_value_norm, stats.norm.pdf(x_value_norm, mu, sigma), ax, 'line', 'Normal distribution', plot_norm=True)
            if x_logscale == False:
                number_of_bins = math.ceil((max_datum - min_datum) / bin_size)
                hist_density, bin_edges = np.histogram(data, bins=np.linspace(min_datum, min_datum+number_of_bins*bin_size,
                                                                              number_of_bins), density=True)
            elif x_logscale == True:
                min_datum = np.amin(np.array(data)[np.array(data) > 0])
                number_of_bins = math.ceil((math.log10(max_datum) - math.log10(min_datum)) / bin_size)
                hist_density, bin_edges = np.histogram(data, bins=np.logspace(math.log10(min_datum),
                                                       math.log10(min_datum)+number_of_bins*bin_size, number_of_bins))
            hist_density = np.array(hist_density, dtype=float)
            hist_density /= np.dot(hist_density, np.diff(bin_edges)) # normalization
            x_value = (bin_edges[1:] + bin_edges[:-1]) / 2
            area_under_graph = np.dot(hist_density, np.diff(bin_edges))
            if remove_zero_density == True:
                non_zero_points = np.argwhere(hist_density == 0)
                hist_density = np.delete(hist_density, non_zero_points, 0)
                x_value = np.delete(x_value, non_zero_points, 0)
            return plot_subroutine(x_value, hist_density, ax, plot_type, plot_label), np.array(list(zip(x_value, hist_density))), area_under_graph

        # Return ax
        def format_plot(ax):
            if x_logscale == True: ax.set_xscale('log')
            if y_logscale == True: ax.set_yscale('log')
            ax.set(xlabel=xlabel, ylabel='Probability density')
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
            ax.grid(True)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(reversed(handles), reversed(labels))#, title='Distribution')
            # textbox
            if textbox != '':
                props = dict(boxstyle='round', pad=0.1, facecolor='white', edgecolor='none', alpha=0.75)
                ax.text(0.00001, 1.05, textbox, fontsize=10, verticalalignment='top', transform=ax.transAxes, bbox=props)
            return ax

        if mpl_ax == None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            ax, plot_data, area_under_graph = make_plot(ax)
            ax = format_plot(ax)
            for ext in file_type:
                if file_label == '': file_path = file_name+'.'+ext
                else: file_path = file_name+'_'+file_label+'.'+ext
                if save_fig: fig.savefig(os.path.join(self.output_path, file_path))
            if show_fig: plt.show()
            plt.clf()
            if return_plot_data and not return_area_under_graph: return plot_data
            elif not return_plot_data and return_area_under_graph: return area_under_graph
            else: return plot_data, area_under_graph
        else:
            ax, plot_data, _ = make_plot(mpl_ax)
            if return_plot_data: return ax, plot_data
            else: return ax

    def fwrite_adjacency_matrix(self, file_name='adjacency_matrix', file_label='', file_type='txt', full_matrix_format=False, _matrix=None, _custom_matrix=False):
        """Write adjacency matrix of the network into a local file."""
        if _custom_matrix == False: _matrix = self.adjacency_matrix
        if file_label == '': file_path = '{}.{}'.format(file_name, file_type)
        else: file_path = '{}_{}.{}'.format(file_name, file_label, file_type)
        if full_matrix_format:
            with open(self.output_path+file_path, 'w') as fp:
                for row in _matrix:
                    if row[0] == 0: fp.write('{:.0f}'.format(row[0])) # To reduce file size
                    else:           fp.write( '{:.8}'.format(row[0]))
                    for element in row[1:]:
                        if element == 0:    fp.write('\t{:.0f}'.format(element)) # To reduce file size
                        else:               fp.write( '\t{:.8}'.format(element))
                    fp.write('\n')
        else:
            with open(self.output_path+file_path, 'w') as fp:
                for i in range(len(_matrix)):
                    for j in range(len(_matrix[i])):
                        if _matrix[i][j] != 0:
                            fp.write('{:d}{}{:d}{}{:f}\n'.format(j+1, ' ', i+1, ' ', _matrix[i][j]))
        print('Writing adjacency matrix: output to [{}] in directory [{}]'.format(file_path, self.output_path))

    def statistics_of_synaptic_weights(self, link_type: str, q=0.5, _adjacency_matrix=None, _custom_matrix=False)->tuple:
        """Find the statistics of synaptic weights of the network.

        Parameters
        ----------
        link_type : str
            the type of links to be returned, options: 'inh' or 'exc' or 'all'
        q : float, optional
            q-th percentile of the return data, by default 0.5

        Returns
        -------
        tuple (scalar, scalar, scalar)
            mean, standanrd deviation, q-th percentile of the requested type of synaptic weights
        """
        syn_w_neg, syn_w_pos, syn_w = self._init_synaptic_weight(_adjacency_matrix, _custom_matrix)
        mean_neg, mean_pos, mean_all = np.mean(syn_w_neg), np.mean(syn_w_pos), np.mean(syn_w)
        std_neg, std_pos, std_all = np.std(syn_w_neg), np.std(syn_w_pos), np.std(syn_w)
        pct_neg, pct_pos, pct_all = np.percentile(syn_w_neg, q), np.percentile(syn_w_pos, q), np.percentile(syn_w, q)
        if link_type == 'inh': return mean_neg, std_neg, pct_neg
        if link_type == 'exc': return mean_pos, std_pos, pct_pos
        else: return mean_all, std_all, pct_all

    def connection_probability(self, _adjacency_matrix=None, _custom_matrix=False)->np.ndarray:
        """Return the connection probability of the network."""
        if _custom_matrix == False:
            matrix_size = np.shape(self.adjacency_matrix)[0]
            _, _, nonzero_weights = self._init_synaptic_weight()
        else:
            matrix_size = np.shape(_adjacency_matrix)[0]
            _, _, nonzero_weights = self._init_synaptic_weight(_adjacency_matrix)
        num_of_links = len(nonzero_weights)
        connection_probability = num_of_links / (matrix_size * (matrix_size - 1))
        return connection_probability

    def number_of_links(self, link_type='', _adjacency_matrix=None, _custom_matrix=False)->np.ndarray:
        """Return the number of links in the network.

        Parameters
        ----------
        link_type : str
            the type of links to be returned, options: 'inh' or 'exc' or 'all'

        Returns
        -------
        numpy.ndarray
            an array of number of links of requested type, else an array with 0th, 1st, 2nd element
            storing the array of number of links of inhibitory, excitatory, all neurons correspondingly
        """
        if _custom_matrix == False: negative_weights, positive_weights, nonzero_weights = self._init_synaptic_weight()
        else: negative_weights, positive_weights, nonzero_weights = self._init_synaptic_weight(_adjacency_matrix)
        number_of_links = []
        number_of_links.append(len(negative_weights))
        number_of_links.append(len(positive_weights))
        number_of_links.append(len(nonzero_weights))
        if link_type == 'inh': return np.array(number_of_links)[0]
        if link_type == 'exc': return np.array(number_of_links)[1]
        if link_type == 'all': return np.array(number_of_links)[2]
        else: return np.array(number_of_links)

    def neuron_type(self, _adjacency_matrix=None, _custom_matrix=False)->np.ndarray:
        """Return the electrophysiological class of neurons in the network."""
        if _custom_matrix == False: _adjacency_matrix = self.adjacency_matrix
        matrix_size = np.shape(_adjacency_matrix)[0]
        neuron_type = np.zeros(matrix_size)
        for row in _adjacency_matrix:
            for idx in range(matrix_size):
                if row[idx] < 0:
                    if neuron_type[idx] == 1: print('Warning: inconsistent classification in neuron type.')
                    neuron_type[idx] = -1
                elif row[idx] > 0:
                    if neuron_type[idx] == -1: print('Warning: inconsistent classification in neuron type.')
                    neuron_type[idx] = 1
        return neuron_type

    def excitatory_to_inhibitory_ratio(self)->float:
        """Return the ratio of excitatory neurons to inhibitory neurons."""
        exc = np.count_nonzero(self.neuron_type() == +1)
        inh = np.count_nonzero(self.neuron_type() == -1)
        if inh == 0: return np.Inf
        else: return exc/inh

    def incoming_degree(self, link_type='', _adjacency_matrix=None, _custom_matrix=False)->np.ndarray:
        """Return the incoming degree of neurons in the network.

        Parameters
        ----------
        link_type : str
            the type of links to be returned, options: 'inh' or 'exc' or 'all'

        Returns
        -------
        numpy.ndarray
            an array of incoming degrees of requested type, else an array with 0th, 1st, 2nd element
            storing the array of incoming degrees of inhibitory, excitatory, all neurons correspondingly
        """
        if _custom_matrix == False: _adjacency_matrix = self.adjacency_matrix
        incoming_degree, temp = [], []
        for row in _adjacency_matrix:
            temp.append(len(row[row < 0]))  #inh
            temp.append(len(row[row > 0]))  #exc
            temp.append(len(row[row != 0])) #all
            incoming_degree.append(np.array(temp))
            temp.clear()
        if link_type == 'inh': return np.array(incoming_degree).T[0]
        if link_type == 'exc': return np.array(incoming_degree).T[1]
        if link_type == 'all': return np.array(incoming_degree).T[2]
        else: return np.array(incoming_degree).T

    def outgoing_degree(self, _adjacency_matrix=None, _custom_matrix=False)->np.ndarray:
        """Return the outgoing degree of neurons in the network."""
        if _custom_matrix == False: _adjacency_matrix = self.adjacency_matrix
        outgoing_degree = []
        for row in _adjacency_matrix.T: outgoing_degree.append(len(row[row != 0]))
        return np.array(outgoing_degree)

    def average_synaptic_weights_of_incoming_links(self, link_type='', _adjacency_matrix=None, _custom_matrix=False)->np.ndarray:
        """Return the avergae synaptic weights of incoming links of neurons in the network.

        Parameters
        ----------
        link_type : str
            the type of links to be returned, options: 'inh' or 'exc' or 'all'

        Returns
        -------
        numpy.ndarray
            an array of average incoming synaptic weights of requested type, else an array with 0th, 1st, 2nd element
            storing the array of average incoming synaptic weights of inhibitory, excitatory, all neurons correspondingly
        """
        if _custom_matrix == False: _adjacency_matrix = self.adjacency_matrix
        average_weights_in = []; temp = []
        for row in _adjacency_matrix:
            k_inh = len(row[row < 0])
            k_exc = len(row[row > 0])
            k_all = len(row[row != 0])
            if k_inh != 0: temp.append(abs(np.sum(row[row < 0]) / k_inh))
            else: temp.append(None)
            if k_exc != 0: temp.append(np.sum(row[row > 0]) / k_exc)
            else: temp.append(None)
            if k_all != 0: temp.append(np.sum(row[row != 0]) / k_all)
            else: temp.append(None)
            average_weights_in.append(np.array(temp))
            temp.clear()
        if link_type == 'inh': return np.array(average_weights_in).T[0]
        if link_type == 'exc': return np.array(average_weights_in).T[1]
        if link_type == 'all': return np.array(average_weights_in).T[2]
        else: return np.array(average_weights_in).T

    def average_synaptic_weights_of_outgoing_links(self, link_type='', _adjacency_matrix=None, _custom_matrix=False)->np.ndarray:
        """Return the avergae synaptic weights of outgoing links of neurons in the network.

        Parameters
        ----------
        link_type : str
            the type of links to be returned, options: 'inh' or 'exc' or 'all'

        Returns
        -------
        numpy.ndarray
            an array of average outgoing synaptic weights of requested type, else an array with 0th, 1st, 2nd element
            storing the array of average outgoing synaptic weights of inhibitory, excitatory, all neurons correspondingly
        """
        if _custom_matrix == False: _adjacency_matrix = self.adjacency_matrix
        average_weights_out, temp = [], []
        for row in _adjacency_matrix.T:
            k_inh = len(row[row < 0])
            k_exc = len(row[row > 0])
            k_all = len(row[row != 0])
            if k_inh != 0: temp.append(abs(np.sum(row[row < 0]) / k_inh))
            else: temp.append(None)
            if k_exc != 0: temp.append(np.sum(row[row > 0]) / k_exc)
            else: temp.append(None)
            if k_all != 0: temp.append(np.sum(row[row != 0]) / k_all)
            else: temp.append(None)
            average_weights_out.append(np.array(temp))
            temp.clear()
        if link_type == 'inh': return np.array(average_weights_out).T[0]
        if link_type == 'exc': return np.array(average_weights_out).T[1]
        if link_type == 'all': return np.array(average_weights_out).T[2]
        else: return np.array(average_weights_out)

    def plot_synaptic_weight_distribution_NEG(self, bin_size=0.001, mpl_ax=None, c='b', m='^', ls='-',
                                              pt='line', ms=8, mfc=True, lw=2, figsize=(9, 6), dpi=150,
                                              show_norm=False, xlim=(None, None), x_logscale=False,
                                              y_logscale=False, remove_zero_density=False,
                                              plot_label='', file_label='', file_type=['svg','png']):
        """Plot the distribution of negative synaptic weights.

        Parameters
        ----------
        bin_size : int, optional
            size of bin, by default 0.001
        mpl_ax : matplotlib.axes.Axes, optional
            if a matplotlib Axes is given, append the plot to the Axes, by default None
        c : str, optional
            colour of lines and markers, by default 'b'
        m : str, optional
            type of marker, e.g., '^', 'o', etc, see matplotlib for details, by default '^'
        ls : str, optional
            style of line, e.g., '-', ':', etc, see matplotlib for details, by default '-'
        pt : str, optional
            plot type, options: 'line' or 'scatter', by default 'line'
        ms : int, optional
            size of marker, by default 8
        mfc : bool, optional
            marker face fill, solid markers if True, open markers if False, by default True
        lw : int, optional
            line weight, by default 2
        figsize : tuple of float, optional
            figure size, by default (9, 6)
        dpi : int, optional
            dots per inch, by default 150
        show_norm : bool, optional
            show a normal distribution with mean and S.D. of the data if True, by default False
        xlim : tuple, optional
            range of x-axis, by default (None, None)
        x_logscale : bool, optional
            x-axis in log scale if True, by default False
        y_logscale : bool, optional
            y-axis in log scale if True, by default False
        remove_zero_density : bool, optional
            remove zero density when `x_logscale` is enabled to maintain a connected graph, by default False
        plot_label : str, optional
            label of the plot to be shown in legend, by default 'line'
        file_label : str, optional
            label of the file to be appended at the end of the file name, by default ''
        file_type : list, optional
            format(s)/extension(s) of the output plot, by default ['svg','png']

        Returns
        -------
        tuple (numpy.ndarray, int)
            if `mpl_ax` is not provided, return a tuple of (plot data, area under curve)
        matplotlib.axes.Axes
            if `mpl_ax` is provided, return the Axes with the plot appended
        """
        negative_weights = self._init_synaptic_weight()[0]
        print('Drawing graph: distribution of negative synaptic weights')
        if mpl_ax == None:
            return self._plot_distribution(negative_weights, bin_size, c, m, ls, pt, ms, mfc, lw,
                                           figsize, dpi, 'Negative w_ij distribution', 'Negative w_ij',
                                           'Distribution of negative synaptic weights', xlim, (0, None),
                                           show_norm, x_logscale, y_logscale, remove_zero_density,
                                           'plot_synaptic_weight_dist_neg', file_label, file_type)
        else:
            return self._plot_distribution(negative_weights, bin_size, c, m, ls, pt, ms, mfc, lw,
                                           x_logscale=x_logscale, y_logscale=y_logscale,
                                           show_norm=show_norm, remove_zero_density=remove_zero_density,
                                           plot_label=plot_label, mpl_ax=mpl_ax)

    def plot_synaptic_weight_distribution_POS(self, bin_size=0.001, mpl_ax=None, c='b', m='^', ls='-',
                                              pt='line', ms=8, mfc=True, lw=2, figsize=(9, 6), dpi=150,
                                              show_norm=False, xlim=(None, None), x_logscale=False,
                                              y_logscale=False, remove_zero_density=False,
                                              plot_label='', file_label='', file_type=['svg','png']):
        """Plot the distribution of positive synaptic weights.

        Parameters
        ----------
        bin_size : int, optional
            size of bin, by default 0.001
        mpl_ax : matplotlib.axes.Axes, optional
            if a matplotlib Axes is given, append the plot to the Axes, by default None
        c : str, optional
            colour of lines and markers, by default 'b'
        m : str, optional
            type of marker, e.g., '^', 'o', etc, see matplotlib for details, by default '^'
        ls : str, optional
            style of line, e.g., '-', ':', etc, see matplotlib for details, by default '-'
        pt : str, optional
            plot type, options: 'line' or 'scatter', by default 'line'
        ms : int, optional
            size of marker, by default 8
        mfc : bool, optional
            marker face fill, solid markers if True, open markers if False, by default True
        lw : int, optional
            line weight, by default 2
        figsize : tuple of float, optional
            figure size, by default (9, 6)
        dpi : int, optional
            dots per inch, by default 150
        show_norm : bool, optional
            show a normal distribution with mean and S.D. of the data if True, by default False
        xlim : tuple, optional
            range of x-axis, by default (None, None)
        x_logscale : bool, optional
            x-axis in log scale if True, by default False
        y_logscale : bool, optional
            y-axis in log scale if True, by default False
        remove_zero_density : bool, optional
            remove zero density when `x_logscale` is enabled to maintain a connected graph, by default False
        plot_label : str, optional
            label of the plot to be shown in legend, by default 'line'
        file_label : str, optional
            label of the file to be appended at the end of the file name, by default ''
        file_type : list, optional
            format(s)/extension(s) of the output plot, by default ['svg','png']

        Returns
        -------
        tuple (numpy.ndarray, int)
            if `mpl_ax` is not provided, return a tuple of (plot data, area under curve)
        matplotlib.axes.Axes
            if `mpl_ax` is provided, return the Axes with the plot appended
        """
        positive_weights = self._init_synaptic_weight()[1]
        print('Drawing graph: distribution of positive synaptic weights')
        if mpl_ax == None:
            return self._plot_distribution(positive_weights, bin_size, c, m, ls, pt, ms, mfc, lw,
                                            figsize, dpi, 'Positive w_ij distribution', 'Positive w_ij',
                                            'Distribution of positive synaptic weights', xlim, (0, None),
                                            show_norm, x_logscale, y_logscale, remove_zero_density,
                                            'plot_synaptic_weight_dist_pos', file_label, file_type)
        else:
            return self._plot_distribution(positive_weights, bin_size, c, m, ls, pt, ms, mfc, lw,
                                            x_logscale=x_logscale, y_logscale=y_logscale,
                                            show_norm=show_norm, remove_zero_density=remove_zero_density,
                                            plot_label=plot_label, mpl_ax=mpl_ax)

    def plot_synaptic_weight_distribution(self, bin_size=0.001, mpl_ax=None, c='b', m='^', ls='-',
                                              pt='line', ms=8, mfc=True, lw=2, figsize=(9, 6), dpi=150,
                                              show_norm=False, xlim=(None, None), x_logscale=False,
                                              y_logscale=False, remove_zero_density=False,
                                              plot_label='', file_label='', file_type=['svg','png']):
        """Plot the distribution of synaptic weights.

        Parameters
        ----------
        bin_size : int, optional
            size of bin, by default 0.001
        mpl_ax : matplotlib.axes.Axes, optional
            if a matplotlib Axes is given, append the plot to the Axes, by default None
        c : str, optional
            colour of lines and markers, by default 'b'
        m : str, optional
            type of marker, e.g., '^', 'o', etc, see matplotlib for details, by default '^'
        ls : str, optional
            style of line, e.g., '-', ':', etc, see matplotlib for details, by default '-'
        pt : str, optional
            plot type, options: 'line' or 'scatter', by default 'line'
        ms : int, optional
            size of marker, by default 8
        mfc : bool, optional
            marker face fill, solid markers if True, open markers if False, by default True
        lw : int, optional
            line weight, by default 2
        figsize : tuple of float, optional
            figure size, by default (9, 6)
        dpi : int, optional
            dots per inch, by default 150
        show_norm : bool, optional
            show a normal distribution with mean and S.D. of the data if True, by default False
        xlim : tuple, optional
            range of x-axis, by default (None, None)
        x_logscale : bool, optional
            x-axis in log scale if True, by default False
        y_logscale : bool, optional
            y-axis in log scale if True, by default False
        remove_zero_density : bool, optional
            remove zero density when `x_logscale` is enabled to maintain a connected graph, by default False
        plot_label : str, optional
            label of the plot to be shown in legend, by default 'line'
        file_label : str, optional
            label of the file to be appended at the end of the file name, by default ''
        file_type : list, optional
            format(s)/extension(s) of the output plot, by default ['svg','png']

        Returns
        -------
        tuple (numpy.ndarray, int)
            if `mpl_ax` is not provided, return a tuple of (plot data, area under curve)
        matplotlib.axes.Axes
            if `mpl_ax` is provided, return the Axes with the plot appended
        """
        nonzero_weights = self._init_synaptic_weight()[2]
        print('Drawing graph: distribution of synaptic weights')
        if mpl_ax == None:
            return self._plot_distribution(nonzero_weights, bin_size, c, m, ls, pt, ms, mfc, lw,
                                           figsize, dpi, 'w_ij distribution', 'w_ij',
                                           'Distribution of synaptic weights', xlim, (0, None),
                                           show_norm, x_logscale, y_logscale, remove_zero_density,
                                           'plot_synaptic_weight_dist', file_label, file_type)
        else:
            return self._plot_distribution(nonzero_weights, bin_size, c, m, ls, pt, ms, mfc, lw,
                                           x_logscale=x_logscale, y_logscale=y_logscale,
                                           show_norm=show_norm, remove_zero_density=remove_zero_density,
                                           plot_label=plot_label, mpl_ax=mpl_ax)

    def plot_incoming_degree_distribution_INH(self, bin_size=1, mpl_ax=None, c='b', m='^', ls='-',
                                              pt='line', ms=8, mfc=True, lw=2, figsize=(9, 6), dpi=150,
                                              show_norm=False, xlim=(None, None), x_logscale=False,
                                              y_logscale=False, remove_zero_density=False,
                                              plot_label='line', file_label='', file_type=['svg','png'],
                                              plot_node=[]):
        """Plot the distribution of inhibitory incoming degrees.

        Parameters
        ----------
        bin_size : int, optional
            size of bin, by default 1
        mpl_ax : matplotlib.axes.Axes, optional
            if a matplotlib Axes is given, append the plot to the Axes, by default None
        c : str, optional
            colour of lines and markers, by default 'b'
        m : str, optional
            type of marker, e.g., '^', 'o', etc, see matplotlib for details, by default '^'
        ls : str, optional
            style of line, e.g., '-', ':', etc, see matplotlib for details, by default '-'
        pt : str, optional
            plot type, options: 'line' or 'scatter', by default 'line'
        ms : int, optional
            size of marker, by default 8
        mfc : bool, optional
            marker face fill, solid markers if True, open markers if False, by default True
        lw : int, optional
            line weight, by default 2
        figsize : tuple of float, optional
            figure size, by default (9, 6)
        dpi : int, optional
            dots per inch, by default 150
        show_norm : bool, optional
            show a normal distribution with mean and S.D. of the data if True, by default False
        xlim : tuple, optional
            range of x-axis, by default (None, None)
        x_logscale : bool, optional
            x-axis in log scale if True, by default False
        y_logscale : bool, optional
            y-axis in log scale if True, by default False
        remove_zero_density : bool, optional
            remove zero density when `x_logscale` is enabled to maintain a connected graph, by default False
        plot_label : str, optional
            label of the plot to be shown in legend, by default 'line'
        file_label : str, optional
            label of the file to be appended at the end of the file name, by default ''
        file_type : list, optional
            format(s)/extension(s) of the output plot, by default ['svg','png']
        plot_node : list, optional
            the selected nodes to be plotted, by default [] (plot all nodes)

        Returns
        -------
        tuple (numpy.ndarray, int)
            if `mpl_ax` is not provided, return a tuple of (plot data, area under curve)
        matplotlib.axes.Axes
            if `mpl_ax` is provided, return the Axes with the plot appended
        """
        inc_deg_inh = self.incoming_degree('inh')
        if len(plot_node) != 0: inc_deg_inh = inc_deg_inh[plot_node]
        if mpl_ax == None:
            return self._plot_distribution(inc_deg_inh, bin_size, c, m, ls, pt, ms, mfc, lw,
                                           figsize, dpi, 'k-_in distribution', 'k-_in',
                                           'Distribution of inhibitory incoming degree | bin size: {}'.format(bin_size),
                                           xlim, (0, None), show_norm, x_logscale, y_logscale,
                                           remove_zero_density, 'plot_dist_incoming_degree_inh',
                                           file_label, file_type)
        else:
            return self._plot_distribution(inc_deg_inh, bin_size, c, m, ls, pt, ms, mfc, lw,
                                           x_logscale=x_logscale, y_logscale=y_logscale,
                                           show_norm=show_norm, remove_zero_density=remove_zero_density,
                                           plot_label=plot_label, mpl_ax=mpl_ax)

    def plot_incoming_degree_distribution_EXC(self, bin_size=1, mpl_ax=None, c='b', m='^', ls='-',
                                              pt='line', ms=8, mfc=True, lw=2, figsize=(9, 6), dpi=150,
                                              show_norm=False, xlim=(None, None), x_logscale=False,
                                              y_logscale=False, remove_zero_density=False,
                                              plot_label='line', file_label='', file_type=['svg','png'],
                                              plot_node=[]):
        """Plot the distribution of excitatory incoming degrees.

        Parameters
        ----------
        bin_size : int, optional
            size of bin, by default 1
        mpl_ax : matplotlib.axes.Axes, optional
            if a matplotlib Axes is given, append the plot to the Axes, by default None
        c : str, optional
            colour of lines and markers, by default 'b'
        m : str, optional
            type of marker, e.g., '^', 'o', etc, see matplotlib for details, by default '^'
        ls : str, optional
            style of line, e.g., '-', ':', etc, see matplotlib for details, by default '-'
        pt : str, optional
            plot type, options: 'line' or 'scatter', by default 'line'
        ms : int, optional
            size of marker, by default 8
        mfc : bool, optional
            marker face fill, solid markers if True, open markers if False, by default True
        lw : int, optional
            line weight, by default 2
        figsize : tuple of float, optional
            figure size, by default (9, 6)
        dpi : int, optional
            dots per inch, by default 150
        show_norm : bool, optional
            show a normal distribution with mean and S.D. of the data if True, by default False
        xlim : tuple, optional
            range of x-axis, by default (None, None)
        x_logscale : bool, optional
            x-axis in log scale if True, by default False
        y_logscale : bool, optional
            y-axis in log scale if True, by default False
        remove_zero_density : bool, optional
            remove zero density when `x_logscale` is enabled to maintain a connected graph, by default False
        plot_label : str, optional
            label of the plot to be shown in legend, by default 'line'
        file_label : str, optional
            label of the file to be appended at the end of the file name, by default ''
        file_type : list, optional
            format(s)/extension(s) of the output plot, by default ['svg','png']
        plot_node : list, optional
            the selected nodes to be plotted, by default [] (plot all nodes)

        Returns
        -------
        tuple (numpy.ndarray, int)
            if `mpl_ax` is not provided, return a tuple of (plot data, area under curve)
        matplotlib.axes.Axes
            if `mpl_ax` is provided, return the Axes with the plot appended
        """
        inc_deg_exc = self.incoming_degree('exc')
        if len(plot_node) != 0: inc_deg_exc = inc_deg_exc[plot_node]
        if mpl_ax == None:
            return self._plot_distribution(inc_deg_exc, bin_size, c, m, ls, pt, ms, mfc, lw,
                                           figsize, dpi, 'k+_in distribution', 'k+_in',
                                           'Distribution of excitatory incoming degree | bin size: {}'.format(bin_size),
                                           xlim, (0, None), show_norm, x_logscale, y_logscale,
                                           remove_zero_density, 'plot_dist_incoming_degree_exc',
                                           file_label, file_type)
        else:
            return self._plot_distribution(inc_deg_exc, bin_size, c, m, ls, pt, ms, mfc, lw,
                                           x_logscale=x_logscale, y_logscale=y_logscale,
                                           show_norm=show_norm, remove_zero_density=remove_zero_density,
                                           plot_label=plot_label, mpl_ax=mpl_ax)

    def plot_incoming_degree_distribution(self, bin_size=1, mpl_ax=None, c='b', m='^', ls='-',
                                          pt='line', ms=8, mfc=True, lw=2, figsize=(9, 6), dpi=150,
                                          show_norm=False, xlim=(None, None), x_logscale=False,
                                          y_logscale=False, remove_zero_density=False,
                                          plot_label='line', file_label='', file_type=['svg','png'],
                                          plot_node=[]):
        """Plot the distribution of incoming degrees.

        Parameters
        ----------
        bin_size : int, optional
            size of bin, by default 1
        mpl_ax : matplotlib.axes.Axes, optional
            if a matplotlib Axes is given, append the plot to the Axes, by default None
        c : str, optional
            colour of lines and markers, by default 'b'
        m : str, optional
            type of marker, e.g., '^', 'o', etc, see matplotlib for details, by default '^'
        ls : str, optional
            style of line, e.g., '-', ':', etc, see matplotlib for details, by default '-'
        pt : str, optional
            plot type, options: 'line' or 'scatter', by default 'line'
        ms : int, optional
            size of marker, by default 8
        mfc : bool, optional
            marker face fill, solid markers if True, open markers if False, by default True
        lw : int, optional
            line weight, by default 2
        figsize : tuple of float, optional
            figure size, by default (9, 6)
        dpi : int, optional
            dots per inch, by default 150
        show_norm : bool, optional
            show a normal distribution with mean and S.D. of the data if True, by default False
        xlim : tuple, optional
            range of x-axis, by default (None, None)
        x_logscale : bool, optional
            x-axis in log scale if True, by default False
        y_logscale : bool, optional
            y-axis in log scale if True, by default False
        remove_zero_density : bool, optional
            remove zero density when `x_logscale` is enabled to maintain a connected graph, by default False
        plot_label : str, optional
            label of the plot to be shown in legend, by default 'line'
        file_label : str, optional
            label of the file to be appended at the end of the file name, by default ''
        file_type : list, optional
            format(s)/extension(s) of the output plot, by default ['svg','png']
        plot_node : list, optional
            the selected nodes to be plotted, by default [] (plot all nodes)

        Returns
        -------
        tuple (numpy.ndarray, int)
            if `mpl_ax` is not provided, return a tuple of (plot data, area under curve)
        matplotlib.axes.Axes
            if `mpl_ax` is provided, return the Axes with the plot appended
        """
        inc_deg_all = self.incoming_degree('all')
        if len(plot_node) != 0: inc_deg_all = inc_deg_all[plot_node]
        if mpl_ax == None:
            return self._plot_distribution(inc_deg_all, bin_size, c, m, ls, pt, ms, mfc, lw,
                                           figsize, dpi, 'k_in distribution', 'k_in',
                                           'Distribution of incoming degree | bin size: {}'.format(bin_size),
                                           xlim, (0, None), show_norm, x_logscale, y_logscale,
                                           remove_zero_density, 'plot_dist_incoming_degree',
                                           file_label, file_type)
        else:
            return self._plot_distribution(inc_deg_all, bin_size, c, m, ls, pt, ms, mfc, lw,
                                           x_logscale=x_logscale, y_logscale=y_logscale,
                                           show_norm=show_norm, remove_zero_density=remove_zero_density,
                                           plot_label=plot_label, mpl_ax=mpl_ax)

    def plot_outgoing_degree_distribution(self, bin_size=1, mpl_ax=None, c='b', m='^', ls='-',
                                          pt='line', ms=8, mfc=True, lw=2, figsize=(9, 6), dpi=150,
                                          show_norm=False, xlim=(None, None), x_logscale=False,
                                          y_logscale=False, remove_zero_density=False,
                                          plot_label='line', file_label='', file_type=['svg','png'],
                                          plot_node=[]):
        """Plot the distribution of outgoing degrees.

        Parameters
        ----------
        bin_size : int, optional
            size of bin, by default 1
        mpl_ax : matplotlib.axes.Axes, optional
            if a matplotlib Axes is given, append the plot to the Axes, by default None
        c : str, optional
            colour of lines and markers, by default 'b'
        m : str, optional
            type of marker, e.g., '^', 'o', etc, see matplotlib for details, by default '^'
        ls : str, optional
            style of line, e.g., '-', ':', etc, see matplotlib for details, by default '-'
        pt : str, optional
            plot type, options: 'line' or 'scatter', by default 'line'
        ms : int, optional
            size of marker, by default 8
        mfc : bool, optional
            marker face fill, solid markers if True, open markers if False, by default True
        lw : int, optional
            line weight, by default 2
        figsize : tuple of float, optional
            figure size, by default (9, 6)
        dpi : int, optional
            dots per inch, by default 150
        show_norm : bool, optional
            show a normal distribution with mean and S.D. of the data if True, by default False
        xlim : tuple, optional
            range of x-axis, by default (None, None)
        x_logscale : bool, optional
            x-axis in log scale if True, by default False
        y_logscale : bool, optional
            y-axis in log scale if True, by default False
        remove_zero_density : bool, optional
            remove zero density when `x_logscale` is enabled to maintain a connected graph, by default False
        plot_label : str, optional
            label of the plot to be shown in legend, by default 'line'
        file_label : str, optional
            label of the file to be appended at the end of the file name, by default ''
        file_type : list, optional
            format(s)/extension(s) of the output plot, by default ['svg','png']
        plot_node : list, optional
            the selected nodes to be plotted, by default [] (plot all nodes)

        Returns
        -------
        tuple (numpy.ndarray, int)
            if `mpl_ax` is not provided, return a tuple of (plot data, area under curve)
        matplotlib.axes.Axes
            if `mpl_ax` is provided, return the Axes with the plot appended
        """
        print('Drawing graph: distribution of outgoing degree')
        out_deg = self.out_deg()
        if len(plot_node) != 0: outgoing_degree = outgoing_degree[plot_node]
        if mpl_ax == None:
            return self._plot_distribution(out_deg, bin_size, c, m, ls, pt, ms, mfc, lw,
                                           figsize, dpi, 'k_out distribution', 'k_out',
                                           'Distribution of outgoing degree | bin size: {}'.format(bin_size),
                                           xlim, (0, None), show_norm, x_logscale, y_logscale,
                                           remove_zero_density, 'plot_dist_outgoing_degree',
                                           file_label, file_type)
        else:
            return self._plot_distribution(out_deg, bin_size, c, m, ls, pt, ms, mfc, lw,
                                           x_logscale=x_logscale, y_logscale=y_logscale,
                                           show_norm=show_norm, remove_zero_density=remove_zero_density,
                                           plot_label=plot_label, mpl_ax=mpl_ax)

    def plot_incoming_average_weight_distribution_INH(self, bin_size=0.0003, mpl_ax=None, c='b', m='^',ls='-',
                                                      pt='line', ms=8, mfc=True, lw=2, figsize=(9, 6), dpi=150,
                                                      show_norm=False, xlim=(None, None), x_logscale=False,
                                                      y_logscale=False, remove_zero_density=False,
                                                      plot_label='line', file_label='', file_type=['svg','png'],
                                                      plot_node=[]):
        avg_w_inc_inh = self.average_synaptic_weights_of_incoming_links('inh')
        if len(plot_node) != 0: avg_w_inc_inh = avg_w_inc_inh[plot_node]
        print('Drawing graph: distribution of average synaptic weights of inhibitory incoming links')
        if mpl_ax == None:
            return self._plot_distribution(avg_w_inc_inh, bin_size, c, m, ls, pt, ms, mfc, lw,
                                           figsize, dpi, '|s-_in| distribution', '|s-_in|',
                                           'Distribution of average synaptic weights of inhibitory incoming links | bin size: {}'.format(bin_size),
                                           xlim, (0, None), show_norm, x_logscale, y_logscale,
                                           remove_zero_density, 'plot_dist_s_in_inh',
                                           file_label, file_type)
        else:
            return self._plot_distribution(avg_w_inc_inh, bin_size, c, m, ls, pt, ms, mfc, lw,
                                           x_logscale=x_logscale, y_logscale=y_logscale,
                                           show_norm=show_norm, remove_zero_density=remove_zero_density,
                                           plot_label=plot_label, mpl_ax=mpl_ax)

    def plot_incoming_average_weight_distribution_EXC(self, bin_size=0.0003, mpl_ax=None, c='b', m='^',ls='-',
                                                      pt='line', ms=8, mfc=True, lw=2, figsize=(9, 6), dpi=150,
                                                      show_norm=False, xlim=(None, None), x_logscale=False,
                                                      y_logscale=False, remove_zero_density=False,
                                                      plot_label='line', file_label='', file_type=['svg','png'],
                                                      plot_node=[]):
        print('Drawing graph: distribution of average synaptic weights of excitatory incoming links')
        avg_w_inc_exc = self.average_synaptic_weights_of_incoming_links('exc')
        if len(plot_node) != 0: avg_w_inc_exc = avg_w_inc_exc[plot_node]
        if mpl_ax == None:
            return self._plot_distribution(avg_w_inc_exc, bin_size, c, m, ls, pt, ms, mfc, lw,
                                           figsize, dpi, 's+_in distribution', 's+_in',
                                           'Distribution of average synaptic weights of excitatory incoming links | bin size: {}'.format(bin_size),
                                           xlim, (0, None), show_norm, x_logscale, y_logscale,
                                           remove_zero_density, 'plot_dist_s_in_exc',
                                           file_label, file_type)
        else:
            return self._plot_distribution(avg_w_inc_exc, bin_size, c, m, ls, pt, ms, mfc, lw,
                                           x_logscale=x_logscale, y_logscale=y_logscale,
                                           show_norm=show_norm, remove_zero_density=remove_zero_density,
                                           plot_label=plot_label, mpl_ax=mpl_ax)

    def plot_incoming_average_weight_distribution(self, bin_size=0.0003, mpl_ax=None, c='b', m='^',ls='-',
                                                  pt='line', ms=8, mfc=True, lw=2, figsize=(9, 6), dpi=150,
                                                  show_norm=False, xlim=(None, None), x_logscale=False,
                                                  y_logscale=False, remove_zero_density=False,
                                                  plot_label='line', file_label='', file_type=['svg','png'],
                                                  plot_node=[]):
        print('Drawing graph: distribution of average synaptic weights of incoming links')
        avg_w_inc = self.average_synaptic_weights_of_incoming_links('all')
        if len(plot_node) != 0: avg_w_inc = avg_w_inc[plot_node]
        if mpl_ax == None:
            return self._plot_distribution(avg_w_inc, bin_size, c, m, ls, pt, ms, mfc, lw,
                                           figsize, dpi, 's_in distribution', 's_in',
                                           'Distribution of average synaptic weights of incoming links | bin size: {}'.format(bin_size),
                                           xlim, (0, None), show_norm, x_logscale, y_logscale,
                                           remove_zero_density, 'plot_dist_s_in',
                                           file_label, file_type)
        else:
            return self._plot_distribution(avg_w_inc, bin_size, c, m, ls, pt, ms, mfc, lw,
                                           x_logscale=x_logscale, y_logscale=y_logscale,
                                           show_norm=show_norm, remove_zero_density=remove_zero_density,
                                           plot_label=plot_label, mpl_ax=mpl_ax)

    def plot_outgoing_average_weight_distribution(self, bin_size=0.0003, mpl_ax=None, c='b', m='^', ls='-',
                                                  pt='line', ms=8, mfc=True, lw=2, figsize=(9, 6), dpi=150,
                                                  show_norm=False, xlim=(None, None), x_logscale=False,
                                                  y_logscale=False, remove_zero_density=False,
                                                  plot_label='line', file_label='', file_type=['svg','png'],
                                                  plot_node=[]):
        print('Drawing graph: distribution of average synaptic weights of outgoing links')
        avg_w_out = self.average_synaptic_weights_of_outgoing_links()
        if len(plot_node) != 0: avg_w_out = avg_w_out[plot_node]
        avg_w_out = avg_w_out[avg_w_out != None] # Remove neuron nodes with no outgoing links
        if mpl_ax == None:
            return self._plot_distribution(avg_w_out, bin_size, c, m, ls, pt, ms, mfc, lw,
                                           figsize, dpi, '|s_out| distribution', '|s_out|',
                                           'Distribution of average synaptic weights of outgoing links | bin size: {}'.format(bin_size),
                                           xlim, (0, None), show_norm, x_logscale, y_logscale,
                                           remove_zero_density, 'plot_dist_s_out',
                                           file_label, file_type)
        else:
            return self._plot_distribution(avg_w_out, bin_size, c, m, ls, pt, ms, mfc, lw,
                                           x_logscale=x_logscale, y_logscale=y_logscale,
                                           show_norm=show_norm, remove_zero_density=remove_zero_density,
                                           plot_label=plot_label, mpl_ax=mpl_ax)

    def plot_links_of_network(self, link_type: str, threshold=0.5, figsize=(9, 9), dpi=150,
                              file_name='strongest_network_links', file_type=['svg','png']):
        """Plot the strongest links in the network.

        Parameters
        ----------
        link_type : str
            _description_
        threshold : float, optional
            links with strength exceed the threshold will be shown, by default 0.5
        figsize : tuple, optional
            figure size, by default (9, 9)
        dpi : int, optional
            dots per inch, by default 150
        file_name : str, optional
            name or path of the output plot, by default 'strongest_network_links'
        file_type : list, optional
            format(s)/extension(s) of the output plot, by default ['svg','png']
        """

        def set_background_with_node_type(node_amt, row_size):
            node_type = self.classify_node_type(self.adjacency_matrix)
            node_count = 0
            for i in range(row_size):
                for j in range(row_size):
                    if node_type[node_count] == -1: ax.scatter([j+1], [i+1], s=5, c='c')
                    elif node_type[node_count] == 1: ax.scatter([j+1], [i+1], s=5, c='m')
                    else: ax.scatter([j+1], [i+1], s=5, c='darkgrey')
                    node_count += 1
                    if node_count == node_amt: break

        def join_two_points(i: int, j: int, row_size: int, color=''):
            ax.plot([i%row_size+1, j%row_size+1], [int(i/row_size)+1, int(j/row_size)+1], color+'-', lw=0.7)

        print('Drawing graph: strongest links in network')
        node_amt = np.shape(self.adjacency_matrix)[0]
        row_size = round(math.sqrt(node_amt))

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        set_background_with_node_type(node_amt, row_size)
        mean, sd, pct = self.statistics_of_synaptic_weights(100-threshold)
        pos_threshold = pct#mean + 1.5 * sd
        mean, sd, pct = self.statistics_of_synaptic_weights(threshold)
        neg_threshold = pct#mean + 1.5 * sd
        for i in range(node_amt):
            for j in range(node_amt):
                if self.adjacency_matrix[i][j] < neg_threshold:
                    if link_type in ['inh', 'all']: join_two_points(i, j, row_size, 'b')
                elif self.adjacency_matrix[i][j] > pos_threshold:
                    if link_type in ['exc', 'all']: join_two_points(i, j, row_size, 'r')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_xlim(0, row_size+1)
        ax.set_ylim(0, row_size+1)
        for ext in file_type: fig.savefig(os.path.join(self.output_path, file_name+'.'+ext))

    def revert_modifications(self):
        self.adjacency_matrix = self._unmodified_adjmat.copy()
        self.isAdjMatModified = False

    def suppression_to_inhibition(self, k: float)->np.ndarray:
        """Suppress inhibition of the network.
        
        Replace the negative synaptic weights wij- by ( wij- + k * sigma_inh ).
        If wij- exceeds zero, replace it by zero to prevent convertion to excitatory links.
        Modified the adjacency matrix in place.
        Revert to the unmodified network by calling `Network.revert_modifications()`.
        The effect of suppression is not accumulated.

        Parameters
        ----------
        k : float
            suppression level

        Returns
        -------
        np.ndarray
            adjacency matrix of the suppressed network
        """
        if not self.isAdjMatModified: self._unmodified_adjmat = self.adjacency_matrix.copy()
        self.isAdjMatModified = True
        self.adjacency_matrix = self._unmodified_adjmat.copy()
        sigma_inh = self.statistics_of_synaptic_weights('inh')[1]
        for i in range(self.number_of_neurons):
            for j in range(self.number_of_neurons):
                if self.adjacency_matrix[i][j] < 0:
                    self.adjacency_matrix[i][j] += k * sigma_inh
                    if self.adjacency_matrix[i][j] > 0:
                        self.adjacency_matrix[i][j] = 0
        return self.adjacency_matrix


class NeuralDynamics:

    def __init__(self, input_folder=None, output_folder=None, spiking_data='spkt.txt', config='cont.dat',
                 membrane_potential='memp.dat', synaptic_current='curr.dat', delimiter='\t'):
        input_path, self.output_path = self._init_input_output_path(input_folder, output_folder)
        if type(config) == list:
            self.spike_count, self.spike_times, self.config = self._init_spiking_data_from_file(spiking_data, config, delimiter, input_path)
            # print('\'config\' input as a list of [N={}, dt={}, T={}].'.format(config[0], config[1], config[2]))
        elif type(config) == str:
            self.spike_count, self.spike_times, self.config = self._init_spiking_data_from_file(spiking_data, config, delimiter, input_path)
            if input_folder == None: input_folder = 'current directory'
            # print('\'config\' input from a local file: [{}] from [{}].'.format(config, input_folder))
        else:
            err = 'ArgumentTypeError: the input argument \'config\' must either be a list of [N, dt, T] or a string storing the path to a local file.'
            print(err); exit(1)
        if input_path == None:
            self.membrane_potential_file = membrane_potential
            self.synaptic_current_file = synaptic_current
        else:
            self.membrane_potential_file = input_path+membrane_potential
            self.synaptic_current_file = input_path+synaptic_current
        print('Spiking data input from a local file: [{}] from [{}].'.format(spiking_data, input_folder))
    
    def __del__(self):
        pass

    def _init_input_output_path(self, input_folder: str, output_folder: str):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        index = this_dir.rfind('\\')
        this_dir = this_dir[:index]
        if input_folder == None: input_folder = ''
        else: input_folder = input_folder + '\\'
        input_path = this_dir + '\\' + input_folder
        if output_folder == None: output_folder = ''
        else: output_folder = output_folder + '\\'
        output_path = this_dir + '\\' + output_folder
        try: os.mkdir(output_path)
        except FileExistsError: pass
        return input_path, output_path

    def _init_spiking_data_from_file(self, spiking_data_file: str, config_file, delimiter: str, input_path: str):
        if type(config_file) == list:
            config = config_file
            matrix_size = config[0]
        elif type(config_file) == str:
            try:
                with open(input_path+config_file, 'r') as fp:
                    reader = csv.reader(fp, delimiter='|')
                    config = np.array(list(reader), dtype=object)
                config = config[1]
                config[0] = int(config[0])   # config[0] = N
                config[1] = float(config[1]) # config[1] = dt
                config[2] = float(config[2]) # config[2] = T
                matrix_size = int(config[0])
            except FileNotFoundError:
                err = 'FileNotFoundError: configuration data file [{}] cannot be found.'.format(input_path+config_file)
                print(err); exit(1)
        try:
            with open(input_path+spiking_data_file, 'r') as fp:
                spike_times = np.empty((matrix_size), dtype=object)
                spike_count = np.zeros(matrix_size)
                reader = csv.reader(fp, delimiter=delimiter)
                counter = 0
                for row in reader:
                    try: spike_times[counter] = np.delete(np.array(list(row)).astype('float'), [0], 0)
                    except ValueError: pass
                    spike_count[counter] = int(row[0])
                    counter += 1
        except FileNotFoundError:
            err = 'FileNotFoundError: spiking data file [{}] cannot be found.'.format(input_path+spiking_data_file)
            print(err); exit(1)
        return spike_count, spike_times, config

    def _init_firing_rate(self, _spike_count=None, _config=None, _custom=False):
        if _custom == False: _spike_count, _config = self.spike_count, self.config
        return _spike_count / _config[2] * 1000 # config[2] = T

    def _init_firing_rate_change(self, neural_dynamics_original: object, specific_node=[]):
        if type(neural_dynamics_original) != NeuralDynamics:
            err = 'ArgumentTypeError: the input argument \'neural_dynamics_original\' must be an object of class:NeuralDynamics.'
            print(err); exit(1)
        spike_count_orig = neural_dynamics_original.spike_count
        config_orig = neural_dynamics_original.config
        if len(self.spike_count) != len(spike_count_orig):
            err = 'ArgumentValueError: the numbers of neurons from two numerical simulations do not match.'
            print(err); exit(1)
        firing_rate_orig = self._init_firing_rate(spike_count_orig, config_orig, custom=True)
        firing_rate = self._init_firing_rate()
        if specific_node == []: return firing_rate - firing_rate_orig
        else: return firing_rate[specific_node] - firing_rate_orig[specific_node]

    def _init_interspike_interval(self, specific_node=[]):
        if specific_node == []:
            interspike_interval = np.empty(self.config[0], dtype=object)
            for node in range(self.config[0]):
                try: interspike_interval[node] = np.array(np.diff(self.spike_times[node]), dtype=float)
                except ValueError: interspike_interval[node] = np.diff(np.array([0]))
        else:
            count = 0
            interspike_interval = np.empty(len(specific_node), dtype=object)
            for node in range(self.config[0]):
                if node in specific_node:
                    try: interspike_interval[count] = np.array(np.diff(self.spike_times[node]), dtype=float)
                    except ValueError: interspike_interval[count] = np.diff(np.array([0]))
                    count += 1
        interspike_interval = np.concatenate([item for item in interspike_interval.flatten()], 0) / 1000
        return interspike_interval

    def _init_membrane_potential(self, node_index: int, _membrane_potential=None, _config=None, _custom=False):
        if _custom == False: _membrane_potential, _config = self.membrane_potential_file, self.config
        time_series = Vector_fl()
        if time_series.read_from_binary(_membrane_potential, node_index, _config[0]) == 0: pass
        else:
            err = 'FileNotFoundError: time series of membrane potential [{}] cannot be found.'.format(
                  self.membrane_potential_file)
            print(err); exit(1)
        return np.array(time_series)

    def _init_synaptic_current(self, node_index: int, _synaptic_current=None, _config=None, _custom=False):
        if _custom == False: _synaptic_current, _config = self.synaptic_current_file, self.config
        time_series = Vector_fl()
        if time_series.read_from_binary(_synaptic_current, node_index, _config[0]) == 0: pass
        else:
            err = 'FileNotFoundError: time series of synaptic current [{}] cannot be found.'.format(
                  self.membrane_potential_file)
            print(err); exit(1)
        return np.array(time_series)

    def _init_membrane_potential_change(self, node_index: int, neural_dynamics_original: object):
        if type(neural_dynamics_original) != NeuralDynamics:
            err = 'ArgumentTypeError: the input argument \'neural_dynamics_original\' must be an object of class:NeuralDynamics.'
            print(err); exit(1)
        if self.config != neural_dynamics_original.config:
            err = 'ArgumentValueError: the configurations of two numerical simulations do not match.'
            print(err); exit(1)
        time_series = self._init_membrane_potential(node_index)
        time_series_orig = self._init_membrane_potential(node_index, neural_dynamics_original.membrane_potential_file,
                                                         neural_dynamics_original.config, custom=True)
        return time_series - time_series_orig, 

    def _plot_distribution(self, data, bin_size=0.15, color='b', marker_style='^', line_style='-',
                           plot_type='line', marker_size=8, marker_fill=True, line_width=2,
                           figsize=(9, 6), dpi=150, plot_label='', xlabel='', textbox='',
                           xlim=(None, None), ylim=(None, None), show_norm=False,
                           x_logscale=False, y_logscale=False, remove_zero_density=False,
                           file_name='plot_dist', file_label='', file_type=['svg','png'],
                           save_fig=True, show_fig=False, mpl_ax=None,
                           return_plot_data=False, return_area_under_graph=False):

        # Return ax
        def plot_subroutine(x, y, ax, ptype: str, plabel: str, plot_norm=False):
            if plot_norm: c='r'; marker='None'; ls='--'
            else: c=color; marker=marker_style; ls=line_style
            if marker_fill == True:
                if ptype == 'line': ax.plot(x, y, c=c, marker=marker, ms=marker_size, ls=ls, lw=line_width,
                                                label=plabel)
                else: ax.scatter(x, y, c=c, marker=marker, s=marker_size, label=plabel)
            else:
                if ptype == 'line': ax.plot(x, y, c=c, marker=marker, ms=marker_size, ls=ls, lw=line_width,
                                                mfc='none', label=plabel)
                else: ax.scatter(x, y, c=c, marker=marker, s=marker_size, facecolors='none', label=plabel)
            return ax
        
        # Return ax, plot_data, area_under_graph
        def make_plot(ax):
            min_datum = np.amin(data); max_datum = np.amax(data)
            if show_norm == True:
                mu = np.mean(data); sigma = np.std(data)
                x_value_norm = np.linspace(mu - 4*sigma, mu + 4*sigma, 150)
                ax = plot_subroutine(x_value_norm, stats.norm.pdf(x_value_norm, mu, sigma), ax,
                                     'line', 'Normal distribution', plot_norm=True)
            if x_logscale == False:
                number_of_bins = math.ceil((max_datum - min_datum) / bin_size)
                hist_density, bin_edges = np.histogram(data, bins=np.linspace(min_datum, min_datum+number_of_bins*bin_size,
                                                                              number_of_bins), density=True)
            elif x_logscale == True:
                min_datum = np.amin(np.array(data)[np.array(data) > 0])
                number_of_bins = math.ceil((math.log10(max_datum) - math.log10(min_datum)) / bin_size)
                hist_density, bin_edges = np.histogram(data, bins=np.logspace(math.log10(min_datum),
                                                       math.log10(min_datum)+number_of_bins*bin_size, number_of_bins))
            hist_density = np.array(hist_density, dtype=float)
            hist_density /= np.dot(hist_density, np.diff(bin_edges)) # normalization
            x_value = (bin_edges[1:] + bin_edges[:-1]) / 2
            area_under_graph = np.dot(hist_density, np.diff(bin_edges))
            if remove_zero_density == True:
                non_zero_points = np.argwhere(hist_density == 0)
                hist_density = np.delete(hist_density, non_zero_points, 0)
                x_value = np.delete(x_value, non_zero_points, 0)
            return plot_subroutine(x_value, hist_density, ax,
                                   plot_type, plot_label), np.array(list(zip(x_value,hist_density))), area_under_graph

        # Return ax
        def format_plot(ax):
            if x_logscale == True: ax.set_xscale('log')
            if y_logscale == True: ax.set_yscale('log')
            ax.set(xlabel=xlabel, ylabel='Probability density')
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
            ax.grid(True)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(reversed(handles), reversed(labels))#, title='Distribution')
            # textbox
            if textbox != '':
                props = dict(boxstyle='round', pad=0.1, facecolor='white', edgecolor='none', alpha=0.75)
                ax.text(0.00001, 1.05, textbox, fontsize=10, verticalalignment='top', transform=ax.transAxes, bbox=props)
            return ax

        if mpl_ax == None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            ax, plot_data, area_under_graph = make_plot(ax)
            ax = format_plot(ax)
            for ext in file_type:
                if file_label == '': file_path = file_name+'.'+ext
                else: file_path = file_name+'_'+file_label+'.'+ext
                if save_fig: fig.savefig(os.path.join(self.output_path, file_path))
            if show_fig: plt.show()
            plt.clf()
            if return_plot_data and not return_area_under_graph: return plot_data
            elif not return_plot_data and return_area_under_graph: return area_under_graph
            else: return plot_data, area_under_graph
        else:
            ax, plot_data, _ = make_plot(mpl_ax)
            if return_plot_data: return ax, plot_data
            else: return ax

    def _plot_time_series(self, node_index, data, time, xlim=(0, -1), ylim=(None, None),
                          mpl_ax=None, color='b', line_style='-', line_width=1,
                          figsize=(10, 6), dpi=150, plot_label='', ylabel='', textbox='',
                          file_name='time_series', file_label='', file_type=['svg','png'],
                          save_fig=True, show_fig=False):
        def format_plot(ax):
            ax.set(xlabel='Time (s)', ylabel=ylabel)
            if xlim[1] == -1: ax.set_xlim(0, self.config[2]/1000) # just fit
            else: ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
            ax.grid(True)
            handles, labels = ax.get_legend_handles_labels()
            if textbox != '':
                props = dict(boxstyle='round', pad=0.1, facecolor='white', edgecolor='none', alpha=0.75)
                ax.text(0.00001, 1.05, textbox, fontsize=10, verticalalignment='top', transform=ax.transAxes, bbox=props)
            return ax

        if mpl_ax == None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            ax.plot(time, data, color+line_style, lw=line_width, label='{} of node {}'.format(plot_label, node_index))
            ax = format_plot(ax)
            for ext in file_type:
                if file_label == '': file_path = file_name+'.'+ext
                else: file_path = file_name+'_'+file_label+'.'+ext
                if save_fig: fig.savefig(os.path.join(self.output_path, file_path))
            if show_fig: plt.show()
            plt.clf()
        else:
            mpl_ax.plot(time, data, color+line_style, lw=line_width, label='{} of node {}'.format(plot_label, node_index))
            return mpl_ax

    def firing_rate(self):
        """Return the firing rate of neurons in the network."""
        return self._init_firing_rate()

    def firing_rate_change(self, neural_dynamics_original: object)->np.ndarray:
        """Return the change in firing rate of neurons in two networks.

        Parameters
        ----------
        neural_dynamics_original : object
            object of class NeuralDynamics

        Returns
        -------
        numpy.ndarray
            array of changes in firing rate
        """
        return self._init_firing_rate_change(neural_dynamics_original)

    def plot_spike_raster(self, width_ratio=1, trim=(None, None),
                          file_name='plot_spike_raster', file_label='', file_type=['png']):
        """Plot a raster plot of spiking activity of the network.

        Parameters
        ----------
        width_ratio : int, optional
            horizontal scale of the plot, by default 1
        trim : tuple of float, optional
            range of simulation time to be drawn, input a negative value to show the full range, by default (0, -1)
        file_name : str, optional
            name or path of the output plot, by default 'plot_spike_raster'
        file_label : str, optional
            label of the file to be appended at the end of the file name, by default ''
        file_type : list, optional
            format(s)/extension(s) of the output plot, by default ['png']
        """
        print('Drawing graph: spike raster plot')
        fig, ax = plt.subplots(figsize=(9*width_ratio, 6), dpi=200)
        if trim[0] == None: trim=(0, trim[1])
        if trim[1] == None: trim=(trim[0], self.config[2]/1000)
        node = 0
        for nodal_spike_times in (self.spike_times / 1000):
            node += 1
            if trim[0] > 0:
                nodal_spike_times = nodal_spike_times[nodal_spike_times > float(trim[0])]
            if trim[1] < self.config[2]/1000:
                nodal_spike_times = nodal_spike_times[nodal_spike_times < float(trim[1])]
            lc = mcol.EventCollection(nodal_spike_times, lineoffset=node, linestyle='-',
                                      linelength=20, linewidth=1.5, color='black')
            ax.add_collection(lc)
        ax.set(xlabel='Time (s)', ylabel='Node index')
        ax.set_xlim(trim[0], trim[1])               # config[2] = T
        start_node, end_node = 0, self.config[0]    # config[0] = N
        ax.set_ylim(start_node-2, end_node+1)
        ax.grid(True)

        for ext in file_type:
            if file_label == '': file_path = file_name+'.'+ext
            else: file_path = file_name+'_'+file_label+'.'+ext
            fig.savefig(os.path.join(self.output_path, file_path))

    def plot_firing_rate_distribution(self, bin_size=0.1, mpl_ax=None, c='b', m='^', ls='-', pt='line',
                                      ms=8, mfc=True, lw=2, xlim=(0, 10), ylim=(0, None),
                                      figsize=(9, 6), dpi=150, show_norm=False, x_logscale=False,
                                      remove_zero_density=False, plot_label='Firing rate distribution',
                                      file_name='plot_firing_rate_dist', file_label='', file_type=['svg','png'],
                                      return_plot_data=False, save_plot_details=False):
        """Plot the distribution of firing rate of neurons in the network.

        Parameters
        ----------
        bin_size : float, optional
            size of bin, by default 0.1
        mpl_ax : matplotlib.axes.Axes, optional
            if a matplotlib Axes is given, append the plot to the Axes, by default None
        c : str, optional
            colour of lines and markers, by default 'b'
        m : str, optional
            type of marker, e.g., '^', 'o', etc, see matplotlib for details, by default '^'
        ls : str, optional
            style of line, e.g., '-', ':', etc, see matplotlib for details, by default '-'
        pt : str, optional
            plot type, options: 'line' or 'scatter', by default 'line'
        ms : int, optional
            size of marker, by default 8
        mfc : bool, optional
            marker face fill, solid markers if True, open markers if False, by default True
        lw : int, optional
            line weight, by default 2
        xlim : tuple, optional
            range of x-axis, by default (0, 10)
        ylim : tuple, optional
            range of y-axis, by default (0, None)
        figsize : tuple of float, optional
            figure size, by default (9, 6)
        dpi : int, optional
            dots per inch, by default 150
        show_norm : bool, optional
            show a normal distribution with mean and S.D. of the data if True, by default False
        x_logscale : bool, optional
            x-axis in log scale if True, by default False
        remove_zero_density : bool, optional
            remove zero density when `x_logscale` is enabled to maintain a connected graph, by default False
        plot_label : str, optional
            label of the plot to be shown in legend, by default 'Firing rate distribution'
        file_name : str, optional
            name or path of the output plot, by default 'plot_firing_rate_dist'
        file_label : str, optional
            label of the file to be appended at the end of the file name, by default ''
        file_type : list, optional
            format(s)/extension(s) of the output plot, by default ['svg','png']
        return_plot_data : bool, optional
            also return the data points of the plot if enabled, by default False
        save_plot_details : bool, optional
            save the details of the plot if enabled, by default False

        Returns
        -------
        matplotlib.axes.Axes
            if `mpl_ax` is provided, return the Axes with the plot appended
        numpy.ndarray
            if `return_plot_data` is True, also return the plot data
        """
        print('Drawing graph: distribution of firing rate')
        firing_rate = self._init_firing_rate()
        min_firing_rate = np.amin(firing_rate)
        max_firing_rate = np.amax(firing_rate)

        # Export plot as a SVG file
        if mpl_ax == None:
            textbox = 'T: {:.7} | dt: {:.5} | min: {:.5} | max: {:.5} | bin size: {:.5}'.format(float(self.config[2]),
                                                float(self.config[1]), float(min_firing_rate), float(max_firing_rate), float(bin_size))
            plot_data, area_under_curve = self._plot_distribution(firing_rate, bin_size, figsize=figsize, dpi=dpi,
                                                xlim=xlim, ylim=ylim, show_norm=show_norm, x_logscale=x_logscale,
                                                remove_zero_density=remove_zero_density,
                                                plot_label=plot_label, xlabel='Firing rate (Hz)', textbox=textbox,
                                                file_name=file_name, file_label=file_label, file_type=file_type)
            if save_plot_details:
                file_name = 'plot_details_fr'
                if file_label == '': file_path = file_name+'.txt'
                else: file_path = file_name+'_'+file_label+'.txt'
                with open(self.output_path+file_path, 'w') as fp:
                    fp.write('Bin size: {}\n'.format(bin_size))
                    fp.write('Min firing rate: {}\nMax firing rate: {}\n'.format(min_firing_rate, max_firing_rate))
                    fp.write('Total density by areal summation: {}\n'.format(area_under_curve))
                    fp.write('Show Gaussian distribution: {}'.format(str(show_norm)))

                file_name = 'plot_data_fr'
                if file_label == '': file_path = file_name+'.txt'
                else: file_path = file_name+'_'+file_label+'.txt'
                with open(self.output_path+file_path, 'w') as fp:
                    for pd in plot_data: fp.write('{}\t{}\n'.format(pd[0], pd[1]))

            if return_plot_data: return plot_data

        # Return plot as a matplotlib.axes
        else:
            return self._plot_distribution(firing_rate, bin_size, c, m, ls, pt, ms, mfc, lw,
                                           show_norm=show_norm, x_logscale=x_logscale,
                                           remove_zero_density=remove_zero_density, plot_label=plot_label,
                                           mpl_ax=mpl_ax, return_plot_data=return_plot_data)

    def plot_firing_rate_change_distribution(self, neural_dynamics_original: object,
                                             bin_size=0.1, mpl_ax=None, c='b', m='^', ls='-', pt='line',
                                             ms=8, mfc=True, lw=2, xlim=(None, None), ylim=(None, None),
                                             figsize=(9, 6), dpi=150, x_logscale=False, y_logscale=True,
                                             remove_zero_density=False, filter_neurons=[],
                                             plot_label='Change in firing rate distribution',
                                             file_name='plot_firing_rate_change_dist',
                                             file_label='', file_type=['svg','png'],
                                             return_plot_data=False, save_plot_details=False):
        """Plot the distribution of change in firing rate of neurons in two networks.

        Parameters
        ----------
        neural_dynamics_original : object
            _description_
        bin_size : float, optional
            size of bin, by default 0.1
        mpl_ax : matplotlib.axes.Axes, optional
            if a matplotlib Axes is given, append the plot to the Axes, by default None
        c : str, optional
            colour of lines and markers, by default 'b'
        m : str, optional
            type of marker, e.g., '^', 'o', etc, see matplotlib for details, by default '^'
        ls : str, optional
            style of line, e.g., '-', ':', etc, see matplotlib for details, by default '-'
        pt : str, optional
            plot type, options: 'line' or 'scatter', by default 'line'
        ms : int, optional
            size of marker, by default 8
        mfc : bool, optional
            marker face fill, solid markers if True, open markers if False, by default True
        lw : int, optional
            line weight, by default 2
        xlim : tuple, optional
            range of x-axis, by default (None, None)
        ylim : tuple, optional
            range of y-axis, by default (None, None)
        figsize : tuple of float, optional
            figure size, by default (9, 6)
        dpi : int, optional
            dots per inch, by default 150
        x_logscale : bool, optional
            x-axis in log scale if True, by default False
        x_logscale : bool, optional
            y-axis in log scale if True, by default True
        remove_zero_density : bool, optional
            remove zero density when `x_logscale` is enabled to maintain a connected graph, by default False
        filter_neurons : list, optional
            list of neuron node indices to be considered, by default [] (include all neurons)
        plot_label : str, optional
            label of the plot to be shown in legend, by default 'Change in firing rate distribution'
        file_name : str, optional
            name or path of the output plot, by default 'plot_firing_rate_change_dist'
        file_label : str, optional
            label of the file to be appended at the end of the file name, by default ''
        file_type : list, optional
            format(s)/extension(s) of the output plot, by default ['svg','png']
        return_plot_data : bool, optional
            also return the data points of the plot if enabled, by default False
        save_plot_details : bool, optional
            save the details of the plot if enabled, by default False

        Returns
        -------
        matplotlib.axes.Axes
            if `mpl_ax` is provided, return the Axes with the plot appended
        numpy.ndarray
            if `return_plot_data` is True, also return the plot data
        """
        print('Drawing graph: distribution of changes in firing rate')
        if filter_neurons == []: change_in_firing_rate = self._init_firing_rate_change(neural_dynamics_original)
        else: change_in_firing_rate = self._init_firing_rate_change(neural_dynamics_original, filter_neurons)
        min_change_in_firing_rate = np.amin(change_in_firing_rate)
        max_change_in_firing_rate = np.amax(change_in_firing_rate)

        # Export plot as a SVG file
        if mpl_ax == None:
            textbox = 'T: {:.7} | dt: {:.5} | min: {:.5} | max: {:.5} | bin size: {:.5}'.format(float(self.config[2]),
                                                        float(self.config[1]), float(min_change_in_firing_rate),
                                                        float(max_change_in_firing_rate), float(bin_size))
            plot_data, area_under_curve = self._plot_distribution(change_in_firing_rate, bin_size,
                                                        figsize=figsize, dpi=dpi, xlim=xlim, ylim=ylim,
                                                        x_logscale=x_logscale, y_logscale=y_logscale,
                                                        remove_zero_density=remove_zero_density, plot_label=plot_label,
                                                        xlabel='Change in firing rate (Hz)', textbox=textbox,
                                                        file_name=file_name, file_label=file_label, file_type=file_type)
            if save_plot_details:
                file_name = 'plot_details_fr_chg'
                if file_label == '': file_path = file_name+'.txt'
                else: file_path = file_name+'_'+file_label+'.txt'
                with open(self.output_path+file_path, 'w') as fp:
                    fp.write('Bin size: {}\n'.format(bin_size))
                    fp.write('Min change in firing rate: {}\nMax change in firing rate: {}\n'.format(min_change_in_firing_rate,
                                                                                                    max_change_in_firing_rate))
                    fp.write('Total density by areal summation: {}'.format(area_under_curve))

                file_name = 'plot_data_fr_chg'
                if file_label == '': file_path = file_name+'.txt'
                else: file_path = file_name+'_'+file_label+'.txt'
                with open(self.output_path+file_path, 'w') as fp:
                    for pd in plot_data: fp.write('{}\t{}\n'.format(pd[0], pd[1]))

            if return_plot_data: return plot_data

        # Return plot as a matplotlib.axes
        else:
            return self._plot_distribution(change_in_firing_rate, bin_size, c, m, ls, pt, ms, mfc, lw,
                                           x_logscale=x_logscale, plot_label=plot_label,
                                           remove_zero_density=remove_zero_density,
                                           mpl_ax=mpl_ax, return_plot_data=return_plot_data)

    def plot_interspike_interval_distribution(self, bin_size=0.025, mpl_ax=None, c='b', m='^', ls='-',
                                              pt='line', ms=8, mfc=True, lw=2, xlim=(0.0005, None), ylim=(0, None),
                                              figsize=(9, 6), dpi=150, x_logscale=True, y_logscale=False,
                                              remove_zero_density=False, filter_neurons=[],
                                              plot_label='Log ISI distribution',
                                              file_name='plot_isi_dist', file_label='', file_type=['svg','png'],
                                              return_plot_data=False, save_plot_details=False):
        """Plot the distribution of inter-spike interval of all neurons in networks.

        Parameters
        ----------
        bin_size : float, optional
            size of bin, by default 0.025
        mpl_ax : matplotlib.axes.Axes, optional
            if a matplotlib Axes is given, append the plot to the Axes, by default None
        c : str, optional
            colour of lines and markers, by default 'b'
        m : str, optional
            type of marker, e.g., '^', 'o', etc, see matplotlib for details, by default '^'
        ls : str, optional
            style of line, e.g., '-', ':', etc, see matplotlib for details, by default '-'
        pt : str, optional
            plot type, options: 'line' or 'scatter', by default 'line'
        ms : int, optional
            size of marker, by default 8
        mfc : bool, optional
            marker face fill, solid markers if True, open markers if False, by default True
        lw : int, optional
            line weight, by default 2
        xlim : tuple, optional
            range of x-axis, by default (0.0005, None)
        ylim : tuple, optional
            range of y-axis, by default (0, None)
        figsize : tuple of float, optional
            figure size, by default (9, 6)
        dpi : int, optional
            dots per inch, by default 150
        x_logscale : bool, optional
            x-axis in log scale if True, by default True
        x_logscale : bool, optional
            y-axis in log scale if True, by default False
        remove_zero_density : bool, optional
            remove zero density when `x_logscale` is enabled to maintain a connected graph, by default False
        filter_neurons : list, optional
            list of neuron node indices to be considered, by default [] (include all neurons)
        plot_label : str, optional
            label of the plot to be shown in legend, by default 'Log ISI distribution'
        file_name : str, optional
            name or path of the output plot, by default 'plot_isi_dist'
        file_label : str, optional
            label of the file to be appended at the end of the file name, by default ''
        file_type : str, optional
            format(s)/extension(s) of the output plot, by default 'svg'
        return_plot_data : bool, optional
            also return the data points of the plot if enabled, by default False
        save_plot_details : bool, optional
            save the details of the plot if enabled, by default False

        Returns
        -------
        matplotlib.axes.Axes
            if `mpl_ax` is provided, return the Axes with the plot appended
        numpy.ndarray
            if `return_plot_data` is True, also return the plot data
        """
        print('Drawing graph: distribution of inter-spike interval (ISI)')
        interspike_interval = self._init_interspike_interval(filter_neurons)
        min_interspike_interval = np.amin(interspike_interval)
        max_interspike_interval = np.amax(interspike_interval)
        # Export plot as a SVG file
        if mpl_ax == None:
            if x_logscale == True:
                textbox = 'T: {:.7} | dt: {:.5} | min: {:.5} | max: {:.5} | bin size (in log scale): {:.5}'.format(float(self.config[2]),
                                                        float(self.config[1]), float(min_interspike_interval),
                                                        float(max_interspike_interval), float(bin_size))
            else: textbox = 'T: {:.7} | dt: {:.5} | min: {:.5} | max: {:.5} | bin size: {:.5}'.format(float(self.config[2]),
                                                        float(self.config[1]), float(min_interspike_interval),
                                                        float(max_interspike_interval), float(bin_size))
            plot_data, area_under_curve = self._plot_distribution(interspike_interval, bin_size,
                                                        figsize=figsize, dpi=dpi, xlim=xlim, ylim=ylim,
                                                        x_logscale=x_logscale, y_logscale=y_logscale,
                                                        plot_label=plot_label, xlabel='ISI (s)', textbox=textbox,
                                                        file_name=file_name, file_label=file_label, file_type=file_type)
            if save_plot_details:
                file_name = 'plot_details_isi'
                if file_label == '': file_path = file_name+'.txt'
                else: file_path = file_name+'_'+file_label+'.txt'
                with open(self.output_path+file_path, 'w') as fp_info:
                    fp_info.write('Bin size in log scale: {}\n'.format(bin_size))
                    fp_info.write('Min ISI: {}\nMax ISI: {}\n'.format(min_interspike_interval, max_interspike_interval))
                    fp_info.write('Total density by areal summation: {}'.format(area_under_curve))

                file_name = 'plot_data_isi'
                if file_label == '': file_path = file_name+'.txt'
                else: file_path = file_name+'_'+file_label+'.txt'
                with open(self.output_path+file_path, 'w') as fp:
                    for pd in plot_data: fp.write('{}\t{}\n'.format(pd[0], pd[1]))

        # Return plot as a matplotlib.axes
        else:
            return self._plot_distribution(interspike_interval, bin_size, c, m, ls, pt, ms, mfc, lw,
                                           x_logscale=True, plot_label=plot_label,
                                           mpl_ax=mpl_ax, return_plot_data=return_plot_data)

    def plot_membrane_potential_time_series(self, node_index: int, trim=(0, -1), mpl_ax=None, c='b',
                                            ls='-', lw=1, figsize=(12, 6), dpi=150, ylim=(None, None),
                                            plot_label='Membrane potential', file_name='plot_potential',
                                            file_label='', file_type=['svg','png']):
        """Plot the time series of membrane potential of a sepecific neuron node in the network.

        Parameters
        ----------
        node_index : int
            index of neuron node to be plot, starts from 1
        trim : tuple of float, optional
            range of simulation time to be drawn, input a negative value to show the full range, by default (0, -1)
        mpl_ax : matplotlib.axes.Axes, optional
            if a matplotlib Axes is given, append the plot to the Axes, by default None
        mpl_ax : matplotlib.axes.Axes, optional
            if a matplotlib Axes is given, append the plot to the Axes, by default None
        figsize : tuple of float, optional
            size, by default (12, 6)
        dpi : int, optional
            dots per inch, by default 150
        ylim : tuple, optional
            range of y-axis, by default (None, None)
        plot_label : str, optional
            label of the plot to be shown in legend, by default 'Membrane potential'
        file_name : str, optional
            name or path of the output plot, by default 'plot_potential'
        file_label : str, optional
            label of the file to be appended at the end of the file name, by default ''
        file_type : list, optional
            format(s)/extension(s) of the output plot, by default ['svg','png']

        Returns
        -------
        matplotlib.axes.Axes
            if `mpl_ax` is provided, return the Axes with the plot appended
        """
        print('Drawing graph: time series of membrane potential of node {}'.format(node_index))
        time_series = self._init_membrane_potential(node_index)
        time = np.arange(0, len(time_series)) * self.config[1] / 1000 # total time steps * dt in seconds
        textbox = 'Node index: {:d} | T: {:.7} | dt: {:.5}'.format(int(node_index), float(self.config[2]), float(self.config[1]))

        # Export plot as a SVG file
        if mpl_ax == None:
            self._plot_time_series(node_index, time_series, time, trim, ylim, None, c, ls, lw,
                                   figsize, dpi, plot_label, 'Membrane potential (mV)', textbox,
                                   file_name+'_'+str(node_index), file_label, file_type)
        # Return plot as a matplotlib.axes
        else:
            return self._plot_time_series(node_index, time_series, time, trim, ylim, mpl_ax,
                                          c, ls, lw, plot_label)

    def plot_membrane_potential_change_time_series(self, node_index: int, neural_dynamics_original: object,
                                                   trim=(0, -1), mpl_ax=None, c='b', ls='-', lw=1,
                                                   figsize=(12, 6), dpi=150, ylim=(None, None),
                                                   plot_label='Change in membrane potential',
                                                   file_name='plot_potential_change',
                                                   file_label='', file_type=['svg','png']):
        """Plot the time series of change in membrane potential of a sepecific neuron node in two networks.

        Parameters
        ----------
        node_index : int
            index of neuron node to be plot, starts from 1
        neural_dynamics_original : object
            object of class NeuralDynamics
        trim : tuple, optional
            range of simulation time to be drawn, input a negative value to show the full range, by default (0, -1)
        mpl_ax : matplotlib.axes.Axes, optional
            if a matplotlib Axes is given, append the plot to the Axes, by default None
        c : str, optional
            colour of lines, by default 'b'
        ls : str, optional
            _description_, by default '-'
        lw : float, optional
            line weight, by default 1
        figsize : tuple of float, optional
            figure size, by default (12, 6)
        dpi : int, optional
            dots per inch, by default 150
        ylim : tuple, optional
            range of y-axis, by default (None, None)
        plot_label : str, optional
            label of the plot to be shown in legend, by default 'Change in membrane potential'
        file_name : str, optional
            name or path of the output plot, by default 'plot_potential_change'
        file_label : str, optional
            label of the file to be appended at the end of the file name, by default ''
        file_type : list, optional
            format(s)/extension(s) of the output plot, by default ['svg','png']

        Returns
        -------
        matplotlib.axes.Axes
            if `mpl_ax` is provided, return the Axes with the plot appended
        """
        print('Drawing graph: time series of changes in membrane potential of node {}'.format(node_index))
        time_series = self._init_membrane_potential_change(node_index, neural_dynamics_original)
        time = np.arange(0, len(time_series)) * self.config[1] / 1000 # total time steps * dt in seconds
        textbox = 'Node index: {:d} | T: {:.7} | dt: {:.5}'.format(int(node_index), float(self.config[2]), float(self.config[1]))
        # Export plot as a SVG file
        if mpl_ax == None:
            self._plot_time_series(node_index, time_series, time, trim, ylim, None, c, ls, lw, figsize, dpi,
                                   plot_label, 'Change in membrane potential (mV)', textbox,
                                   file_name+'_'+str(node_index), file_label, file_type)
        # Return plot as a matplotlib.axes
        else:
            return self._plot_time_series(node_index, time_series, time, trim, ylim, mpl_ax, c, ls, lw,
                                          plot_label)

    def plot_potential_current_graph(self, node_index: int, start_time=None, end_time=None,
                                     figsize=(9, 9), dpi=150, plot_label='', file_name='plot_potential_current',
                                     file_label='', file_type=['png']):
        """Plot the graph of membrane potential against synaptic current of a specific neuron in the network.

        Parameters
        ----------
        node_index : int
            index of neuron node to be plot, starts from 1
        start_time : _type_, optional
            starting time in millisecond (ms) of the plot, set to None to plot from the begining, by default None
        end_time : _type_, optional
            ending time in millisecond (ms) of the plot, set to None to plot until the end, by default None
        figsize : tuple, optional
            figure size, by default (9, 9)
        dpi : int, optional
            dots per inch, by default 150
        plot_label : str, optional
            label of the plot to be shown in legend, by default ''
        file_name : str, optional
            name or path of the output plot, by default 'plot_potential_current'
        file_label : str, optional
            label of the file to be appended at the end of the file name, by default ''
        file_type : list, optional
            format(s)/extension(s) of the output plot, by default ['png']
        """
        print('Drawing graph: membrane potential against synaptic current for neuron node {}'.format(node_index))
        membrane_potential = self._init_membrane_potential(node_index)
        synaptic_current = self._init_synaptic_current(node_index)

        if start_time == None: start_time = 0
        if end_time == None: end_time = self.config[2]
        beg = int(start_time*1000/self.config[1])
        end = int(end_time*1000/self.config[1])

        fig, ax = plt.subplots(figsize=figsize, dpi=100)
        lc = _colorline(membrane_potential[beg:end], synaptic_current[beg:end], cmap='copper', linewidth=1)
        cbar = plt.colorbar(lc)
        cbar.ax.set_yticklabels(['{:.5}'.format(x) for x in np.arange(start_time,end_time+(end_time-start_time)/5,
                                                                      (end_time-start_time)/5)])
        ax.autoscale_view()
        for ext in file_type:
            if file_label == '': file_path = file_name+'_'+str(node_index)+'.'+ext
            else: file_path = file_name+'_'+str(node_index)+'_'+file_label+'.'+ext
            fig.savefig(os.path.join(self.output_path, file_path))


class Grapher:

    def __init__(self) -> None:
        self.xdata      = None
        self.ydata      = None
        self.fig        = None
        self.ax         = None
        self.x          = None
        self.y          = None

        self.c          = 'b'
        self.m          = '^'
        self.ms         = 5
        self.ls         = 'none'
        self.lw         = 2
        self.plotlabel  = ''
        self.axislabel  = ['','']
        self.grid       = True
        self.legend     = False
        self.textbox    = ''

        self.xlim       = (None, None)
        self.ylim       = (None, None)
        self.xlogscale  = False
        self.ylogscale  = False
        self.xsymlogscale   = False
        self.xlinthresh = 1
        self.ysymlogscale   = False
        self.ylinthresh = 1

    def import_fig(self, ax, fig=None):
        self.fig = fig
        self.ax = ax

    def create_fig(self, figsize=(7, 7), dpi=150):
        self.__init__()
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)

    def stylize_plot(self, **kwargs):
        """Stylize matplotlib axes.

        Parameters
        ----------
        color | c : str
            color of markers and lines, e.g., `'b', 'C1', 'darkorange'`
        marker | m : str
            marker style, e.g., `'o', '^', 'D', ','`
        markersize | ms : float
            marker size
        linestyle | ls : str
            line style, e.g., `'-', '--', '-.', ':'`
        lineweight | lw : float
            line weight
        grid : bool
            grid on/off, default: `True`
        """
        for key, value in kwargs.items():
            if key == 'color' or key == 'c': self.c = value
            elif key == 'marker' or key == 'm': self.m = value
            elif key == 'markersize' or key == 'ms': self.ms = value
            elif key == 'linestyle' or key == 'ls': self.ls = value
            elif key == 'lineweight' or key == 'lw': self.lw = value
            elif key == 'grid': self.grid = value
            else: print('The optional argument: [{}] is not supported'.format(key))

    def label_plot(self, **kwargs):
        """Label matplotlib axes.

        Parameters
        ----------
        plotlabel : str
            plot label to be displayed in legend
        axislabel : list of str
            x-axis and y-axis label, e.g., `['time', 'voltage']`
        xlabel : str
            x-axis label
        ylabel : str
            y-axis label
        legend : bool
            legend on/off, default: `False`
        textbox : str
            plot information to be displayed at top-left corner
        """
        for key, value in kwargs.items():
            if key == 'plotlabel': self.plotlabel = value
            elif key == 'axislabel':
                self.axislabel[0] = '{} {}'.format(value[0], self.axislabel[0])
                self.axislabel[1] = '{} {}'.format(value[1], self.axislabel[1])
            elif key == 'xlabel': self.axislabel[0] = '{} {}'.format(value[0], self.axislabel[0])
            elif key == 'ylabel': self.axislabel[1] = '{} {}'.format(value[1], self.axislabel[1])
            elif key == 'legend': self.legend = value
            elif key == 'textbox': self.textbox = value
            else: print('The optional argument: [{}] is not supported'.format(key))

    def plot(self):
        # self.ax.tick_params(axis='both', direction='in', which='both')
        # locmin = tck.LogLocator(base=10.0, subs=(.2,.4,.6,.8), numticks=12)
        # self.ax.yaxis.set_minor_locator(locmin)
        # self.ax.yaxis.set_minor_formatter(tck.NullFormatter())
        # self.ax.yaxis.set_minor_locator(tck.AutoMinorLocator(4))
        self.ax.minorticks_on()
        if not all(x is None for x in self.xlim): self.ax.set_xlim(self.xlim)
        if not all(x is None for x in self.ylim): self.ax.set_ylim(self.ylim)
        if self.ylogscale: self.ax.set_yscale('log')
        elif self.ysymlogscale: self.ax.set_yscale('symlog', linthresh=self.ylinthresh)
        if self.xlogscale: self.ax.set_xscale('log')
        elif self.xsymlogscale: self.ax.set_xscale('symlog', linthresh=self.xlinthresh)
        self.ax.plot(self.x, self.y, c=self.c, marker=self.m, markersize=self.ms,
                     ls=self.ls, lw=self.lw, label=self.plotlabel, zorder=2)

    def _set_fmt(self):
        if self.grid: self.ax.grid(True)
        if self.legend: self.ax.legend()
        self.ax.set(xlabel=self.axislabel[0], ylabel=self.axislabel[1])
        if self.textbox != '':
            props = dict(boxstyle='round', pad=0.1, facecolor='white', edgecolor='none', alpha=0.75)
            self.ax.text(0.00001, 1.05, self.textbox, fontsize=10, verticalalignment='top',
                         transform=self.ax.transAxes, bbox=props)

    def save_fig(self, filename: str, label='', ext='png', path='', tight_layout=True):
        """Save figure into file.

        Parameters
        ----------
        filename : str
            name of the output file
        label : str, optional
            label attached at the end of the file name, by default ''
        ext : str, optional
            file extension, by default 'png'
        path : str, optional
            path to the output file, by default ''
        tight_layout : bool, optional
            enable tight layout, by default True
        """
        self._set_fmt()
        if label != '': filename += '_' + label
        if ext != '': filename += '.' + ext
        if tight_layout: plt.tight_layout()
        self.fig.savefig(os.path.join(path, filename))

    def show_fig(self):
        self._set_fmt()
        plt.show()

    def return_fig(self):
        self._set_fmt()
        return self.fig, self.ax

class GraphDataRelation(Grapher):

    def __init__(self) -> None:
        """Graph the relation between two matrices.

        Step:
        1. `creat_fig()` or `import_fig()`
        2. `add_data()`
        3. (optional)
        4. `plot()`
        5. `save_fig()` or `return_fig()` or `show_fig()`
        
        Optional:
        - `stylize_plot()`
        - `label_plot()`
        - `draw_xyline()`
        - `fit_linear()`

        Example:
        >>> A = [[1, 3], [2, 2]]
        ... B = [[2, 3], [1, 4]]
        >>> g = GraphMatrixRelation()
        ... g.create_fig()
        ... g.add_data(A, B)
        ... coef = g.fit_linear()
        ... g.plot()
        ... g.show_fig()
        """
        super().__init__()
        self.m           = 'o'
        self.ms          = 1.5
        self.ls          = 'none'

    def add_data(self, xdata, ydata, **kwargs):
        """Add data to be plotted.

        Parameters
        ----------
        xdata : np.ndarray
            data in x-axis
        ydata : np.ndarray
            data in y-axis
        xlim : tuple
            horizontal plot range, default: fitted to data
        ylim : tuple
            vertical plot range, default: fitted to data
        xlogscale : bool
            use log scale in x-axis, default: `False`
        ylogscale : bool
            use log scale in y-axis, default: `False`
        xsymlogscale : bool
            use symmetric log scale in x-axis (should be used with `xlinthresh`), default: `False`
        xlinthresh : float
            the threshold of linear range when using `xsymlogscale`, default: 1
        ysymlogscale : bool
            use symmetric log scale in y-axis (should be used with `ylinthresh`), default: `False`
        ylinthresh : float
            the threshold of linear range when using `ysymlogscale`, default: 1
        """
        for key, value in kwargs.items():
            if key == 'xlim': self.xlim = value
            elif key == 'ylim': self.ylim = value
            elif key == 'xlogscale': self.xlogscale = value
            elif key == 'ylogscale': self.ylogscale = value
            elif key == 'xsymlogscale': self.xsymlogscale = value
            elif key == 'xlinthresh': self.xlinthresh = value
            elif key == 'ysymlogscale': self.ysymlogscale = value
            elif key == 'ylinthresh': self.ylinthresh = value
            else: print('The optional argument: [{}] is not supported'.format(key))
        self.x = np.array(xdata).flatten()
        self.y = np.array(ydata).flatten()

    def draw_xyline(self, color='k'):
        min_elem = min(np.amin(self.x), np.amin(self.y))
        max_elem = max(np.amax(self.x), np.amax(self.y))
        elem_range = np.linspace(min_elem, max_elem)
        self.ax.plot(elem_range, elem_range, c=color, ls='--', lw=1, zorder=1)

    def fit_linear(self, color='darkorange') -> tuple:
        min_elem = min(np.amin(self.x), np.amin(self.y))
        max_elem = max(np.amax(self.x), np.amax(self.y))
        elem_range = np.linspace(min_elem, max_elem)
        coef = np.polyfit(self.x, self.y, 1)  # slope = coef[0], y-intercept = coef[1]
        poly1d_fn = np.poly1d(coef)
        self.ax.plot(elem_range, poly1d_fn(elem_range), c=color, ls='-.', lw=1.25, zorder=3)
        return coef

class GraphMatrixRelation(GraphDataRelation):

    def __init__(self) -> None:
        """Graph the relation between two matrices.

        Step:
        1. `creat_fig()` or `import_fig()`
        2. `add_data()`
        3. (optional)
        4. `plot()`
        5. `save_fig()` or `return_fig()` or `show_fig()`
        
        Optional:
        - `stylize_plot()`
        - `label_plot()`
        - `draw_xyline()`
        - `fit_linear()`

        Example:
        >>> A = [[1, 3], [2, 2]]
        ... B = [[2, 3], [1, 4]]
        >>> g = GraphMatrixRelation()
        ... g.create_fig()
        ... g.add_data(A, B)
        ... coef = g.fit_linear()
        ... g.plot()
        ... g.show_fig()
        """
        super().__init__()
        self.onlyOffDiag = False
        self.onlyDiag    = False
        self.ms          = 1

    def add_data(self, xdata, ydata, **kwargs):
        """Add data to be plotted.

        Parameters
        ----------
        xdata : np.ndarray
            data in x-axis
        ydata : np.ndarray
            data in y-axis
        onlyOffDiag : bool
            show only off-diagonal elements, default: `False`
        onlyDiag : bool
            show only diagonal elements, default: `False`
        xlim : tuple
            horizontal plot range, default: fitted to data
        ylim : tuple
            vertical plot range, default: fitted to data
        xlogscale : bool
            use log scale in x-axis, default: `False`
        ylogscale : bool
            use log scale in y-axis, default: `False`
        xsymlogscale : bool
            use symmetric log scale in x-axis (should be used with `xlinthresh`), default: `False`
        xlinthresh : float
            the threshold of linear range when using `xsymlogscale`, default: 1
        ysymlogscale : bool
            use symmetric log scale in y-axis (should be used with `ylinthresh`), default: `False`
        ylinthresh : float
            the threshold of linear range when using `ysymlogscale`, default: 1
        """
        for key, value in kwargs.items():
            if key == 'onlyDiag': self.onlyDiag = value
            elif key == 'onlyOffDiag': self.onlyOffDiag = value
            elif key == 'xlim': self.xlim = value
            elif key == 'ylim': self.ylim = value
            elif key == 'xlogscale': self.xlogscale = value
            elif key == 'ylogscale': self.ylogscale = value
            elif key == 'xsymlogscale': self.xsymlogscale = value
            elif key == 'xlinthresh': self.xlinthresh = value
            elif key == 'ysymlogscale': self.ysymlogscale = value
            elif key == 'ylinthresh': self.ylinthresh = value
            else: print('The optional argument: [{}] is not supported'.format(key))
        self.xdata = np.array(xdata)
        self.ydata = np.array(ydata)
        if self.onlyOffDiag: # keep only off-diagonals
            self.x = self._rm_diag(self.xdata).flatten()
            self.y = self._rm_diag(self.ydata).flatten()
            self.axislabel[0] += ' (off-diagonals)'
            self.axislabel[1] += ' (off-diagonals)'
        elif self.onlyDiag: # keep only diagonals
            self.x = np.diag(self.xdata)
            self.y = np.diag(self.ydata)
            self.axislabel[0] += ' (diagonals)'
            self.axislabel[1] += ' (diagonals)'
        else:
            self.x = self.xdata.flatten()
            self.y = self.ydata.flatten()

    def _rm_diag(self, A: np.ndarray) -> np.ndarray:
        if A.shape[0] == A.shape[1]:
            return A[~np.eye(A.shape[0],dtype=bool)].reshape(A.shape[0],-1)
        else:
            err = 'ArgumentValueError: the input matrix must be square.'
            print(err); exit(1)

class GraphDensityDistribution(Grapher):

    def __init__(self) -> None:
        """Graph the density distribution of a set of data.

        Step:
        1. `creat_fig()` or `import_fig()`
        2. `add_data()`
        3. (optional)
        4. `plot()`
        5. `save_fig()` or `return_fig()` or `show_fig()`

        Optional:
        - `stylize_plot()`
        - `label_plot()`
        - `draw_gaussian()`

        Example:
        >>> A = np.random.normal(0, 10, 5000)
        >>> g = GraphMatrixRelation()
        ... g.create_fig()
        ... g.add_data(A, 0.1)
        ... g.plot()
        ... g.show_fig()
        """
        super().__init__()
        self.ms = 7.5
        self.lw = 2
        self.ls = '-'

    def add_data(self, data, binsize: float, **kwargs):
        """Add data to be plotted.

        Parameters
        ----------
        data : np.ndarray
            data
        xlim : tuple
            horizontal plot range, default: fitted to data
        ylim : tuple
            vertical plot range, default: fitted to data
        xlogscale : bool
            use log scale in x-axis, default: `False`
        ylogscale : bool
            use log scale in y-axis, default: `False`
        xsymlogscale : bool
            use symmetric log scale in x-axis (should be used with `xlinthresh`), default: `False`
        xlinthresh : float
            the threshold of linear range when using `xsymlogscale`, default: 1
        """
        for key, value in kwargs.items():
            if key == 'xlim': self.xlim = value
            elif key == 'ylim': self.ylim = value
            elif key == 'xlogscale': self.xlogscale = value
            elif key == 'ylogscale': self.ylogscale = value
            elif key == 'xsymlogscale': self.xsymlogscale = value
            elif key == 'xlinthresh': self.xlinthresh = value
            else: print('The optional argument: [{}] is not supported'.format(key))
        self.xdata = np.array(data).flatten()
        min_elem = np.amin(self.xdata); max_elem = np.amax(self.xdata)
        if self.xlogscale:
            pos_nearzero_datum = np.amin(np.array(data)[np.array(data) > 0])
            number_of_bins = math.ceil((math.log10(max_elem) - math.log10(pos_nearzero_datum)) / binsize)
            density, binedge = np.histogram(
                data, bins=np.logspace(math.log10(pos_nearzero_datum), math.log10(pos_nearzero_datum)
                                       + number_of_bins*binsize, number_of_bins)
            )
        elif self.xsymlogscale:
            # pos_nearzero_datum = np.amin(np.array(data)[np.array(data) > 0])
            # neg_nearzero_datum = np.amax(np.array(data)[np.array(data) < 0])
            pos_nearzero_datum = self.xlinthresh
            neg_nearzero_datum = -self.xlinthresh
            lin_bins_amt = math.ceil((max_elem - min_elem) / binsize)
            pos_bins_amt = math.ceil((math.log10(max_elem) - math.log10(pos_nearzero_datum)) / binsize)
            neg_bins_amt = math.ceil((math.log10(-min_elem) - math.log10(-neg_nearzero_datum)) / binsize)
            linbins = np.linspace(neg_nearzero_datum, pos_nearzero_datum, lin_bins_amt)
            pos_logbins = np.logspace(math.log10(pos_nearzero_datum), math.log10(pos_nearzero_datum)+pos_bins_amt*binsize, pos_bins_amt)
            neg_logbins = -np.flip(np.logspace(math.log10(-neg_nearzero_datum), math.log10(-neg_nearzero_datum)+neg_bins_amt*binsize, neg_bins_amt))
            density, binedge = np.histogram(data, bins=np.concatenate((neg_logbins, linbins, pos_logbins)), density=True)
        else:
            bins_amt = math.ceil((max_elem - min_elem) / binsize)
            density, binedge = np.histogram(data, bins=np.linspace(min_elem, min_elem+bins_amt*binsize, bins_amt), density=True)
        density = np.array(density, dtype=float)
        density /= np.dot(density, np.diff(binedge)) # normalization
        self.x = (binedge[1:] + binedge[:-1]) / 2
        self.y = density

    def draw_gaussian(self, **kwargs):
        c = 'r'; lw = 2.5
        for key, value in kwargs.items():
            if key == 'color' or key == 'c': c = value
            if key == 'lineweight' or key == 'lw': lw = value
        mu = np.mean(self.xdata); sigma = np.std(self.xdata)
        norm_xval = np.linspace(mu - 4*sigma, mu + 4*sigma, 150)
        self.ax.plot(norm_xval, stats.norm.pdf(norm_xval, mu, sigma), '--', c=c, lw=lw, label='Normal distribution', zorder=1)



# Other tools
def plot_distribution(data, bin_size=0.15, color='b', marker_style='^', line_style='-',
                      plot_type='line', marker_size=8, marker_fill=True, line_width=2,
                      figsize=(9, 6), dpi=150, plot_label='', xlabel='', textbox='',
                      xlim=(None, None), ylim=(None, None), show_norm=False,
                      x_logscale=False, y_logscale=False, remove_zero_density=False,
                      file_name='plot_dist', file_label='', file_type=['svg','png'],
                      output_path='', save_fig=True, show_fig=False, mpl_ax=None,
                      return_plot_data=False, return_area_under_graph=False):

    # Return ax
    def plot_subroutine(x, y, ax, plot_type: str, plot_label: str, plot_norm=False):
        if plot_norm: c='r'; marker='None'; ls='--'
        else: c=color; marker=marker_style; ls=line_style
        if marker_fill == True:
            if plot_type == 'line': ax.plot(x, y, c=c, marker=marker, ms=marker_size, ls=ls,
                                            lw=line_width, label=plot_label)
            else: ax.scatter(x, y, c=c, marker=marker, s=marker_size, label=plot_label)
        else:
            if plot_type == 'line': ax.plot(x, y, c=c, marker=marker, ms=marker_size, ls=ls,
                                            lw=line_width, mfc='none', label=plot_label)
            else: ax.scatter(x, y, c=c, marker=marker, s=marker_size, facecolors='none', label=plot_label)
        return ax
    
    # Return ax, plot_data, area_under_graph
    def make_plot(ax):
        min_datum = np.amin(data); max_datum = np.amax(data)
        if show_norm == True:
            mu = np.mean(data); sigma = np.std(data)
            x_value_norm = np.linspace(mu - 4*sigma, mu + 4*sigma, 150)
            ax = plot_subroutine(x_value_norm, stats.norm.pdf(x_value_norm, mu, sigma), ax,
                                 'line', 'Normal distribution', plot_norm=True)
        if x_logscale == False:
            number_of_bins = math.ceil((max_datum - min_datum) / bin_size)
            hist_density, bin_edges = np.histogram(data, bins=np.linspace(min_datum, min_datum+number_of_bins*bin_size,
                                                                          number_of_bins), density=True)
        elif x_logscale == True:
            min_datum = np.amin(np.array(data)[np.array(data) > 0])
            number_of_bins = math.ceil((math.log10(max_datum) - math.log10(min_datum)) / bin_size)
            hist_density, bin_edges = np.histogram(data, bins=np.logspace(math.log10(min_datum),
                                                    math.log10(min_datum)+number_of_bins*bin_size, number_of_bins))
        hist_density = np.array(hist_density, dtype=float)
        hist_density /= np.dot(hist_density, np.diff(bin_edges)) # normalization
        x_value = (bin_edges[1:] + bin_edges[:-1]) / 2
        area_under_graph = np.dot(hist_density, np.diff(bin_edges))
        if remove_zero_density == True:
            non_zero_points = np.argwhere(hist_density == 0)
            hist_density = np.delete(hist_density, non_zero_points, 0)
            x_value = np.delete(x_value, non_zero_points, 0)
        return plot_subroutine(x_value, hist_density, ax,
                               plot_type, plot_label), np.array(list(zip(x_value, hist_density))), area_under_graph

    # Return ax
    def format_plot(ax):
        if x_logscale == True: ax.set_xscale('log')
        if y_logscale == True: ax.set_yscale('log')
        ax.set(xlabel=xlabel, ylabel='Probability density')
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.grid(True)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels))#, title='Distribution')
        # textbox
        if textbox != '':
            props = dict(boxstyle='round', pad=0.1, facecolor='white', edgecolor='none', alpha=0.75)
            ax.text(0.00001, 1.05, textbox, fontsize=10, verticalalignment='top', transform=ax.transAxes, bbox=props)
        return ax

    if mpl_ax == None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax, plot_data, area_under_graph = make_plot(ax)
        ax = format_plot(ax)
        for ext in file_type:
            if file_label == '': file_path = file_name+'.'+ext
            else: file_path = file_name+'_'+file_label+'.'+ext
            if save_fig: fig.savefig(os.path.join(output_path, file_path))
        if show_fig: plt.show()
        plt.clf()
        if return_plot_data and not return_area_under_graph: return plot_data
        elif not return_plot_data and return_area_under_graph: return area_under_graph
        else: return plot_data, area_under_graph
    else:
        ax, plot_data, _ = make_plot(mpl_ax)
        if return_plot_data: return ax, plot_data
        else: return ax

def plot_mean_of_average_synaptic_weights_vs_spike_count(network: object, dynamics: object,
                        in_or_out: str, inh_or_exc: str, bins=16, mpl_ax=None, color='b', marker='^',
                        marker_fill=True, marker_size=8, line_style='', line_width=2,
                        xlim=(None, None), ylim=(0, 0.03), plot_label='',
                        return_bin_info=False, return_plot_data=False, custom_bins=[]):
    """Plot the dependence of mean values of average synaptic weights on the number of spikes.
    
    Currently do not use without supplying `mpl_ax`.

    Parameters
    ----------
    network : object
        object of class Network
    dynamics : object
        object of class NeuralDynamics
    in_or_out : str
        the incoming / outgoing links to be plotted, options: 'in' or 'out'
    inh_or_exc : str
        the inhibitory (negative) / excitatory (positive) links to be plotted, options: 'inh' or 'exc'
    bins : int, optional
        number of bins, the resulting number of bins may be smaller due to duplicated bin values, by default 16
    mpl_ax : matplotlib.axes.Axes, optional
        if a matplotlib Axes is given, append the plot to the Axes, by default None
    color : str, optional
        color of lines and markers, by default 'b'
    marker : str, optional
        type of marker, by default '^'
    marker_fill : bool, optional
        marker face fill, solid markers if True, open markers if False, by default True
    marker_size : int, optional
        size of marker, by default 8
    line_style : str, optional
        style of line, by default ''
    line_width : int, optional
        line weight, by default 2
    xlim : tuple, optional
        range of x-axis, by default (None, None)
    ylim : tuple, optional
        range of y-axis, by default (0, 0.03)
    plot_label : str, optional
        label of the plot to be shown in legend, by default ''
    return_bin_info : bool, optional
        also return the details on bin size, number of data in each bin, etc. if enabled, by default False
    return_plot_data : bool, optional
        also return the data points of the plot if enabled, by default False
    custom_bins : list, optional
        customized bin edges, override `bins` if used, by default []

    Returns
    -------
    matplotlib.axes.Axes
        if `mpl_ax` is provided, return the Axes with the plot appended
    numpy.ndarray
        if `return_bin_info` is True, also return the details on bin size, number of data in each bin, etc.
    numpy.ndarray
        if `return_plot_data` is True, also return the data points of the plot
    """
    
    '''Settings'''
    print_info = False
    save_info = False

    spike_count = dynamics.spike_count
    if in_or_out == 'in': s_in_out = network.average_synaptic_weights_of_incoming_links(inh_or_exc)
    elif in_or_out == 'out': s_in_out = network.average_synaptic_weights_of_outgoing_links(inh_or_exc)
    s_in_out = np.array([x if x != None else 0 for x in s_in_out]) # If a neuron has no outgoing/incoming links, assign 0

    spike_count_idxsort = np.flip(spike_count.argsort())
    spike_count = spike_count[spike_count_idxsort[::-1]]
    s_in_out = s_in_out[spike_count_idxsort[::-1]]

    # Calculate bin edges
    bin_edges = [0, ] # You can set the first element set to 0.1 to exclude neurons with no spiking activity
    num_of_data_each_bin = int(len(spike_count) / bins)
    for b in range(bins-1):
        bin_edges.append(spike_count[num_of_data_each_bin + b * num_of_data_each_bin])
    bin_edges.append(spike_count[-1]+1) # +1 as the bin does not include its right edge
    if print_info:
        print('> Number of bins: {}'.format(bins)); print('> Bin edges: {}'.format(bin_edges))
    if save_info:
        fout = open('bin_info.txt')
        fout.write('> Number of bins: {}\n'.format(bins)); fout.write('> Bin edges: {} '.format(bin_edges))
    # Custom bin edges
    if len(custom_bins) != 0:
        bin_edges = custom_bins
        bins = len(bin_edges) - 1
    # Managing duplicates
    bin_edges, duplicate_count = np.unique(bin_edges, return_counts=True)
    duplicate_count = np.sum(duplicate_count-1)
    bins -= duplicate_count
    if print_info:
        print('>> Removed {} duplicated bin.'.format(duplicate_count))
        print('>> Number of bins: {}'.format(bins))
        print('>> Bin edges: {}'.format(list(bin_edges)))
    if save_info:
        fout.write('Removed {} duplicated bin.\n'.format(duplicate_count))
        fout.write('Number of bins: {}\n'.format(bins))
        fout.write('Bin edges: {}\n'.format(str(bin_edges)))

    # Calculate number of data, average spike count for each bin
    bin_number = np.zeros(len(spike_count))
    num_of_data_in_bin = np.zeros(bins)
    avg_spike_count = np.zeros(bins)
    avg_s_in_out = np.zeros(bins)
    for i in range(len(spike_count)):
        for b in range(bins):
            if bin_edges[b] <= spike_count[i] < bin_edges[b+1]:
                bin_number[i] = b
                num_of_data_in_bin[b] += 1
                avg_spike_count[b] += spike_count[i]
                avg_s_in_out[b] += s_in_out[i]
                break
    for b in range(len(avg_spike_count)):
        avg_spike_count[b] /= num_of_data_in_bin[b]
        avg_s_in_out[b] /= num_of_data_in_bin[b]
    if print_info:
        print('>> Number of data in each in: {}'.format(num_of_data_in_bin))
        print('>> S.D. of No. of data: {:f}'.format(np.std(num_of_data_in_bin)))
    if save_info:
        fout.write('Number of data in each in: {}\n'.format(str(num_of_data_in_bin)))
        fout.write('S.D. of No. of data: {:f}'.format(np.std(num_of_data_in_bin)))
    bin_details = (bins, bin_edges, num_of_data_in_bin)

    del spike_count_idxsort
    del spike_count; del s_in_out
    del bin_edges; del bin_number
    del num_of_data_in_bin

    if not mpl_ax == None:
        ax = mpl_ax
        if marker_fill:
            ax.plot(avg_spike_count, avg_s_in_out, marker=marker, c=color, ms=marker_size,
                    ls=line_style, lw=line_width, label=plot_label)
        else:
            ax.plot(avg_spike_count, avg_s_in_out, mfc='none', marker=marker, c=color,
                    ms=marker_size, ls=line_style, lw=line_width, label=plot_label)
        if return_bin_info == True and return_plot_data == True:
            return ax, bin_details, np.array(list(zip(avg_spike_count, avg_s_in_out)))
        elif return_bin_info == True and return_plot_data == False:
            return ax, bin_details
        elif return_bin_info == False and return_plot_data == True:
            return ax, np.array(list(zip(avg_spike_count, avg_s_in_out)))
        else: return ax
    del avg_spike_count, avg_s_in_out

def fread_array(filename: str, dtype=float)->np.ndarray:
    try:
        return np.array(list(csv.reader(open(filename, 'r')))).astype(dtype).flatten()
    except FileNotFoundError:
        err = 'FileNotFoundError: matrix file "{}" cannot be found.'.format(filename)
        print(err); exit(1)

def fread_dense_matrix(filename: str, delim='\t')->np.ndarray:
    """Read a matrix from a file.
    
    The file should store every elements with each row elements separated by `delim`.

    Parameters
    ----------
    filename : str
        name or path of the file
    delim : str, optional
        delimiter of the input matrix file, by default `'\\t'`

    Returns
    -------
    np.ndarray
        the matrix in the file
    """
    try:
        return np.array(list(csv.reader(open(filename, 'r'), delimiter=delim))).astype(float)
    except FileNotFoundError:
        err = 'FileNotFoundError: matrix file "{}" cannot be found.'.format(filename)
        print(err); exit(1)

def fread_sparse_matrix(filename: str, A_size: tuple, delim=' ', start_idx=1)->np.ndarray:
    """Read a matrix from a file.
    
    The file should store only nonzero elements in each row, with format: j i w_ji, separated by `delim`.

    Parameters
    ----------
    filename : str
        name or path of the file
    A_size : tuple of int
        size of the matrix (row size, col size)
    delim : str or chr, optional
        delimiter of the matrix file, by default 'whitespace'
    start_idx : int, optional
        the index i and j start from, usually it's 0 or 1, by default 1

    Returns
    -------
    numpy.ndarray
        the matrix in the file
    """
    if type(A_size) == int: A_size = (A_size, A_size)
    try:
        with open(filename, 'r', newline='') as fp:
            content = list(csv.reader(fp, delimiter=delim))
            for i in range(len(content)):
                content[i] = remove_all_occurrences('', content[i])
                content[i][0] = int(content[i][0])-start_idx # j
                content[i][1] = int(content[i][1])-start_idx # i
                content[i][2] = float(content[i][2]) # w_ij
            matrix = np.zeros((A_size[0], A_size[1]))
            for item in content: matrix[item[1]][item[0]] = item[2]
            return np.array(matrix).astype(float)
    except FileNotFoundError:
        err = 'FileNotFoundError: matrix file "{}" cannot be found.'.format(filename)
        print(err); exit(1)

def fread_ragged_2darray(filename: str, delim='\t', dtype=float) -> np.ndarray:
    """Read a 2d array with varying row length from a file.

    Parameters
    ----------
    filename : str
        name or path of the file
    delim : str or chr, optional
        delimiter of the file, by default `'\\t'`
    dtype : datatype, optional
        datatype of elements, by default `float`

    Returns
    -------
    np.ndarray
        the 2d ragged array in the file
    """
    try:
        return np.array([np.array(
            list(filter(None, x))).astype(dtype) for x in
                list(csv.reader(open(filename, 'r'), delimiter=delim)
        )], dtype=object)
    except FileNotFoundError:
        err = 'FileNotFoundError: matrix file "{}" cannot be found.'.format(filename)
        print(err); exit(1)

def fwrite_dense_matrix(A: iter, filename: str, delim='\t', dtype=float):
    """Write a matrix into a file.
    
    Write the full matrix, that is, every element including 0. File size will be large.

    Parameters
    ----------
    A : iter
        matrix to be written
    filename : str
        name or path of the file
    delim : str, optional
        delimiter of the output matrix file, by default `'\\t'`
    dtype : datatype, optional
        datatype of elements, by default `float`
    """
    try:
        with open(filename, 'w') as fp:
            if dtype == int:
                for row in A:
                    if row[0] == 0: fp.write('{:d}'.format(row[0])) # To reduce file size
                    else:           fp.write( '{:d}'.format(row[0]))
                    for element in row[1:]:
                        if element == 0:    fp.write('{}{:d}'.format(delim, element)) # To reduce file size
                        else:               fp.write( '{}{:d}'.format(delim, element))
                    fp.write('\n')      
            else:
                for row in A:
                    if row[0] == 0: fp.write('{:.0f}'.format(row[0])) # To reduce file size
                    else:           fp.write( '{:.8}'.format(row[0]))
                    for element in row[1:]:
                        if element == 0:    fp.write('{}{:.0f}'.format(delim, element)) # To reduce file size
                        else:               fp.write( '{}{:.8}'.format(delim, element))
                    fp.write('\n')
    except FileNotFoundError:
        err = 'FileNotFoundError: cannot write to file "{}", e.g., the path to the directory does not exist.'.format(filename)
        print(err); exit(1)

def fwrite_sparse_matrix(A: iter, filename: str, delim=' ', start_idx=1, dtype=float):
    """Write a matrix into a file.
    
    Write only nonzero elements into each row with format: index j, index i, value (separated by `delim`).
    Efficient for sparse matrix with large amount of zero elements.

    Parameters
    ----------
    A : iter
        matrix to be written
    filename : str
        name or path of the file
    delim : str, optional
        delimiter of the output matrix file, by default 'whitespace'
    start_idx : int, optional
        the index i and j start from, usually it's 0 or 1, by default 1
    dtype : datatype, optional
        datatype of elements, by default `float`
    """
    try:
        with open(filename, 'w') as fp:
            if dtype == int:
                for i in range(len(A)):
                    for j in range(len(A[i])):
                        if A[i][j] != 0:
                            fp.write('{:d}{}{:d}{}{:d}\n'.format(j+start_idx, delim, i+start_idx, delim, A[i][j]))
            else:
                for i in range(len(A)):
                    for j in range(len(A[i])):
                        if A[i][j] != 0:
                            fp.write('{:d}{}{:d}{}{:f}\n'.format(j+start_idx, delim, i+start_idx, delim, A[i][j]))
    except FileNotFoundError:
        err = 'FileNotFoundError: cannot write to file "{}", e.g., the path to the directory does not exist.'.format(filename)
        print(err); exit(1)

def remove_diag(A: np.ndarray)->np.ndarray:
    """Remove diagonal elements from a numpy.ndarray square matrix."""
    if A.shape[0] == A.shape[1]:
        return A[~np.eye(A.shape[0],dtype=bool)].reshape(A.shape[0],-1)
    else:
        err = 'ArgumentValueError: the input matrix must be square.'
        print(err); exit(1)

def remove_rowcol(A: np.ndarray, rm_idx)->np.ndarray:
    return np.delete(np.delete(A, rm_idx, axis=0), rm_idx, axis=1)

def is_strictly_increasing(seq: iter)->bool:
    """Return True if the input sequence is strictly increasing, else, return False."""
    return all(x > y for x, y in zip(seq, seq[1:]))

def is_strictly_decreasing(seq: iter)->bool:
    """Return True if the input sequence is strictly decreasing, else, return False."""
    return all(x > y for x, y in zip(seq, seq[1:]))

def is_non_increasing(seq: iter)->bool:
    """Return True if the input sequence is non increasing, else, return False."""
    return all(x >= y for x, y in zip(seq, seq[1:]))

def is_non_decreasing(seq: iter)->bool:
    """Return True if the input sequence is non decreasing, else, return False."""
    return all(x <= y for x, y in zip(seq, seq[1:]))

def is_monotonic(seq: iter)->bool:
    """Return True if the input sequence is monotonically increasing or decreasing, else, return False."""
    return is_non_decreasing(seq) or is_non_increasing(seq)

def remove_all_occurrences(x, a: list)->list:
    """Remove all occurences of an element from a list or string."""
    return list(filter((x).__ne__, a))


# Help
def ask_for_help():
    """Evoke this function to show help and advanced instructions.
    """
    print('\n  HOW TO FORMAT PLOTS AND GRAPHS')
    print('  ==============================')
    print(help_graph_formatting.__doc__)

def help_graph_formatting():
    """
    Below shows additional parameters for graphing functions.

    Note that not all functions take every parameter.

    Parameters
    ----------
    color : str, optional
        color of lines and markers, by default 'b'
    marker : str, optional
        type of marker, usually by default '^'
    marker_fill : bool, optional
        marker face fill, solid markers if True, open markers if False, usually by default True
    marker_size : int, optional
        size of marker, usually by default 8
    line_style : str, optional
        style of line, by default ''
    line_width : int, optional
        line weight, by usually default 2
    xlim : tuple, optional
        range of x-axis
    ylim : tuple, optional
        range of y-axis
    plot_label : str, optional
        label of the plot to be shown in legend, usually by default ''
    return_bin_info : bool, optional
        also return the details on bin size, number of data in each bin, etc. if enabled, usually by default False
    return_plot_data : bool, optional
        also return the data points of the plot if enabled, usually by default False
    mpl_ax : matplotlib.axes.Axes, optional
        if a matplotlib Axes is given, append the plot to the Axes, usually by default None
    custom_bins : list, optional
        customized bin edges, override `bins` if used, by default []
    """
    print('\n  HOW TO FORMAT PLOTS AND GRAPHS')
    print('  ==============================')
    print(help_graph_formatting.__doc__)


# Miscellaneous
def _colorline(x, y, z=None, cmap='hsv', norm=plt.Normalize(0.0, 1.0), linewidth=1, alpha=1):
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    if not hasattr(z, "__iter__"):
        z = np.array([z])
    z = np.asarray(z)
    segments = _make_segments(x, y)
    lc = mcol.LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    ax = plt.gca()
    ax.add_collection(lc)
    return lc

def _make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments
