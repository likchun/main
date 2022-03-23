import sys
from lib.neuronlib import Network, NeuralDynamics, plot_mean_of_average_synaptic_weights_vs_spike_count
from matplotlib import pyplot as plt
import numpy as np


'''Control Panel'''
def menu():
    # Input and plot settings
    b45 = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 12.0, 20.0, 30.0, 48.0, 336.0] # custom bin_s_vs_spkno for DIV45
    dataA = [
        # | 0: name            |
        # | 1: folder          |
        # | 2: color           |
        # | 3: marker          |
        # | 4: bin_fr          |
        # | 5: bin_fr_logden   |
        # | 6: bin_logisi      |
        # | 7: bin_fr_chg_inh  |
        # | 8: bin_fr_chg_exc  |
        # | 9: bin_s_vs_spkno  | < int: no of bins or list: bin edges
        # ('DIV11', 'DIV11', 'k',  'o', 0.185, 0.21, 0.050, 0.100, 0.20, 16 ),
        # ('DIV22', 'DIV22', 'r',  's', 0.200, 0.35, 0.050, 0.097, 0.20, 16 ),
        # ('DIV25', 'DIV25', 'g',  'D', 0.185, 0.35, 0.050, 0.200, 0.45, 16 ),
        # ('DIV33', 'DIV33', 'b',  '^', 0.185, 0.21, 0.050, 0.200, 0.50, 20 ),
        # ('DIV45', 'DIV45', 'C1', '<', 0.185, 0.21, 0.050, 0.100, 0.50, b45),
        # ('DIV52', 'DIV52', 'C7', 'v', 0.185, 0.21, 0.050, 0.100, 0.30, 19 ),
        # ('DIV59', 'DIV59', 'm',  '>', 0.185, 0.21, 0.075, 0.095, 0.40, 16 ),
        # ('DIV66', 'DIV66', 'c',  'X', 0.185, 0.21, 0.050, 0.100, 0.30, 20 ),
        ('DIV66', 'sample', 'b',  '^', 0.185, 0.21, 0.050, 0.100, 0.30, 20 )
    ]
    # Plot selection
    preview = False
    quickplot_each_network(dataA)
    # quickplot_firing_rate(dataA, preview)
    # quickplot_logisi(dataA, preview)
    # quickplot_firing_rate_change_inh(dataA, preview)
    # quickplot_firing_rate_change_exc(dataA, preview)
    # quickplot_s_vs_spike_count(dataA)
    # neuronal_response_on_random_seeds()
    # inconsistent_heterogenous_response_on_firing_rate_change(preview)


def quickplot_each_network(data: iter):
    '''
    Plot firing rate distribution, ISI distribution, raster of spking activity 
    and membrane potential time series of node 25
    '''
    # (0: name, 1: folder, 2: color, 3: marker, 4: bins_fr, 5: bin_fr_logy, 6: bin_logisi)
    print('\n> start <\n')
    for d in data:
        ndyn = NeuralDynamics(d[1], 'plot_output')
        ndyn.plot_spike_raster(1.5, file_label=d[0])
        ndyn.plot_firing_rate_distribution(d[4], file_label=d[0])
        ndyn.plot_interspike_interval_distribution(d[6], file_label=d[0])
        try: ndyn.plot_membrane_potential_time_series(25, figsize=(12, 6), file_label=d[0])
        except FileNotFoundError: pass
        del ndyn
    print('\n> completed <\n')

def quickplot_firing_rate(data: iter, preview=False):
    print('\n> start <\n')
    fig1, ax1 = plt.subplots(figsize=(12, 7), dpi=150)
    fig2, ax2 = plt.subplots(figsize=(12, 7), dpi=150) # log density
    for d in data:
        ndyn = NeuralDynamics(d[1], 'plot_output')
        ax1 = ndyn.plot_firing_rate_distribution(d[4], ax1, d[2], d[3], ms=5, lw=1.5, plot_label=d[0])
        ax2 = ndyn.plot_firing_rate_distribution(d[5], ax2, d[2], d[3], ms=5, lw=1.5, plot_label=d[0], remove_zero_density=True)
        del ndyn
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(reversed(handles), reversed(labels), title='Networks') # reverse
    ax1.legend(title='Networks'); ax2.legend(title='Networks')
    ax1.set(xlabel='Firing rate (Hz)', ylabel='Probability density')
    ax2.set(xlabel='Firing rate (Hz)', ylabel='Probability density')
    ax1.grid(True); ax2.grid(True)
    ax1.set_xlim(0, 15)
    ax2.set_xlim(0, 25)
    ax2.set_yscale('log')
    if preview: fig1.show(); fig2.show()
    else:
        fig1.savefig('plot_firing_rate.svg'); fig1.savefig('plot_firing_rate.png')
        fig2.savefig('plot_firing_rate_logdensity.svg'); fig2.savefig('plot_firing_rate_logdensity.png')
    print('\n> completed <\n')

def quickplot_logisi(data: iter, preview=False):
    print('\n> start <\n')
    fig1, ax1 = plt.subplots(figsize=(9, 9), dpi=150)
    fig2, ax2 = plt.subplots(figsize=(9, 9), dpi=150) # log density
    for d in data:
        ndyn = NeuralDynamics(d[1], d[1])
        ax1 = ndyn.plot_interspike_interval_distribution(d[6], ax1, d[2], d[3], ms=5, lw=1.5, plot_label=d[0])
        ax2 = ndyn.plot_interspike_interval_distribution(d[6], ax1, d[2], d[3], ms=5, lw=1.5, plot_label=d[0])
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(reversed(handles), reversed(labels), title='Networks') # reverse
    ax1.legend(title='Networks'); ax2.legend(title='Networks')
    ax1.set(xlabel='Inter-spike interval (s)', ylabel='Probability density')
    ax2.set(xlabel='Inter-spike interval (s)', ylabel='Probability density')
    ax1.grid(True); ax2.grid(True)
    ax2.set_xscale('log')
    if preview: fig1.show(); fig2.show()
    else:
        fig1.savefig('plot_logisi.svg'); fig1.savefig('plot_logisi.png')
        fig2.savefig('plot_logisi_logdensity.svg'); fig2.savefig('plot_logisi_logdensity.png')
    print('\n> completed <\n')

def quickplot_firing_rate_change_inh(data: iter, preview=False):
    print('\n> start <\n')
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    for d in data:
        orig = NeuralDynamics(d[1])
        ndyn = NeuralDynamics(d[0]+'_'+'INH_SUP')
        ndyn.plot_firing_rate_change_distribution(orig, d[7], ms=5, file_label=d[0],
                                                  remove_zero_density=True)
        ax = ndyn.plot_firing_rate_change_distribution(orig, d[7], ax, d[2], d[3],
                                                       plot_label=d[0],
                                                       remove_zero_density=True)
        del orig, ndyn
    ax.set(xlabel='Change in firing rate (Hz)', ylabel='Probability density')
    ax.set_yscale('log')
    ax.grid(True)
    ax.legend(title='Networks')
    if preview: fig.show()
    else:
        fig.savefig('plot_firing_rate_change_inh.svg')
        fig.savefig('plot_firing_rate_change_inh.png')
    print('\n> completed <\n')

def quickplot_firing_rate_change_exc(data: iter, preview=False):
    print('\n> start <\n')
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    for d in data:
        orig = NeuralDynamics(d[1])
        ndyn = NeuralDynamics(d[0]+'_'+'EXC_SUP')
        ndyn.plot_firing_rate_change_distribution(orig, d[8], ms=5, file_label=d[0],
                                                  remove_zero_density=True)
        ax = ndyn.plot_firing_rate_change_distribution(orig, d[8], ax, d[2], d[3],
                                                       plot_label=d[0],
                                                       remove_zero_density=True)
        del orig, ndyn
    ax.set(xlabel='Change in firing rate (Hz)', ylabel='Probability density')
    ax.set_yscale('log')
    ax.grid(True)
    ax.legend(title='Networks')
    if preview: fig.show()
    else:
        fig.savefig('plot_firing_rate_change_exc.svg')
        fig.savefig('plot_firing_rate_change_exc.png')
    print('\n> completed <\n')

def quickplot_s_vs_spike_count(data: iter, preview=False):
    '''
    Dependence of average incoming/outgoing inhibitory/excitatory
    synaptic weights on numbers of spikes
    '''
    print('\n> start <\n')
    for in_or_out in ['in', 'out']:
        if in_or_out == 'out': ylim = (-0.0025, 0.03)
        elif in_or_out == 'in': ylim = (-0.001, 0.02)

        fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
        for d in data:
            if type(d[9]) == int: bins = d[9]; custom_bin_edges = []
            elif type(d[9]) == list: custom_bin_edges = d[9]; bins = 10
            else: print('[ArgumentTypeError]'); exit(1)
            net = Network(d[0]+'_gji', d[1])
            dyn = NeuralDynamics(d[1])
            ax, bd, pd_inh = plot_mean_of_average_synaptic_weights_vs_spike_count(net,
                                        dyn, in_or_out, 'inh', bins, d[2], d[3], mpl_ax=ax,
                                        marker_fill=False, return_bin_info=True,
                                        return_plot_data=True, custom_bins=custom_bin_edges)
            ax, pd_exc = plot_mean_of_average_synaptic_weights_vs_spike_count(net,
                                        dyn, in_or_out, 'exc', bins, d[2], d[3], mpl_ax=ax,
                                        plot_label=d[0], return_plot_data=True,
                                        custom_bins=custom_bin_edges)
            with open('bins_info_'+d[0]+'.txt', 'w') as fp:
                fp.write('Number of bins: {}\n'.format(bd[0]))
                fp.write('Bin edges:\n  {}\n'.format(str(list(bd[1].astype(int)))))
                fp.write('Number of data in each bin:\n')
                fp.write('[1]average number of spikes\n[2]number of data\n[3]bin interval#\n')
                fp.write('  [1]\t\t [2]\t\t[3]#\n')
                for i in range(bd[0]):
                    fp.write('{:5.1f}\t\t {:3d}\t\t[{:3.0f}, {:3.0f})\n'.format(pd_inh[i][0], int(bd[2][i]), bd[1][i], bd[1][i+1]))
                fp.write('---------------- -----------------\n')
                fp.write('total\t\t{:d}\n\n'.format(int(np.sum(bd[2]))))
                fp.write('# left boundary is included, right boundary is excluded')
            with open('plot_data_s_'+in_or_out+'_inh_'+d[0]+'.txt', 'w') as fp:
                for x in pd_inh: fp.write('{}\t{}\n'.format(x[0], x[1]))
            with open('plot_data_s_'+in_or_out+'_exc_'+d[0]+'.txt', 'w') as fp:
                for x in pd_exc: fp.write('{}\t{}\n'.format(x[0], x[1]))
            del net; del dyn
        ax.set_xscale('log')
        ax.set_ylim(ylim[0], ylim[1])
        ax.set(xlabel='Number of spikes', ylabel='s_'+in_or_out+'+ and s_'+in_or_out+'-')
        ax.grid(True)
        ax.legend(title='Networks')
        if preview: fig.show()
        else:
            fig.savefig('plot_s_'+in_or_out+'_vs_spike_count.svg')
            fig.savefig('plot_s_'+in_or_out+'_vs_spike_count.png')
    print('\n> completed <\n')

def neuronal_response_on_random_seeds():
    '''
    Dependence of heterogeneous neuronal responses on random number seeds
    '''

    '''Settings'''
    changes = 'INH_SUP'
    enableFilter = False
    node_index_filter = np.arange(10,20)
    plot_limit = 10

    print('\n> start <\n')
    np.set_printoptions(threshold=sys.maxsize)
    for network in ['DIV11', 'DIV22', 'DIV25', 'DIV33', 'DIV45', 'DIV52', 'DIV59', 'DIV66']:
        data = [
            (network,         network+'_'+changes,         0),
            (network+'_s100', network+'_'+changes+'_s100', 100),
            (network+'_s789', network+'_'+changes+'_s789', 789),
        ]

        fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
        ax.plot([-1, 3], [0, 0], 'k:') # line y=0

        fr_chg = []
        for d in range(len(data)):
            dyn_org = NeuralDynamics(data[d][0])
            dyn_inh = NeuralDynamics(data[d][1])
            if enableFilter == False:
                fr_chg.append(dyn_inh.firing_rate_change(dyn_org))
                node_index_filter = np.arange(0, len(fr_chg[0]))
            else:
                fr_chg.append(dyn_inh.firing_rate_change(dyn_org)[node_index_filter])
            del dyn_org, dyn_inh

        haveSameSign = np.zeros(len(fr_chg[0]))
        plot_count = 0
        for i in range(len(fr_chg[0])):
            x, y = [], []
            for d in range(len(data)): x.append(d); y.append(fr_chg[d][i])
            x, y = np.array(x), np.array(y)
            hasIncreases, hasDecreases = np.any(y > 0), np.any(y < 0)
            if hasIncreases and hasDecreases: haveSameSign[i] = False
            else: haveSameSign[i] = True
            if haveSameSign[i] == False and plot_count < plot_limit:
                ax.plot(x, y, 'o-', label=str(node_index_filter[i]+1))
                plot_count += 1

        with open('info_'+network+'_'+changes+'.txt', 'w') as fp:
            fp.write('Number of neuron nodes that have *inconsistent heterogeneous response under different random number seeds*: {:d}\n'.format(len(haveSameSign[haveSameSign == False])))
            fp.write('(*that is, the response of a neuron which increases its firing rate for one random number seed but decreases for another, or vice versa, upon inhibition/excitation suppression)\n\n')
            fp.write('The corresponding neuron node indices:\n{}'.format(str(np.argwhere(haveSameSign == False).flatten()+1)))

        # yabs_max = abs(max(ax.get_ylim(), key=abs))
        # ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)
        ax.set_xlim(-0.2, 2.2)
        ax.set_xticks(x)
        ax.set_xticklabels(['0', '100', '789'])
        ax.set(xlabel='Seed for random number', ylabel='Change in firing rate (Hz)')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Node index')
        ax.grid(True)
        fig.savefig('fr_chg_vs_rand_seed_'+network+'_'+changes+'.svg')
        fig.savefig('fr_chg_vs_rand_seed_'+network+'_'+changes+'.png')
    print('\n> completed <\n')

def inconsistent_heterogenous_response_on_firing_rate_change(preview=False):
    '''
    Dependence of numbers of nodes with inconsistent heterogenoues response
    under different random number seeds VS average changes in firing rate
    '''

    '''Settings'''
    seed = ''

    print('\n> start <\n')
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    networks = ['DIV11', 'DIV22', 'DIV25', 'DIV33', 'DIV45', 'DIV52', 'DIV59', 'DIV66']
    colors = ['k', 'r', 'g', 'b', 'C1', 'C7', 'm', 'c'] 
    ydata = [(1095, 207), (147, 27), (232, 47), (314, 198), (1109, 362), (842, 238), (960, 305), (950, 333)]
    for i in range(len(networks)):
        avg_fr_chg = [] # DIVxx_INH_SUP, DIVxx_EXC_SUP
        dyn_0 = NeuralDynamics(networks[i]+seed)
        dyn_i = NeuralDynamics(networks[i]+'_INH_SUP'+seed)
        dyn_e = NeuralDynamics(networks[i]+'_EXC_SUP'+seed)
        avg_fr_chg.append(np.mean(dyn_i.firing_rate_change(dyn_0)))
        avg_fr_chg.append(np.mean(dyn_e.firing_rate_change(dyn_0)))
        ax.plot(avg_fr_chg[0], ydata[i][0], colors[i]+'o', mfc='none')
        ax.plot(avg_fr_chg[1], ydata[i][1], colors[i]+'o', label=networks[i])
    ax.set(xlabel='Average changes in firing rate (Hz)', ylabel='Number of nodes')
    ax.legend()
    if preview: fig.show()
    else: fig.savefig('plot.svg')
    print('\n> completed <\n')


if __name__=='__main__':
    menu()