"""
Neuron.py
=========

A handy tool simulate the dynamics of a single spiking neuron.

"""


import lib.spikingneuron as sn
import lib.neurolib as nlib
import numpy as np


"""Settings"""
model_parameters = dict(
                       delta_time = 0.05,
                    current_drive = 0,
                                    # threshold for stimulating spikes in absence of noise:
                                    # 3.7741432 (for regular spiking neuron)
                                    # 3.858551  (for fast spiking neuron)
     whitenoise_standarddeviation = 4,
    random_number_generation_seed = 0,
       initial_membrane_potential = -65.0,
        initial_recovery_variable = 8
)
simulation_duration = 50000
neurontype = sn.NeuronType.regular_spiking
# The neuron types you can choose from:
# Excitatory cells:
# - regular_spiking
# - intrinsically_bursting
# - chattering
# Inhibitory cells:
# - fast_spiking
# - low_threshold_spiking





class Simulation:

    def __init__(self, simulation_duration: float, model_parameters: dict, neurontype: dict) -> None:
        self.model_parameters = model_parameters
        self.simulation_duration = simulation_duration
        self.neurontype = neurontype
        self.spike_times, self.potential, self.recovery, self.current, self.noise = [], [], [], [], []

    def start(self) -> None:
        snm = sn.SpikingNeuronModel(self.neurontype, self.model_parameters)
        steps = int(self.simulation_duration/self.model_parameters['delta_time'])
        # Simulation loop
        for i in range(steps):
            snm.step()
            self.potential.append(snm.membrane_potential())
            self.current.append(snm.current_driving())
            self.recovery.append(snm.recovery_variable())
        self.number_of_spikes = snm.number_of_spikes();
        for i in range(self.number_of_spikes): self.spike_times.append(snm.spike_timestep(i))
        self.interspike_intervals = np.array(np.diff(self.spike_times), dtype=float)*model_parameters['delta_time']/1000

def plot_time_series(sim: object, saveplot=True, showplot=False):
    # Time series
    xtime = np.arange(0, int(simulation_duration/model_parameters['delta_time']))
    g = nlib.GraphDataRelation()
    g.create_plot(figsize=(9,7))
    g.stylize_plot(dict(ms=0, ls='-'))
    g.stylize_plot(dict(c='b'))
    g.label_plot(dict(plotlabel=r'Membrane potential'))
    g.add_data(xtime, sim.potential)
    g.make_plot()
    g.stylize_plot(dict(c='darkorange'))
    g.label_plot(dict(plotlabel=r'Recovery variable'))
    g.add_data(xtime, sim.recovery)
    g.make_plot()
    g.stylize_plot(dict(c='m'))
    g.label_plot(dict(plotlabel=r'Synaptic current'))
    g.add_data(xtime, sim.current)
    g.make_plot()
    g.set_scale(dict(xlim=(0, simulation_duration)))
    # g.ax.legend(prop={'size':15})
    g.label_plot(dict(axislabel=['Time (ms)', ''], title='Time series', legend=True,
                      textbox='Neuron type: {} | step size: {} ms'.format(neurontype['name'],
                               model_parameters['delta_time'])))
    if saveplot: g.save_plot('Time series - single neuron')
    if showplot: g.show_plot()

def plot_firing_rate_distribution(sim: object, binsize=0.1, saveplot=True, showplot=False):
    # Firing rate distribution
    g = nlib.GraphDensityDistribution()
    g.create_plot(figsize=(9,7))
    g.add_data(sim.firing_rate, binsize)
    g.stylize_plot(dict(c='b'))
    g.label_plot(dict(plotlabel=r''))
    g.make_plot()
    g.label_plot(dict(xlabel='Firing rate (Hz)', textbox='step size: {} ms'.format(model_parameters['delta_time'])))
    g.apply_format()
    if saveplot: g.save_plot('Firing rate distribution - single neuron')
    if showplot: g.show_plot()

def plot_isi_distribution(sim: object, binsize: 0.1, saveplot=True, showplot=False):
    # ISI distribution
    g = nlib.GraphDensityDistribution()
    g.create_plot(figsize=(9,7))
    g.add_data(sim.interspike_intervals, binsize, logdata=True)
    g.stylize_plot(dict(c='b'))
    g.label_plot(dict(plotlabel=r''))
    g.make_plot()
    g.label_plot(dict(xlabel='Inter-spike interval ISI (s)', textbox='step size: {} ms'.format(model_parameters['delta_time'])))
    g.set_scale(dict(xlogscale=True))
    g.apply_format()
    if saveplot: g.save_plot('log(ISI) distribution - single neuron')
    if showplot: g.show_plot()

def main():
    sim = Simulation(simulation_duration, model_parameters, neurontype)
    sim.start()
    plot_time_series(sim, False, True)
    # plot_firing_rate_distribution(sim)
    # plot_isi_distribution(sim)
    pass

if __name__ == "__main__": main()