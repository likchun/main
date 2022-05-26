from lib.spikingneuron import *
from lib.neuronlib import *
import numpy as np


model_parameters = dict(
                       delta_time = 0.05,
                    current_drive = 0,
     whitenoise_standarddeviation = 0,
    random_number_generation_seed = 0,
       initial_membrane_potential = -65.0,
        initial_recovery_variable = 8
)


snm = SpikingNeuronModel(NeuronType.regular_spiking, model_parameters)

steps = 10000

potential, current, noise = [], [], []
for i in range(steps):
    # if i == 2000: snm.set_current_drive(-5)
    # if i == 3000: snm.set_current_drive(12)
    if i == 5000: snm.set_current_driving(4)
    # if i == 5100: snm.set_current_drive(0)
    # if i == 5150: snm.set_current_drive(24)
    if i == 5250: snm.set_current_driving(0)
    snm.step()
    potential.append(snm.membrane_potential())
    current.append(snm.current())
    noise.append(snm.current_stochastic())

g = Grapher()
g.create_plot(figsize=(9,7))
g.stylize_plot(dict(c='0.4', ls='--', ms=0))
g.add_data(np.arange(0, steps)*model_parameters['delta_time'], noise)
g.make_plot()
g.stylize_plot(dict(c='b', ms=0, ls='-'))
g.label_plot(dict(axislabel=['Time (ms)', 'Membrane potential (mV)'], textbox='Step size = {} ms | Total steps = {}'.format(model_parameters['delta_time'], steps)))
g.add_data(np.arange(0, steps)*model_parameters['delta_time'], potential)
g.make_plot()
g.stylize_plot(dict(c='darkorange'))
g.add_data(np.arange(0, steps)*model_parameters['delta_time'], current)
g.make_plot()
g.set_scale(dict(xlim=(0, steps*model_parameters['delta_time'])))
g.show_plot()