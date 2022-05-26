from ctypes import *


class NeuronType():
    # excitatory cells
    regular_spiking = dict(
        a = 0.02,
        b = 0.2,
        c = -65.0,
        d = 8.0
    )
    intrinsically_bursting = dict(
        a = 0.02,
        b = 0.2,
        c = -55.0,
        d = 4.0
    )
    chattering = dict(
        a = 0.02,
        b = 0.2,
        c = -50.0,
        d = 2.0
    )

    # inhibitory cells
    fast_spiking = dict(
        a = 0.1,
        b = 0.2,
        c = -65.0,
        d = 2.0
    )
    low_threshold_spiking = dict(
        a = 0.02,
        b = 0.25,
        c = -65.0,
        d = 2.0
    )

class SpikingNeuronModel():
    lib = cdll.LoadLibrary('./lib/spikingneuron.so')
    lib.parameters.restype = None
    lib.parameters.argtypes = [c_double, c_double, c_double, c_double,
                               c_double, c_double, c_double, c_double,
                               c_double, c_double]
    lib.step.restype = c_int
    lib.step.argtypes = []
    lib.get_membrane_potential.restype = c_double
    lib.get_membrane_potential.argtypes = []
    lib.get_membrane_potential_stochastic.restype = c_double
    lib.get_membrane_potential_stochastic.argtypes = []
    lib.get_recovery_variable.restype = c_double
    lib.get_recovery_variable.argtypes = []
    lib.get_current.restype = c_double
    lib.get_current.argtypes = []
    lib.get_current_driving.restype = c_double
    lib.get_current_driving.argtypes = []
    lib.set_current_driving.restype = None
    lib.set_current_driving.argtypes = [c_double]
    lib.get_current_stochastic.restype = c_double
    lib.get_current_stochastic.argtypes = []

    def __init__(self, neuron_parameters, model_parameters):
        try:
            SpikingNeuronModel.lib.parameters(
                model_parameters['delta_time'],
                neuron_parameters['a'],
                neuron_parameters['b'],
                neuron_parameters['c'],
                neuron_parameters['d'],
                model_parameters['current_drive'],
                model_parameters['whitenoise_standarddeviation'],
                model_parameters['random_number_generation_seed'],
                model_parameters['initial_membrane_potential'],
                model_parameters['initial_recovery_variable']
            )
        except KeyError as e:
            print('KeyError: the parameters {} is missing\n'.format(e))
            exit(1)

    def step(self, number_of_step=1):
        SpikingNeuronModel.lib.step(number_of_step)

    def membrane_potential(self):
        return SpikingNeuronModel.lib.get_membrane_potential()

    def membrane_potential_stochastic(self):
        return SpikingNeuronModel.lib.get_membrane_potential_stochastic()

    def recovery_variable(self):
        return SpikingNeuronModel.lib.get_recovery_variable()

    def current(self):
        return SpikingNeuronModel.lib.get_current()

    def current_driving(self):
        return SpikingNeuronModel.lib.get_current_driving()

    def set_current_driving(self, current):
        SpikingNeuronModel.lib.set_current_driving(current)

    def current_stochastic(self):
        return SpikingNeuronModel.lib.get_current_stochastic()
