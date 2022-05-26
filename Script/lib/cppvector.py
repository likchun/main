"""
CppVector
=========

Provide access to C++ vectors for Python.
"""

from ctypes import *
import numpy as np

class Vector_fl(object):
    """C++ vector wrapped in Python ctypes.

    Returns
    -------
    pointer
        pointer to the C++ vector

    Raises
    ------
    IndexError
        if Vector index out of range
    """
    lib = cdll.LoadLibrary('./lib/cvectorlib.so')
    lib.new_vector_fl.restype = c_void_p
    lib.new_vector_fl.argtypes = []
    lib.delete_vector_fl.restype = None
    lib.delete_vector_fl.argtypes = [c_void_p]
    lib.vector_size_fl.restype = c_int
    lib.vector_size_fl.argtypes = [c_void_p]
    lib.vector_get_fl.restype = c_float
    lib.vector_get_fl.argtypes = [c_void_p, c_int]
    lib.vector_set_fl.restype = None
    lib.vector_set_fl.argtypes = [c_void_p, c_int, c_float]
    lib.vector_push_back_fl.restype = None
    lib.vector_push_back_fl.argtypes = [c_void_p, c_float]
    lib.read_from_binary_fl.restype = c_int
    lib.read_from_binary_fl.argtypes = [c_void_p, c_char_p, c_int, c_int]

    def __init__(self):
        self.vector = Vector_fl.lib.new_vector_fl()

    def __del__(self):                                  # when reference count hits 0 in Python,
        Vector_fl.lib.delete_vector_fl(self.vector)     # call C++ vector destructor

    def __len__(self):
        return Vector_fl.lib.vector_size_fl(self.vector)

    def __getitem__(self, i):
        if 0 <= i < len(self):
            return Vector_fl.lib.vector_get_fl(self.vector, c_int(i))
        raise IndexError('Vector index out of range')

    def __setitem__(self, i, x):
        if 0 <= i < len(self):
            return Vector_fl.lib.vector_set_fl(self.vector, c_int(i), c_float(x))
        raise IndexError('Vector index out of range')

    def __repr__(self):
        return '[{}]'.format(', '.join(str(self[i]) for i in range(len(self))))

    def push(self, x):
        Vector_fl.lib.vector_push_back_fl(self.vector, c_float(x))

    def read_from_binary(self, file_name, neuron_index: int, matrix_size: int):
        return Vector_fl.lib.read_from_binary_fl(self.vector, c_char_p(file_name.encode('utf-8')), c_int(neuron_index), c_int(matrix_size))

    def to_numpy_array(self):
        return np.array([x for x in self])

    def to_list(self):
        return [x for x in self]