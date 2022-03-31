!!!CAUTION!!!
This program cannot be compiled or run on department cluster,
as it requires a newer version of C++ dependencies



How to use the program
----------------------

0 Optional: compile 'start_simulation.cpp'
  > g++ start_simulation.cpp -o start_simulation.exe -O3 -std=c++20

1 Put the folllowings files in the same folder:
  - start_simulation.exe
  - vars.txt
  - the adjacency matrix file (e.g. DIV66_gji)

2 Adjust the variables and settings in 'vars.txt'

3 Execute 'start_simulation.exe'


Output files
------------

1 spkt
  - record the number of spikes & spiking timestamps / spike trains
  - the i-th row corresponds to the spiking data for the i-th neuron node
  - column 1: the number of spikes n(i) of the i-th neuron node
  - column 2 and onwards: n(i) timestamps (in ms)
    at which the spikes occur for the i-th neuron node

2 spks
  - record the number of spikes & spiking timesteps / spike trains
  - the i-th row corresponds to the spiking data for the i-th neuron node
  - column 1: the number of spikes n(i) of the i-th neuron node
  - column 2 and onwards: n(i) timesteps (in simulation step)
    at which the spikes occur for the i-th neuron node

3 memp
  - record the time series of the membrane potential
  - a binary file that can be read by a C++ program into a C++ vector

4 info
  - record the information of the computation (e.g. T, dt, ...)

5 cont
  - record the information of the computation
  - to be used by the program

6 log file


Notes
-----

0 Compatibility
  This code is intended to be compiled in C++20 verion.

1 Time series exported as binary files
  To save disk storage and read/write time, the large time series data are written into binary files,
  i.e., memp.dat, curr.dat, etc.
  These binary files can ONLY be read by a C++ program into a 'vector<float>()'.
  The specific format is (e.g. memp.dat; N = 4095):
    v_1(t0), v_2(t0), v_3(t0), ..., v_4095(t0), v_1(t1), v_2(t1), v_3(t1), ..., v_4095(t1), ...,
    ..., v_1(tn), v_2(tn), v_3(tn), ..., v_4095(tn), ......
  You can make use of 'f.seekg()', 'f.read()' to read the time series of a specific neuron node.
  Examples on how to read the data into C++ and Python programs are provided in the 'lib' folder.

2 Auto determined spiking truncation time
  The program speeds up the execution time by summing only the recent spiking activities.
  The time is denoted by 'trunc_time_inh'/'trunc_time_exc' in 'struct Variables'.
  The value of 'trunc_time', limited by the precision of float or double that a machine can store,
  is calculated from:
    G_i = beta * num of exc incoming links * max of presynaptic w_ij of neuron i * exp(-T_trunc/tau)
        ~ maximum precision of float or double
  =>  T_trunc = tau * ln(beta * p * y)
  where p = maximum precision of float or double,
        y = num of exc incoming links * max of presynaptic w_ij of neuron i.
  disable this function by inserting your desired value of 'trunc_time'.
