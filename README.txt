# Cerebellum_Neuron_Simulation_Using_RKC
This is a program for cerebellum multi-compartment neuron model simulation using an explicit RKC method on a GPU.


# Features
Runge-Kutta-Chebyshev (RKC) method is an explicit solver developed to solve parabolic PDEs ( Sommejier, B.P. et al. 1998 ).
It has adaptive time steps and 2nd order accuracy.

In a multi-compartment neuron model simulation, PDEs that describe the dynamics of neurons have to be solved numerically for each time step, 
but implicit methods need to solve simultaneous equations, which can make numerical simulation slow on GPUs.
Although explicit methods can be unstable against PDEs, using the RKC method shows sufficient stability for most cases, 
better computational performance than the Crank-Nicolson (CN) method on a GPU, and good reproducibility.

This program can simulate cerebellar granule cells (GrCs), Golgi cells (GoCs), Purkinje cells (PCs), and inferior olive cells (IOs) by using the RKC method or the CN method.
Also, simple cellebelar network simulation ( 2048 GrCs, 512 GoCs, and 512 PCs with 780404 synapses ) is possible by rewriting some parameters.

Each neuron's parameters are based on the following original models.
GrC model -> Dover, K., et al. 2016
GoC model -> Solinas, S., et al. 2007
PC model  -> De Schutter, E., et al. 1994
IO model  -> De Grujil, JR., et al. 2012


# Requirement
 
* CUDA : Toolkit 11.1
* CPU  : Compiler GCC 7.5
* OS   : Ubuntu 16.04


# Our environment

* CPU : AMD EPYC 7552
* GPU : Tesla V100 32GB
* Memory : 125 GB


# Usage
After makefile, execute main with solver name (RKC or CN), number of cells on the X axis, and number of cells on the Y axis for GrC, GoC, PC, and IO, respectively. 
For example, when simulating 9 (3*3) PCs using the RKC method, it will be as follows: 
./main CN 0 0 CN 0 0 RKC 0 0 CN 0 0
or run each shell script (gr1.sh, go1.sh, pkj1.sh, or io1.sh).

When simulating the cerebellar network model, you must change some parameters as follows and run network1.sh.
1. bool network_sim: "false" -> "true"  in main.cu
2. PKJ_RD_G_KM: "0.013" -> "0.1" in pkj.cuh

The parameter of random seed, simulation time, input currents of each neuron model is in main.cu, param.h, or "neuron".cu (replace "neuron" with gr, go, pkj, or io).


# Note
If you have any questions, please feel free to contact us at "tairakobayashi.bip at gmail.com"
The program codes will be updated sequentially.


# Author and address
* Taira Kobayashi, Rin Kuriyama, Tadashi Yamazaki
* Graduate School of Informatics and Engineering, The University of Electro-Communications, Tokyo, Japan
* tairakobayashi.bip at gmail.com


# License
The program codes are under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
