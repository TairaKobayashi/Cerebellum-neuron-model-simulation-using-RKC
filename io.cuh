#ifndef __IO_CUH__
#define __IO_CUH__
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "param.h"
#include "io_ion.cuh"
//#include <cuda_runtime.h>

#define IO_COMP ( 3 )
#define IO_COMP_DEND ( 1 )
#define IO_COMP_SOMA ( 1 )
#define IO_COMP_AXON ( 1 )
//#define PARAM_FILE_IO "./IO.txt"
#define PARAM_FILE_IO "./IO.txt"

#define V_INIT_IO ( -60.0 )
//Hasegawa
#define V_LEAK_IO   ( 10.0 )  
#define V_Na_IO     ( 55.0 )  
#define V_Ca_IO     ( 120.0 ) 
#define V_K_IO      ( -75.0 ) 
#define V_H_IO      ( -43.0 ) 
//#define Cm          ( 1.0 )           // Cm (単位:μF/cm^2)

// passive leak conductance
#define G_LEAK_IO      (0.016)         // passive leak conductance  (mS/cm^2), dendrite, soma, axon-hilloc 共通
// Somatic conductances
#define SOMA_G_CAL_IO  (0.83)           // soma, Low-threshold calcium (default) (mS/cm^2) : range 0.55 - 0.9 (mS/cm^2)
//#define SOMA_G_CAL  (0.55)           // soma, Low-threshold calcium (default) (mS/cm^2)
#define SOMA_G_NA_IO   (120.0)         // soma, Sodium (mS/cm^2)
#define SOMA_G_KDR_IO  (9.0)           // soma, Potassium, slow component (mS/cm^2)
#define SOMA_G_K_IO    (5.0)           // soma, Potassium, fast compornet (mS/cm^2)
// Dendritic conductances
#define DNDR_G_H_IO    (0.15)          // dendrite, h-current (mS/cm^2)
#define DNDR_G_CAH_IO  (4.5)           // dendrite, High-threshold calcium (mS/cm^2)
#define DNDR_G_KCA_IO  (35.0)          // dendrite, Calcium-dependent potassium (mS/cm^2)
// Axon hillock conductances
#define AXON_G_NA_IO   (240.0)         // axon-hilloc, Sodium (mS/cm^2)
#define AXON_G_K_IO    (20.0)          // axon-hilloc, Potassium (mS/cm^2)
// Cell morphology
#define G_INT_IO       (0.13)          // G_internal, Cell morphology conductance (mS/cm^2)
#define G_P_DS_IO      (0.200)         // deendrite:soma = 4:1, Cell morphology parameter, 0.200 = (1.0 / (4.0 + 1.0)) * 1.0
//#define G_P_DS_IO      (0.875)         // deendrite:soma = (4/28):1, Cell morphology parameter, 0.875 = (1.0 / (0.143 + 1.0)) * 1.0
#define G_P_SA_IO      (0.869)         // soma:axon-hillock = 20:3, Cell morphology parameter, 0.869 = (1.0 / (20.0 + 3.0)) * 20.0
// Maximum coupling strength between olivary cells
#define G_C_IO         (0.04)          // 0.04 mS/cm^2 for all simulations of ensembles of cells.

#define Ri_IO          (0.2)           // Ri (単位:kΩ-cm) : 200Ω-cm  Yair Manor (1995), Construction of an Experimentally-Based Neuronal Network Model to Explore the Function of the Inferior Olive Nucleus　
#define CA_PH_IO       (3.0)           // Ca2+ ion current to ion concentration (checked paper, P.806)
#define BETA_CA_IO     (0.075)         // (単位:1/mS) (checked paper, P.816)

typedef enum { io_dend, io_soma, io_axon, io_n_comptype } io_comptype_t;

typedef enum { k_CaL_io, l_CaL_io, m_Na_io, h_Na_io, n_Kdr_io, p_Kdr_io,
                 x_K_io, r_CaH_io, s_KCa_io, q_H_io, io_n_ion } io_ion_t;

typedef enum { g_leak_io, g_CaL_io, g_Na_io, g_Kdr_io, g_K_io, g_CaH_io, g_KCa_io, g_H_io, io_n_cond } io_cond_t;

__host__ neuron_t *io_initialize ( const int, const int, const char *, neuron_t * );
__host__ void io_finalize ( neuron_t *, neuron_t *, FILE *, FILE * );
__global__ void io_set_current ( neuron_t *, const double );
__host__ void io_output_file  ( neuron_t *, double *, double *, const double, FILE *, FILE * );

#endif // __IO_CUH__
