#ifndef __GO_ION_CUH__
#define __GO_ION_CUH__
#include <math.h>
#include <stdlib.h>
#include "param.h"
#include "go.cuh"
#include "go_solve.cuh"

__global__ void go_KAHP_update_2order ( const int, const double *, double *, double *, double *, 
    double *, double *, double *, double *, const double );
__global__ void go_KAHP_update ( const int, const double *, double *, double *, 
    double *, double *, double *, double *, const double );
__global__ void go_update_ion_exp_imp ( neuron_t *, neuron_solve_t *, const double );//extern void gr_update_ion_exeuler ( gr_t *, const double );
__global__ void go_update_ion_RKC_exp_imp ( neuron_t *, neuron_solve_t *, double *, const double );//extern void gr_update_ion_exeuler ( gr_t *, const double );
__global__ void go_update_ion ( neuron_t *, neuron_solve_t *, const double );//extern void go_update_ion_exeuler ( go_t *, const double );
__global__ void go_update_ion_RKC ( neuron_t *, neuron_solve_t *, double *, const double );//extern void go_update_ion_exeuler ( go_t *, const double );
__host__ void go_initialize_ion ( neuron_t * );

#endif // __GO_ION_CUH__
