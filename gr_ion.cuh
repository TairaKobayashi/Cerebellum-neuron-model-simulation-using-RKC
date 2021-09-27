#ifndef __GR_ION_CUH__
#define __GR_ION_CUH__
#include <math.h>
#include <stdlib.h>
#include "param.h"
#include "gr.cuh"
#include "gr_solve.cuh"


__global__ void gr_Na_update_2order ( const int, const double *, const double *, const double, const double *,
    double *,  double *, double *, double *, double *, double *,
    double *, double *, double *, double *, double *, double * );
__global__ void gr_Na_update ( const int, const double *, const double, const double *,
    double *,  double *, double *, double *, double *, double *,
    double *, double *, double *, double *, double *, double * );
__global__ void gr_update_ion_RKC_RK4 ( neuron_t *, neuron_solve_t *, double *, const double );//extern void gr_update_ion_exeuler ( gr_t *, const double );
__global__ void gr_update_ion_exp_imp ( neuron_t *, neuron_solve_t *, const double );//extern void gr_update_ion_exeuler ( gr_t *, const double );
__global__ void gr_update_ion_RKC_exp_imp ( neuron_t *, neuron_solve_t *,double *, const double );//extern void gr_update_ion_exeuler ( gr_t *, const double );
__global__ void gr_update_ion_RK2 ( neuron_t *, neuron_solve_t *, const double );//extern void gr_update_ion_exeuler ( gr_t *, const double );
__global__ void gr_update_ion_RKC_RK2 ( neuron_t *, neuron_solve_t *, double *, const double );//extern void gr_update_ion_exeuler ( gr_t *, const double );
__global__ void gr_update_ion ( neuron_t *, neuron_solve_t *, const double );//extern void gr_update_ion_exeuler ( gr_t *, const double );
__global__ void gr_update_ion_RKC ( neuron_t *, neuron_solve_t *, double *, const double );//extern void gr_update_ion_exeuler ( gr_t *, const double );
__host__ void gr_initialize_ion ( neuron_t * );

#endif // __GR_ION_CUH__
