#ifndef __PKJ_ION_CUH__
#define __PKJ_ION_CUH__
#include <math.h>
#include <stdlib.h>
#include <math.h>
#include "param.h"
#include "pkj.cuh"
#include "pkj_solve.cuh"


__global__ void pkj_update_ion ( neuron_t *, neuron_solve_t *, const double );//extern void pkj_update_ion_exeuler ( pkj_t *, const double );
__global__ void pkj_update_ion_2nd ( neuron_t *, neuron_solve_t *, const double );//extern void pkj_update_ion_exeuler ( pkj_t *, const double );
__global__ void pkj_update_ion_RK2 ( neuron_t *, neuron_solve_t *, const double );//extern void pkj_update_ion_exeuler ( pkj_t *, const double );
__global__ void pkj_update_ion_RKC ( neuron_t *, neuron_solve_t *, double *, const double );//extern void pkj_update_ion_exeuler ( pkj_t *, const double );
__host__ void pkj_initialize_ion ( neuron_t * );

#endif // __PKJ_ION_CUH__
