#ifndef __IO_ION_CUH__
#define __IO_ION_CUH__
#include <math.h>
#include <stdlib.h>
#include "param.h"
#include "io.cuh"
#include "io_solve.cuh"

__global__ void io_update_ion_2nd ( neuron_t *, neuron_solve_t *, const double );//extern void io_update_ion_exeuler ( io_t *, const double );
__global__ void io_update_ion_RKC ( neuron_t *, neuron_solve_t *,double *, const double );//extern void io_update_ion_exeuler ( io_t *, const double );
__global__ void io_update_ion ( neuron_t *, neuron_solve_t *, const double );//extern void io_update_ion_exeuler ( io_t *, const double );
__host__ void io_initialize_ion ( neuron_t * );

#endif // __IO_ION_CUH__
