#ifndef __IO_SOLVE_CUH__
#define __IO_SOLVE_CUH__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include "param.h"
#include "io.cuh"
#include "io_ion.cuh"
#include "solve_bem.cuh"
#include "solve_rkc.cuh"
#include "solve_cnm.cuh"
#include "syn.cuh"


#define io_n_vec_RK4 4 

__host__ neuron_solve_t *io_solve_initialize ( neuron_solve_t *, const char *, neuron_t *, neuron_t * );
__host__ void io_solve_finalize ( const int , neuron_solve_t *, neuron_solve_t * );
__host__ void io_solve_update_v ( neuron_t *, neuron_solve_t *, neuron_t *, neuron_solve_t *, gap_t * );

#endif // __IO_SOLVE_CUH__
