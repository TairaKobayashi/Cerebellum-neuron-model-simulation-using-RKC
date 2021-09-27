#ifndef __SOLVE_IMPLICIT_CUH__
#define __SOLVE_IMPLICIT_CUH__
#include <stdio.h>
#include <math.h>
#include "param.h"
#include "go.cuh"
#include "go_solve.cuh"
#include "go_ion.cuh"
#include "gr.cuh"
#include "gr_solve.cuh"
#include "gr_ion.cuh"
#include "pkj.cuh"
#include "pkj_solve.cuh"
#include "pkj_ion.cuh"
#include "io.cuh"
#include "io_solve.cuh"
#include "io_ion.cuh"
#include "syn.cuh"
#include "gap.cuh"
//#include "cg.h"

__host__ void go_solve_by_bem ( neuron_t *, neuron_solve_t *, neuron_t *, neuron_solve_t *, synapse_t * );
__host__ void gr_solve_by_bem ( neuron_t *, neuron_solve_t *, neuron_t *, neuron_solve_t *, synapse_t *, synapse_t * );
__host__ void pkj_solve_by_bem ( neuron_t *, neuron_solve_t *, neuron_t *, neuron_solve_t * );
__host__ void io_solve_by_bem ( neuron_t *, neuron_solve_t *, neuron_t *, neuron_solve_t *, gap_t * );

#endif // __SOLVE_IMPLICIT_CUH__