#ifndef __SOLVE_RKC_CUH__
#define __SOLVE_RKC_CUH__
#include <stdio.h>
#include <math.h>
#include "param.h"
#include "gr.cuh"
#include "gr_solve.cuh"
#include "gr_ion.cuh"
#include "go.cuh"
#include "go_solve.cuh"
#include "go_ion.cuh"
#include "pkj.cuh"
#include "pkj_solve.cuh"
#include "pkj_ion.cuh"
#include "io.cuh"
#include "io_solve.cuh"
#include "io_ion.cuh"
#include "syn.cuh"
#include "gap.cuh"
//#include "cg.h"

#define UROUND ( 2.22e-16 )
#define H_MAX ( RKC_DT )
#define H_MIN_GR  ( pow ( 2.0, -12.0 ) ) //( pow ( 2.0, -19.0 ) ) // pow ( 2.0, -10.0 )
#define H_MIN_GO  ( pow ( 2.0, -10.0 ) ) // pow ( 2.0, -10.0 )
#define H_MIN_PKJ ( pow ( 2.0, -10.0 ) ) //( pow ( 2.0, -10.0 ) ) 
#define H_MIN_IO  ( pow ( 2.0, -10.0 ) ) // pow ( 2.0, -10.0 )
#define TOL_GR  ( 1.0e-9 )
#define TOL_GO  ( 1.0e-6 )//( 1.0e-6 )
#define TOL_PKJ ( 1.0e-6 )//( 1.0e-6 )
#define TOL_IO  ( 1.0e-6 )

__host__ void gr_solve_by_rkc ( neuron_t *, neuron_solve_t *, neuron_t *, neuron_solve_t *, synapse_t *, synapse_t * );
__host__ void go_solve_by_rkc ( neuron_t *, neuron_solve_t *, neuron_t *, neuron_solve_t *, synapse_t * );
__host__ void pkj_solve_by_rkc ( neuron_t *, neuron_solve_t *, neuron_t *, neuron_solve_t *, synapse_t *, synapse_t * );
__host__ void io_solve_by_rkc ( neuron_t *, neuron_solve_t *, neuron_t *, neuron_solve_t *, gap_t * );
__host__ void RKC_vec_initialize (  neuron_t *, neuron_solve_t *, neuron_t *, neuron_solve_t * );

#endif // __SOLVE_RKC_CUH__
