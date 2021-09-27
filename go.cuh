#ifndef __GO_CUH__
#define __GO_CUH__
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "param.h"
#include "go_ion.cuh"
//#include <cuda_runtime.h>
/*
//5 comps
#define GO_COMP ( 5 )
#define GO_COMP_DEND ( 1 )
#define GO_COMP_AXON ( 1 )
#define PARAM_FILE_GO "./Go_5.txt"
*/
///*
//131 comps
#define GO_COMP ( 131 )
#define GO_COMP_DEND ( 10 )
#define GO_COMP_AXON ( 100 )
#define PARAM_FILE_GO "./Go_131.txt"
//*/
/////////////Dveice array/////////////////
//#define numThreadsPerBlock ( 128 )
//#define numBlocks ( ( int ) ( ( GO_X * GO_Y * GO_COMP ) / 128 ) + 1  )

#define V_INIT_GO ( -60.0 )
#define V_LEAK_GO ( -55.0 )
#define V_Na_GO   ( 87.39 )
#define V_K_GO    ( -84.69 )
#define V_Ca_GO   ( 120.0 ) 
#define V_H_GO    ( -20.0 )

#define SHELL1_D_GO ( 0.2e-4  ) // [0.2mum -> 0.2e-4cm]
#define B_Ca1_GO    ( 1.3     )//( 1.5     ) // [/ms]
#define Ca1_0_GO    ( 5e-5    )//( 1e-4    ) // [mM]
#define Ca1OUT_GO   ( 2.0     )  // [mM]
#define F_GO        ( 9.6485e4 ) // Faraday constant [s*A/mol]

#define G_GJ_GO    ( 2.5e-6 ) //( 2.5e-6 ) // Gap Junctional Conductance [mS]

//#define go_n_elem 7
//typedef enum { v, Ca, Cm, area, connect, compart, i_ext } go_elem_t;
// declared in param.h

#define go_n_ion 24
typedef enum { m_NaT_go, h_NaT_go, r_NaR_go, s_NaR_go, p_NaP_go, n_KV_go, a_KA_go, b_KA_go, c_KC_go, 
	       sl_Kslow_go, ch_CaHVA_go, ci_CaHVA_go, cl_CaLVA_go, cm_CaLVA_go,
	       hf_HCN1_go, hf_HCN2_go, hs_HCN1_go, hs_HCN2_go,
	       o1_KAHP_go, o2_KAHP_go, c1_KAHP_go, c2_KAHP_go, c3_KAHP_go, c4_KAHP_go } go_ion_t;

#define go_n_cond 13
typedef enum { g_leak_go, g_NaT_go, g_NaR_go, g_NaP_go, g_KV_go, g_KA_go, g_KC_go,
	       g_Kslow_go, g_CaHVA_go, g_CaLVA_go, g_HCN1_go, g_HCN2_go, g_KAHP_go } go_cond_t;


__host__ neuron_t *go_initialize ( const int, const int, const char *, neuron_t * );
__host__ void go_finalize ( neuron_t *, neuron_t *, FILE *, FILE * );
__global__ void go_set_current ( neuron_t *, const double );
__host__ void go_output_file  ( neuron_t *, double *, double *, const double, FILE *, FILE * );

#endif // __GO_CUH__
