#ifndef __GR_CUH__
#define __GR_CUH__
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "param.h"
#include "gr_ion.cuh"
//#include <cuda_runtime.h>

#define GR_COMP ( 578 )
#define PARAM_FILE_GR "./Gr.txt"

#define V_INIT_GR   ( -80.0 )
#define V_LEAK1_GR  ( -16.5 )
#define V_LEAK2_GR  ( -65.0 )
#define V_LEAK3_GR  ( -74.5 )
#define V_Na_GR     (  87.39 )
#define V_K_GR      ( -84.69 ) //-84.39//-94.6
#define V_Ca_GR	    ( 129.33 )

#define SHELL1_D    ( 0.2e-4 ) // [0.2mum -> 0.2e-4cm]
#define B_Ca1       ( 1.5 ) // [/ms] // 0.6; Tsuyuki // 1.5; //Dover 
#define Ca1_0       ( 1e-4 ) // [mM] // 2.25e-4; Tsuyuki// 1e-4; //Dover
#define Ca1OUT      ( 2.0 )  // [mM]
#define F           ( 9.6485e4 ) // Faraday constant [s*A/mol]

#define gr_n_comptype 9
typedef enum { GR_dend1, GR_dend2, GR_dend3, GR_dend4,
	           GR_soma, GR_hill, GR_AIS, GR_axon, GR_pf } gr_comptype_t;

//#define gr_n_elem 7
//typedef enum { v, Ca, Cm, area, connect, compart, i_ext } go_elem_t;
// declared in param.h

#define gr_n_ion 20
typedef enum { n_KV, a_KA, b_KA, c_KCa, s_KM, ir_KIR, ch_Ca, ci_Ca,
	       o_Na, c1_Na, c2_Na, c3_Na, c4_Na, c5_Na, i1_Na, i2_Na, i3_Na, i4_Na, i5_Na, i6_Na } gr_ion_t;

#define gr_n_cond 10
typedef enum { g_leak1, g_leak2, g_leak3, g_Na, g_KV, g_KA, g_KCa, g_KM, g_KIR, g_Ca } gr_cond_t;

__host__ neuron_t *gr_initialize ( const int, const int, const char *, neuron_t * );
__host__ void gr_finalize ( neuron_t *, neuron_t *, FILE *, FILE * );
__global__ void gr_set_current ( neuron_t *, const double );
__host__ void gr_output_file  ( neuron_t *, double *, double *, const double, FILE *, FILE * );

#endif // __GR_CUH__
