#ifndef __SYN_CUH__
#define __SYN_CUH__

#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include "param.h"
#include "gr.cuh"
#include "go.cuh"
#include "pkj.cuh"

#define G_MFGR_AMPA ( 0.18e-9 ) //( 24000e-9 ) 
#define G_MFGR_NMDA ( 0.025e-9 ) //( 32000e-9 )
#define E_MFGR ( 0.0 ) //( -3.7 )
#define D_MFGR_AMPA ( exp( - RKC_DT / 1.2 ) )
#define D_MFGR_NMDA ( exp( - RKC_DT / 52.0 ) )
// Decay :exp( - DT / tau ) = 0.90107510572 ( DT = 0.125, tau = 1.2 ), 0.81193634615 (DT = 0.025)
// Decay :exp( - DT / tau ) = 0.99759904077 ( DT = 0.125, tau = 52 ) , 0.99520384614 (DT = 0.025)


#define G_GRGO_AMPA ( 45.5e-9 ) //( 180e-9 ) 
#define G_GRGO_NMDA ( 30.0e-9 ) //( 180e-9 )
#define E_GRGO ( 0.0 ) //( -3.7 )
#define D_GRGO_AMPA  ( exp( - RKC_DT / 1.5   ) )
#define D_GRGO_NMDA1 ( exp( - RKC_DT / 31.0  ) )
#define D_GRGO_NMDA2 ( exp( - RKC_DT / 170.0 ) )
// Decay = exp(-0.125 / tau ) = 0.92004441462 //tau = 1.5  = exp(-0.25 / tau ) = 0.84648172489
// Decay = exp(-0.125 / tau ) = 0.99597586057 //tau = 31   = exp(-0.25 / tau ) = 0.99196791484
// Decay = exp(-0.125 / tau ) = 0.99926497614 //tau = 170  = exp(-0.25 / tau ) = 0.99853049255


#define G_GOGR_GABA ( 0.028e-9 ) //( 61e-9 ) 
#define E_GOGR ( -82.0 ) //( -80.0 )
#define D_GOGR_GABAA ( exp( - RKC_DT / 7.0 ) )
#define D_GOGR_GABAB ( exp( - RKC_DT / 59.0 ) )
// Decay = exp(-0.125 / tau ) = 0.9823013511  //tau = 7  = exp(-0.25 / tau ) = 0.96491594437
// Decay = exp(-0.125 / tau ) = 0.99788359867 //tau = 59  = exp(-0.25 / tau ) = 0.9957716765


#define G_GRPKJ_AMPA ( 0.7e-9 ) //( 210000.0e-9 ) 
#define E_GRPKJ ( 0.0 ) //( -3.7 )
//#define D_GRPKJ_AMPA ( exp( - RKC_DT / 0.6 ) )
#define D_GRPKJ_AMPA ( exp( - RKC_DT / 6.0 ) ) // 6.7  //7.0 // 8.3
// Decay = exp(-0.125 / tau ) = 0.81193634615  //tau = 0.6  = exp(-0.25 / tau ) = 0.6592406302


#define G_MLIPKJ_GABA ( 1.0e-9 ) //( 210000.0e-9 ) 
#define E_MLIPKJ ( -75.0 ) //( -3.7 )

#define N_Syn_Per_GRPKJ 25
#define N_Syn_Per_MLIPKJ 1

#define W_MFGR  40000.0//40000.0 //0.4
#define W_GRGO  64.0 / 4.0 //16.0
#define W_GOGR  160000.0 * 4.0 //80.0

#define W_GRPKJ  1000000.0 * 0.3 // 0.3 //6.25 Best //60000.0 //0.05
#define W_MLIPKJ 0.0//1000.0//5000.0
//#define W_GRPKJ 0.1//1.0
//#define W_PKJDCN 1.0

typedef enum { pre_comp, post_comp, syn_n_comp } syn_compnum_t;

typedef enum { mfgr_ampa,  mfgr_nmda,               mfgr_weight,  mfgr_val,  mfgr_n_elem } mfgr_elem_t;
typedef enum { grgo_ampa,  grgo_nmda1, grgo_nmda2,  grgo_weight,  grgo_val,  grgo_old_v,  grgo_n_elem  } grgo_elem_t;
typedef enum { gogr_gabaa, gogr_gabab,              gogr_weight,  gogr_val,  gogr_old_v,  gogr_n_elem  } gogr_elem_t;
typedef enum { grpkj_ampa,                          grpkj_weight, grpkj_val, grpkj_old_v, grpkj_n_elem } grpkj_elem_t;
typedef enum { mlipkj_gaba, mlipkj_weight, mlipkj_val, mlipkj_n_elem } mlipkj_elem_t;
//typedef enum { grpkj_ampa, grpkj_weight, grpkj_old_v, grpkj_n_elem } grpkj_elem_t;

//fake
//#define pkjdcn_n_elem 1
//typedef enum { pkjdcn_old_v } pkjdcn_elem_t;

struct _synapse_t {
  int n;
  curandState *cstate;
  int *comp;
  double *elem;
  FILE *f_out;
};

extern synapse_t *mfgr_create ( const int );
extern void mfgr_finalize ( synapse_t *, const int );
extern void mf_output_file ( synapse_t *, const double, neuron_t * );
//extern __global__ void mfgr_update ( neuron_t *, int *, double *, const double, const int , curandState * );

extern synapse_t *grgo_create ( const int, const int, const int, const int );
extern void grgo_finalize ( synapse_t *, const int );

extern synapse_t *gogr_create ( const int, const int, const int, const int );
extern void gogr_finalize ( synapse_t *, const int );

extern synapse_t *grpkj_create ( const int, const int, const int, const int );
extern void grpkj_finalize ( synapse_t *, const int );

extern synapse_t *mlipkj_create ( const int, const int );
extern void mlipkj_finalize ( synapse_t *, const int );
extern void mli_output_file ( synapse_t *, const double, neuron_t * );

extern void gr_synapse_update  ( const double, const double, synapse_t *, synapse_t *, neuron_t *, neuron_solve_t * );
extern void go_synapse_update  ( const double, const double, synapse_t *, neuron_t *, neuron_solve_t * );
extern void pkj_synapse_update ( const double, const double, synapse_t *, synapse_t *, neuron_t *, neuron_solve_t * );

#endif // __SYN_CUH__
