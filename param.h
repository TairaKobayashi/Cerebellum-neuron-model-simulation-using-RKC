#ifndef _PARAM_H_
#define _PARAM_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define TS ( 1500 )	 //[ms]
#define S_Stimuli 900.0//( 900.0 )
#define E_Stimuli 1000.0//( 1100.0 )
#define BE_DT ( 0.025 )	 //[ms]
#define CN_DT ( 0.025 )//( 0.025 )  //[ms]
#define RKC_DT ( 0.125 )//( 0.125 )  //[ms]

#define MAX_BUF 1024

typedef enum { kBE, kCN, kRKC, kSolver_num } kSolver_type;
typedef enum { N_TYPE_GR, N_TYPE_GO, N_TYPE_PKJ, N_TYPE_IO, NEURON_TYPE } neuron_type_t;

typedef enum { v, Ca, Cm, area, connect, compart, i_ext, n_elem } elem_t;
typedef enum { cn_gamma1, cn_gamma2, cn_ommega1, cn_ommega2, cn_v_old, n_vec_CNm } vec_CNm_t;
typedef enum { y, yn_r, fn, vtemp1, vtemp2, ede, v_new, n_vec_RKC } vec_RKC_t;
typedef enum { mmax, absh, sprad, naccpt, atol_r, rtol, idid, gt, gt_old, lt, tend, errold, hold, nstsig, n_vec_RKC_others} vec_RKC_others_t;
typedef enum { last, newspc, jacatt, n_h_bool } h_bool_t;// true, true, false
typedef enum { cellGO, cellGR, n_type_cell } type_cell_t;
//typedef enum { rk4_k1, rk4_k2, rk4_k3, rk4_k4 } go_vec_RK4_t;

typedef struct {
  int n_solver, neuron_type;
  int n, nx, ny, nc;
  double **elem, **ion, **cond;
  double *ca2, *rev_ca2, *ca_old;
  double *shell;
  double DT;
} neuron_t;

typedef struct {
  int nnz;
  double *val, *val_ori, *b;
  int *col, *row, *dig;
  double **vec;
  double *h_work, *h_others;// for RKC
  bool *h_bool;// for RKC
  double *dammy;  
  int numThreadsPerBlock, numBlocks;  // Device array
  char type [ MAX_BUF ];
} neuron_solve_t;

typedef struct _synapse_t synapse_t;
// declared in syn.cuh
typedef struct _gap_t gap_t;

#endif // _PARAM_H_
