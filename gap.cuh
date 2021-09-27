#ifndef __GAP_CUH__
#define __GAP_CUH__

#include <stdio.h>
#include <stdlib.h>
#include "param.h"
#include "io.cuh"

typedef enum { pre_comp_gap, post_comp_gap, gap_n_comp } gap_comp_t;

typedef enum { gap_current, gap_n_elem } gap_elem_t;

struct _gap_t {
  int n;
  int *comp;
  double *elem;
};

__host__ gap_t *io_gap_create ( const int, const int );
__global__ void io_gap_update ( neuron_t *, int *, double *, const int ); 
__host__ void io_gap_finalize ( gap_t * );

#endif // __GAP_CUH__
