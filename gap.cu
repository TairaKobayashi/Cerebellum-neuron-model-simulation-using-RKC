#include "gap.cuh"

//////////////////////////////// IO_GJ //////////////////////////////
__global__ static 
void io_gap_initialize ( int *d_comp, double *d_elem, const int nx, const int ny, const int num_io_gap )
{    
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int num_io = nx * ny;

  if ( id < num_io )
  {
    int idx = id % nx;
    int idy = id / ny;
    for ( int i = 0; i < 4; i++ ) { d_comp [ post_comp_gap   * num_io_gap + id * 4 + i ] =  id; }
    for ( int i = 0; i < 4; i++ ) { d_elem [ gap_current * num_io_gap + id * 4 + i ] = 0.0; }

    if ( idx == 0 && idy == 0 )
    {
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 0 ] = -1;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 1 ] = -1;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 2 ] = id + 1;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 3 ] = id + nx;
    }
    else if ( idx == nx - 1 && idy == 0 )
    {
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 0 ] = id - 1;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 1 ] = -1;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 2 ] = -1;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 3 ] = id + nx;
    }
    else if ( idx == 0 && idy == ny - 1 )
    {
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 0 ] = -1;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 1 ] = id - nx;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 2 ] = id + 1;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 3 ] = -1;
    }
    else if ( idx == nx - 1 && idy == ny - 1 )
    {
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 0 ] = id - 1;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 1 ] = id - nx;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 2 ] = -1;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 3 ] = -1;
    }
    else if ( idy == 0 ) // idx != 0 || nx - 1
    {
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 0 ] = id - 1;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 1 ] = -1;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 2 ] = id + 1;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 3 ] = id + nx;
    }
    else if ( idy == ny - 1 ) // idx != 0 || nx - 1, idy != 0
    {
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 0 ] = id - 1;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 1 ] = id - nx;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 2 ] = id + 1;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 3 ] = -1;
    }
    else if ( idx == 0 ) // idy != 0 || ny - 1
    {
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 0 ] = -1;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 1 ] = id - nx;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 2 ] = id + 1;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 3 ] = id + nx;
    }
    else if ( idx == nx - 1 ) // idy != 0 || ny - 1, idx == 0
    {
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 0 ] = id - 1;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 1 ] = id - nx;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 2 ] = -1;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 3 ] = id + nx;
    }/*
    else if ( id % 3 == 0 ){
      int p_id = id - 1;
      int p_idx = p_id % nx;
      int p_idy = p_id / ny;
      if ( p_idx != nx - 1 && p_idx != 0 && p_idy != 0 && p_idy != ny - 1 ){           
        d_comp [ pre_comp_gap * num_io_gap + id * 4 + 0 ] = -1;
        d_comp [ pre_comp_gap * num_io_gap + id * 4 + 1 ] = id - nx;
        d_comp [ pre_comp_gap * num_io_gap + id * 4 + 2 ] = id + 1;
        d_comp [ pre_comp_gap * num_io_gap + id * 4 + 3 ] = id + nx;
      }
      else
      {
        d_comp [ pre_comp_gap * num_io_gap + id * 4 + 0 ] = id - 1;
        d_comp [ pre_comp_gap * num_io_gap + id * 4 + 1 ] = id - nx;
        d_comp [ pre_comp_gap * num_io_gap + id * 4 + 2 ] = id + 1;
        d_comp [ pre_comp_gap * num_io_gap + id * 4 + 3 ] = id + nx;
      }
    }
    else if ( (id + 1 ) % 3 == 0 ){
      int p_id = id + 1;
      int p_idx = p_id % nx;
      int p_idy = p_id / ny;
      if ( p_idx != nx - 1 && p_idx != 0 && p_idy != 0 && p_idy != ny - 1 ){
        d_comp [ pre_comp_gap * num_io_gap + id * 4 + 0 ] = id - 1;
        d_comp [ pre_comp_gap * num_io_gap + id * 4 + 1 ] = id - nx;
        d_comp [ pre_comp_gap * num_io_gap + id * 4 + 2 ] = -1;
        d_comp [ pre_comp_gap * num_io_gap + id * 4 + 3 ] = id + nx;
      }
      else
      {
        d_comp [ pre_comp_gap * num_io_gap + id * 4 + 0 ] = id - 1;
        d_comp [ pre_comp_gap * num_io_gap + id * 4 + 1 ] = id - nx;
        d_comp [ pre_comp_gap * num_io_gap + id * 4 + 2 ] = id + 1;
        d_comp [ pre_comp_gap * num_io_gap + id * 4 + 3 ] = id + nx;
      }
    }*/
    else
    {
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 0 ] = id - 1;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 1 ] = id - nx;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 2 ] = id + 1;
      d_comp [ pre_comp_gap * num_io_gap + id * 4 + 3 ] = id + nx;
    }

    for ( int i = 0; i < 4; i++ ) { d_comp [ post_comp_gap * num_io_gap + id * 4 + i ] *= IO_COMP; }
    for ( int i = 0; i < 4; i++ ) { d_comp [ pre_comp_gap  * num_io_gap + id * 4 + i ] *= IO_COMP; }
  }
}

__host__ 
gap_t *io_gap_create ( const int nx, const int ny )
{
  int num_io_gap;
  if ( nx * ny > 1 ) { num_io_gap = nx * ny * 4; }
  else { num_io_gap = 0; }

  gap_t *d_io_gap = ( gap_t * ) malloc ( sizeof ( gap_t ) );
  d_io_gap -> n = num_io_gap;  

  if ( num_io_gap == 0 ) { printf ( "# of io_GJ is 0\n" ); return d_io_gap; }

  cudaMalloc ( ( int    ** ) & ( d_io_gap -> comp ), gap_n_comp * num_io_gap * sizeof ( int    ) );  
  cudaMalloc ( ( double ** ) & ( d_io_gap -> elem ), gap_n_elem * num_io_gap * sizeof ( double ) );    
  io_gap_initialize <<< ( ( nx * ny ) + 127 ) / 128, 128 >>> ( d_io_gap -> comp, d_io_gap -> elem, nx, ny, num_io_gap ); 
  printf ( "io_GJ = %d\n", num_io_gap );
  return d_io_gap;
}

__global__ 
void io_gap_update ( neuron_t *d_io, int *gap_comp, double *gap_elem, const int num_io_gap ) 
{  
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < num_io_gap )
  {
    int pre  = gap_comp [ pre_comp_gap  * num_io_gap + id ];
    int post = gap_comp [ post_comp_gap * num_io_gap + id ];
    
    //double post_v = d_io -> elem [ v ] [ post ];
    //double pre_v;
    double diff_v = 0.0;
    ( pre > -1 )? diff_v = ( d_io -> elem [ v ] [ post ] - d_io -> elem [ v ] [ pre ] ) : diff_v = 0.0;
    
    gap_elem [ gap_current * num_io_gap + id ] 
      = G_C_IO * ( 0.8 * exp ( - 0.01 * diff_v * diff_v ) + 0.2 ) * ( diff_v );
    //gap_elem [ gap_current * num_io_gap + id ] 
   //   = 0.2 * ( diff_v );
  }
}

__host__ 
void io_gap_finalize ( gap_t *d_io_gap )
{
  if ( d_io_gap -> n > 0 )
  {
    cudaFree ( d_io_gap -> comp );
    cudaFree ( d_io_gap -> elem );
  }
  free ( d_io_gap );
}
