#include "syn.cuh"
#include <time.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>

// Debug
__global__ static void 
printf_GPU ( const int *comp, const double *elem, const int n_syn, const int n_comp, const int n_elem )
{
  for ( int i = 0; i < n_syn; i++ )
  {
    //printf ( "Pre = %d, PostNum = %d, PostComp = %d\n", comp [ i ] / 578, comp [ n_syn + i ] % 1600, comp [ n_syn + i ] / 1600 );
    printf ( "Pre = %d, Post = %d\n", comp [ i ], comp [ n_syn + i ] );
    for ( int j = 0; j < n_elem; j++ ) { printf ( "%f, ", elem [ j * n_syn + i ] ); }
    printf ("\n");
  }
}

__host__ __device__ static
void reset_array ( int * array, int num ) 
{ 
  for ( int i = 0; i < num; i++ )
  {
    array [ i ] = -1; 
  }
}
__host__ __device__ static
void reset_zero ( int * array, int num ) 
{ 
  for ( int i = 0; i < num; i++ )
  {
    array [ i ] = 0; 
  }
}

__global__ 
void setCurand ( unsigned long seed, curandState *state, const int num )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < num ) { curand_init ( seed, id, 0, & ( state [ id ] ) ); }
}

//////////////////////////////// MFGR //////////////////////////////
__global__ static 
void mfgr_initialize ( int *d_comp, double *d_elem, const int num_mfgr, const int n_gr )
{    
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < n_gr )
  {
    for( int j = 0; j < 4; j++ )
    {
      d_comp [ pre_comp  * num_mfgr + id * 4 + j ] = id; // not use
      d_comp [ post_comp * num_mfgr + id * 4 + j ] = j + id * GR_COMP;
      d_elem [ mfgr_ampa   * num_mfgr +  id * 4 + j ] = 0.0;
      d_elem [ mfgr_nmda   * num_mfgr +  id * 4 + j ] = 0.0;
      d_elem [ mfgr_weight * num_mfgr +  id * 4 + j ] = W_MFGR / 4.0;
      d_elem [ mfgr_val    * num_mfgr +  id * 4 + j ] = 0.0;
    }
  }
}

__host__ 
synapse_t *mfgr_create ( const int n_gr )
{
  int num_mfgr = n_gr * 4;

  synapse_t *d_mfgr = ( synapse_t * ) malloc ( sizeof ( synapse_t ) );
  d_mfgr -> n = num_mfgr;  

  if ( num_mfgr == 0 ) { printf ( "# of mfgr = 0\n" ); return d_mfgr; }
  else { printf ( "# of mfgr = %d\n", num_mfgr ); }

  cudaMalloc ( ( int    ** ) & ( d_mfgr -> comp ), syn_n_comp  * num_mfgr * sizeof ( int    ) );  
  cudaMalloc ( ( double ** ) & ( d_mfgr -> elem ), mfgr_n_elem * num_mfgr * sizeof ( double ) );    
  d_mfgr -> f_out = fopen ( "MF_RASTER.csv", "w" );

  mfgr_initialize <<< ( ( n_gr ) + 127 ) / 128, 128 >>> ( d_mfgr -> comp, d_mfgr -> elem, num_mfgr, n_gr );
   
  // Set rand
  cudaMalloc ( ( void ** ) &( d_mfgr -> cstate ), num_mfgr * sizeof ( curandState ) );
  setCurand <<< ( num_mfgr + 127 ) / 128, 128 >>>  ( rand (), d_mfgr -> cstate, num_mfgr );
  return d_mfgr;
}

__global__ 
void mfgr_update ( int *mfgr_comp, double *mfgr_elem, const double t, const int num_mfgr, curandState *S ) 
{  
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < num_mfgr )
  {
    int firing_flag;
    double fr_mf = 5.0;
    //( 0 <= t && t < 250 )? fr_mf = 60.0 : fr_mf = 5.0;
    ( S_Stimuli <= t && t < E_Stimuli )? fr_mf = 60.0 : fr_mf = 5.0;

    double f = curand_uniform ( & ( S [ id ] ) );
    ( fr_mf * RKC_DT * 0.001 > f )? firing_flag = 1 : firing_flag = 0;
    
    // Decay :exp( - DT / tau ) = 0.90107510572 ( DT = 0.125, tau = 1.2 )// gmax = 24000e-9, 0.81193634615 (DT = 0.025)
    mfgr_elem [ mfgr_ampa * num_mfgr + id ] = mfgr_elem [ mfgr_ampa * num_mfgr + id ] * D_MFGR_AMPA + firing_flag;
    // Decay :exp( - DT / tau ) = 0.99759904077 ( DT = 0.125, tau = 52 ) // gmax = 32000e-9, 0.99520384614 (DT = 0.025)
    mfgr_elem [ mfgr_nmda * num_mfgr + id ] = mfgr_elem [ mfgr_ampa * num_mfgr + id ] * D_MFGR_NMDA + firing_flag;
    mfgr_elem [ mfgr_val  * num_mfgr + id ] = mfgr_elem [ mfgr_weight * num_mfgr + id ]  
                                           * ( G_MFGR_AMPA * mfgr_elem [ mfgr_ampa * num_mfgr + id ] 
                                             + G_MFGR_NMDA * mfgr_elem [ mfgr_nmda * num_mfgr + id ] ); // 0.88 : 0.12

    //int l_comp = mfgr_comp [ post_comp * num_mfgr + id ];
   // d_gr -> elem [ g_syn ] [ l_comp ] = mfgr_elem [ mfgr_val  * num_mfgr + id ]; // +=大丈夫
  }
}

__host__
void mf_output_file ( synapse_t *d_mfgr, const double t, neuron_t *p_gr )
{
  FILE *f = d_mfgr -> f_out;
  double *ret = ( double * ) malloc ( sizeof ( double ) * mfgr_n_elem * d_mfgr -> n );
  cudaMemcpy ( ret, d_mfgr -> elem, mfgr_n_elem * d_mfgr -> n * sizeof ( double ), cudaMemcpyDeviceToHost );
  double val = 0.0;
  fprintf ( f, "%lf,", t );
  for ( int j = 0; j < d_mfgr -> n; j++ ) {
    val = G_MFGR_AMPA * 0.88 * ret [ mfgr_ampa * d_mfgr -> n + j ] + G_MFGR_NMDA * 0.12 * ret [ mfgr_nmda * d_mfgr -> n + j ];
    val *= ret [ mfgr_weight * d_mfgr -> n + j ] *1000000;
    fprintf ( f, "%lf,", val );
  }
  fprintf ( f, "\n" );
  free ( ret ); 
}

__host__ 
void mfgr_finalize ( synapse_t *d_mfgr , const int n_gr )
{
  if ( n_gr > 0 )
  {
    cudaFree ( d_mfgr -> comp );
    cudaFree ( d_mfgr -> elem );
    cudaFree ( d_mfgr -> cstate );
    fclose   ( d_mfgr -> f_out );
  }
  free ( d_mfgr );
}


//////////////////////////////// GRGO //////////////////////////////
__global__ static 
void grgo_initialize ( int *d_comp, double *d_elem, const int n_gr, 
                      const int nx_gr, const int ny_gr, const int nx_go, const int ny_go, 
                      const int num_grgo, const int *d_label_gogr, 
                      const int *d_num_syn_gr, const int *d_num_syn_go )
{    
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < n_gr )
  {
    int l_num_syn = d_num_syn_gr [ id ]; // # of synapses from id-th GrC
    int s_syn_id = 0;                 // # of synapses from GrCs 0 to (id-1)
    for ( int j = 0; j < id; j++ ) { s_syn_id += d_num_syn_gr [ j ]; }

    for ( int i = s_syn_id; i < s_syn_id + l_num_syn; i++ )
    {
    int l_n_gr = id;
    int l_n_go = d_label_gogr [ l_n_gr * ny_go + i - s_syn_id ];

    int iy_gr = l_n_gr / nx_gr; int iy_go = l_n_go / nx_go;
    float diff_grgo = 8.0 + iy_go * 32.0 - iy_gr * 16.0; // Distance between GrC and GoC somas
    int gr_ax_comp = 250 + 77 - ( int ) ( diff_grgo / 10.0 );
    // Debug
    if ( l_n_go < 0 || fabs ( diff_grgo ) > 220.0 || gr_ax_comp > 577 ) { printf ( "error in grgo_initialize\n" ); }
    
    d_comp [ pre_comp  * num_grgo + i ] = gr_ax_comp + GR_COMP * l_n_gr;
    d_comp [ post_comp * num_grgo + i ] = 0 + GO_COMP * l_n_go;  // soma

    d_elem [ grgo_ampa   * num_grgo + i ] = 0.0;
    d_elem [ grgo_nmda1  * num_grgo + i ] = 0.0;
    d_elem [ grgo_nmda2  * num_grgo + i ] = 0.0;
    d_elem [ grgo_weight * num_grgo + i ] = W_GRGO / ( d_num_syn_go [ l_n_go ] * 1.0/4.0 );//W_GRGO / ( l_num_syn * 1.0 );
    d_elem [ grgo_val    * num_grgo + i ] = 0.0;
    d_elem [ grgo_old_v  * num_grgo + i ] = 1000.0;
    }
  }
}

__host__ 
synapse_t *grgo_create ( const int nx_gr, const int ny_gr, const int nx_go, const int ny_go )
{
  int n_gr = nx_gr * ny_gr;
  int n_go = nx_go * ny_go;
  int max_n_grgo = ny_go * n_gr;

  synapse_t *d_grgo = ( synapse_t * ) malloc ( sizeof ( synapse_t ) );  

  if ( n_go * n_gr == 0 ) 
  { 
    d_grgo -> n = n_go * n_gr;  
    printf ( "# of grgo = 0\n" ); 
    return d_grgo; 
  }
  
  int *label_grgo = ( int * ) malloc ( max_n_grgo * sizeof ( int ) ); // GoC labels are connected by each GrC
  int *num_syn_gr = ( int * ) malloc ( n_gr       * sizeof ( int ) ); // # of synapses from each GrC
  int *num_syn_go = ( int * ) malloc ( n_go       * sizeof ( int ) ); // # of synapses from each GoC
  reset_array ( label_grgo, max_n_grgo );
  reset_zero ( num_syn_gr, n_gr );   
  reset_zero ( num_syn_go, n_go );   
  
  int num_grgo = 0;
  for ( int i_gr = 0; i_gr < n_gr; i_gr++ ) 
  {
    double lx_gr = ( int ) ( i_gr % nx_gr ) * 16.0; // i_gr's x-coordinate
    double ly_gr = ( int ) ( i_gr / nx_gr ) * 16.0; // i_gr's y-coordinate
    int l_count = 0;

    for ( int i_go = 0; i_go < n_go; i_go++ ) 
    {
      double lx_go = ( int ) ( i_go % nx_go ) * 32.0 + 8.0; // i_go's x-coordinate
      double ly_go = ( int ) ( i_go / nx_go ) * 32.0 + 8.0; // i_go's y-coordinate

      if ( abs ( lx_go - lx_gr ) < 16.0 && abs ( ly_go - ly_gr ) < 220.0 )
      {
          label_grgo [ i_gr * ny_go + l_count ] = i_go;
          l_count++;
          num_syn_go [ i_go ]++;
      }
    }

    num_syn_gr [ i_gr ] = l_count;
    num_grgo += l_count;
    // Debug
    if ( l_count > ny_go ) { printf ( "Error in grgo_create\n" ); exit ( 1 ); }
  }

  d_grgo -> n = num_grgo;  
  cudaMalloc ( ( int    ** ) & ( d_grgo -> comp ), syn_n_comp  * num_grgo * sizeof ( int    ) );  
  cudaMalloc ( ( double ** ) & ( d_grgo -> elem ), grgo_n_elem * num_grgo * sizeof ( double ) );   
  printf ( "# of grgo = %d\n", d_grgo -> n );

  // Copy host array to device array
  int *d_label_grgo;
  int *d_num_syn_gr;
  int *d_num_syn_go;
  cudaMalloc ( ( int ** ) & ( d_label_grgo ), max_n_grgo * sizeof ( int ) );  
  cudaMalloc ( ( int ** ) & ( d_num_syn_gr ), n_gr       * sizeof ( int ) );  
  cudaMalloc ( ( int ** ) & ( d_num_syn_go ), n_go       * sizeof ( int ) );  
  cudaMemcpy ( d_label_grgo, label_grgo, max_n_grgo * sizeof ( int ), cudaMemcpyHostToDevice );  
  cudaMemcpy ( d_num_syn_gr, num_syn_gr, n_gr * sizeof ( int ),       cudaMemcpyHostToDevice );
  cudaMemcpy ( d_num_syn_go, num_syn_go, n_go * sizeof ( int ),       cudaMemcpyHostToDevice );


  grgo_initialize <<< ( ( n_gr ) + 127 ) / 128, 128 >>> 
    ( d_grgo -> comp, d_grgo -> elem, n_gr, nx_gr, ny_gr, nx_go, ny_go, 
      num_grgo, d_label_grgo, d_num_syn_gr, d_num_syn_go );
  
  // Debug
  //printf ("\nDebug for grgo");
  //printf_GPU <<< 1, 1 >>> ( d_grgo -> comp, d_grgo -> elem, num_grgo, syn_n_comp, grgo_n_elem );
  //cudaDeviceSynchronize();
  
  free ( label_grgo );  free ( num_syn_gr );  free ( num_syn_go );
  cudaFree ( d_label_grgo );  cudaFree ( d_num_syn_gr );  cudaFree ( d_num_syn_go );
  
  return d_grgo;
}

__global__ static
void grgo_update ( int *grgo_comp, double *grgo_elem, const int num_grgo, neuron_t *d_gr ) 
{  
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < num_grgo )
  {
    int firing_flag = 0;
    int pre_num = grgo_comp [ pre_comp * num_grgo + id ];
    double pre_comp_v = d_gr -> elem [ v ] [ pre_num ];
    if ( pre_comp_v > 0.0 && grgo_elem [ grgo_old_v * num_grgo + id ] < 0.0 )
    {
      firing_flag = 1;
    }
    // Decay = exp(-0.125 / tau ) = 0.92004441462 //tau = 1.5  = exp(-0.25 / tau ) = 0.84648172489
    grgo_elem [ grgo_ampa  * num_grgo + id ] = grgo_elem [ grgo_ampa  * num_grgo + id ] * D_GRGO_AMPA + firing_flag;
    // Decay = exp(-0.125 / tau ) = 0.99597586057 //tau = 31   = exp(-0.25 / tau ) = 0.99196791484
    grgo_elem [ grgo_nmda1 * num_grgo + id ] = grgo_elem [ grgo_nmda1 * num_grgo + id ] * D_GRGO_NMDA1 + firing_flag;
    // Decay = exp(-0.125 / tau ) = 0.99926497614 //tau = 170  = exp(-0.25 / tau ) = 0.99853049255
    grgo_elem [ grgo_nmda2 * num_grgo + id ] = grgo_elem [ grgo_nmda2 * num_grgo + id ] * D_GRGO_NMDA2 + firing_flag;
    grgo_elem [ grgo_val   * num_grgo + id ] = grgo_elem [ grgo_weight * num_grgo + id ] * 
                                               ( G_GRGO_AMPA * grgo_elem [ grgo_ampa  * num_grgo + id ] 
                                               + G_GRGO_NMDA * ( 0.33 * grgo_elem [ grgo_nmda1 * num_grgo + id ] 
                                                               + 0.67 * grgo_elem [ grgo_nmda2 * num_grgo + id ] ) );
    grgo_elem [ grgo_old_v * num_grgo + id ]  = pre_comp_v;
  }
}

__host__ 
void grgo_finalize ( synapse_t *d_grgo, const int n_grgo )
{
  if ( n_grgo > 0 )
  {
    cudaFree ( d_grgo -> comp );
    cudaFree ( d_grgo -> elem );
    //fclose   ( d_grgo -> f_out );
  }
  free ( d_grgo );
}


//////////////////////////////// GOGR //////////////////////////////

__global__ static 
void gogr_initialize3 ( int *d_comp, double *d_elem, const int n_go,  const int n_gr,
  const int num_gogr, const int *d_label_gogr, const int *d_num_syn_go, const int *d_num_syn_gr )
{    
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < n_go )
  {
    int l_num_syn = d_num_syn_go [ id ]; // # of synapses from id-th GoC
    int s_syn_id = 0;                 // # of synapses from GoCs 0 to (id-1)
    for ( int j = 0; j < id; j++ ) { s_syn_id += d_num_syn_go [ j ]; }

    for ( int i = s_syn_id; i < s_syn_id + l_num_syn; i++ )
    {
    int l_n_go = id;
    int l_n_gr = d_label_gogr [ l_n_go * 4 * 4 + i - s_syn_id ];
    // Debug
    if ( l_n_gr < 0 ) { printf ( "error in gogr_initialize\n" ); }

    d_comp [ pre_comp  * num_gogr + i ] = ( GO_COMP_DEND * 3 ) + GO_COMP * l_n_go;
    d_comp [ post_comp * num_gogr + i ] = 16 + GR_COMP * l_n_gr;  // soma
      
    d_elem [ gogr_gabaa  * num_gogr + i ] = 0.0;
    d_elem [ gogr_gabab  * num_gogr + i ] = 0.0;
    d_elem [ gogr_weight * num_gogr + i ] = W_GOGR / ( d_num_syn_gr [ l_n_gr ] * 4.0 );
    d_elem [ gogr_val    * num_gogr + i ] = 0.0;
    d_elem [ gogr_old_v  * num_gogr + i ] = 1000.0;
    }
  }
}

__host__ 
synapse_t *gogr_create ( const int nx_go, const int ny_go, const int nx_gr, const int ny_gr )
{
  int n_go = nx_go * ny_go;
  int n_gr = nx_gr * ny_gr;
  int max_n_gogr = n_go * 4 * 4;

  synapse_t *d_gogr = ( synapse_t * ) malloc ( sizeof ( synapse_t ) );  

  if ( n_go * n_gr == 0 ) 
  { 
    d_gogr -> n = n_go * n_gr;  
    printf ( "# of gogr = 0\n" ); 
    return d_gogr; 
  }
  
  int *label_gogr = ( int * ) malloc ( max_n_gogr * sizeof ( int ) ); // GrC labels are connected by each GoC
  int *num_syn_go    = ( int * ) malloc ( n_go       * sizeof ( int ) ); // # of synapses from each GoC
  int *num_syn_gr    = ( int * ) malloc ( n_gr       * sizeof ( int ) ); // # of synapses from each GoC
  reset_array ( label_gogr, max_n_gogr );
  reset_zero ( num_syn_go, n_go );   
  reset_zero ( num_syn_gr, n_gr );   
  
  int num_gogr = 0;
  for ( int i_go = 0; i_go < n_go; i_go++ ) 
  {
    int l_count = 0;
    double lx_go = ( int ) ( i_go % nx_go ) * 32.0 + 8.0; // i_go's x-coordinate
    double ly_go = ( int ) ( i_go / nx_go ) * 32.0 + 8.0; // i_go's y-coordinate

    for ( int i_gr = 0; i_gr < n_gr; i_gr++ ) 
    {
      double lx_gr = ( int ) ( i_gr % nx_gr ) * 16.0; // i_gr's x-coordinate
      double ly_gr = ( int ) ( i_gr / nx_gr ) * 16.0; // i_gr's y-coordinate

      if ( fabs ( lx_go - lx_gr ) < 32.0 && fabs ( ly_go - ly_gr ) < 32.0 )
      {     
          label_gogr [ i_go * 4 * 4 + l_count ] = i_gr;
          l_count++;
          num_syn_gr [ i_gr ]++;
      }
    }
    num_syn_go [ i_go ] = l_count;
    num_gogr += l_count;
    // Debug
    if ( l_count > 4 * n_gr ) { printf ( "Error in gogr_create\n" ); exit ( 1 ); }
  }
  
  d_gogr -> n = num_gogr;  
  cudaMalloc ( ( int    ** ) & ( d_gogr -> comp ), syn_n_comp  * num_gogr * sizeof ( int    ) );  
  cudaMalloc ( ( double ** ) & ( d_gogr -> elem ), gogr_n_elem * num_gogr * sizeof ( double ) );   
  printf ( "# of gogr = %d\n", d_gogr -> n );

  // Copy host array to device array
  int *d_label_gogr;
  int *d_num_syn_go, *d_num_syn_gr;
  cudaMalloc ( ( int ** ) & ( d_label_gogr ), max_n_gogr * sizeof ( int ) );  
  cudaMalloc ( ( int ** ) & ( d_num_syn_go ),    n_go       * sizeof ( int ) );  
  cudaMalloc ( ( int ** ) & ( d_num_syn_gr ),    n_gr       * sizeof ( int ) );  
  cudaMemcpy ( d_label_gogr, label_gogr, max_n_gogr * sizeof ( int ), cudaMemcpyHostToDevice );  
  cudaMemcpy ( d_num_syn_go, num_syn_go, n_go * sizeof ( int ),       cudaMemcpyHostToDevice );
  cudaMemcpy ( d_num_syn_gr, num_syn_gr, n_gr * sizeof ( int ),       cudaMemcpyHostToDevice );
  
  gogr_initialize3 <<< ( ( n_go ) + 127 ) / 128, 128 >>> 
    ( d_gogr -> comp, d_gogr -> elem, n_go, n_gr, num_gogr, d_label_gogr, d_num_syn_go, d_num_syn_gr );
   
  free ( label_gogr );  free ( num_syn_go );  free ( num_syn_gr );
  cudaFree ( d_label_gogr );  cudaFree ( d_num_syn_go );  cudaFree ( d_num_syn_gr );
  
  // Debug
  //printf ("\nDebug for gogr");
  //printf_GPU <<< 1, 1 >>> ( d_gogr -> comp, d_gogr -> elem, num_gogr, syn_n_comp, gogr_n_elem );
  //cudaDeviceSynchronize();

  return d_gogr;
}

__global__ static
void gogr_update ( int *gogr_comp, double *gogr_elem, const int num_gogr, neuron_t *d_go ) 
{  
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < num_gogr )
  {
    int firing_flag = 0;
    int pre_num = gogr_comp [ pre_comp * num_gogr + id ];
    double pre_comp_v = d_go -> elem [ v ] [ pre_num ];
    if ( pre_comp_v > 0.0 && gogr_elem [ gogr_old_v * num_gogr + id ] < 0.0 )
    {
      firing_flag = 1;
    }

    // Decay = exp(-0.125 / tau ) = 0.9823013511  //tau = 7  = exp(-0.25 / tau ) = 0.96491594437
    gogr_elem [ gogr_gabaa * num_gogr + id ] = gogr_elem [ gogr_gabaa * num_gogr + id ] * D_GOGR_GABAA  + firing_flag;
    // Decay = exp(-0.125 / tau ) = 0.99788359867 //tau = 59  = exp(-0.25 / tau ) = 0.9957716765
    gogr_elem [ gogr_gabab * num_gogr + id ] = gogr_elem [ gogr_gabab * num_gogr + id ] * D_GOGR_GABAB + firing_flag;
    gogr_elem [ gogr_val   * num_gogr + id ] = gogr_elem [ gogr_weight * num_gogr + id ] * G_GOGR_GABA
                                           * ( 0.43 * gogr_elem [ gogr_gabaa * num_gogr + id ] 
                                             + 0.57 * gogr_elem [ gogr_gabab * num_gogr + id ] );
    gogr_elem [ gogr_old_v * num_gogr + id ]  = pre_comp_v;
    //int l_comp = gogr_comp [ post_comp * num_gogr + id ];
    //d_gr -> elem [ g_syn ] [ l_comp ] += l_val; // +=大丈夫
  }
}

__host__ 
void gogr_finalize ( synapse_t *d_gogr, const int n_gogr )
{
  if ( n_gogr > 0 )
  {
    cudaFree ( d_gogr -> comp );
    cudaFree ( d_gogr -> elem );
    //cudaFree ( d_gogr -> cstate );
    //fclose   ( d_gogr -> f_out );
  }
  free ( d_gogr );
}


//////////////////////////////// GRPKJ //////////////////////////////

__global__ static 
void grpkj_initialize ( int *d_comp, double *d_elem,
                      const int nx_gr, const int ny_gr, const int nx_pkj, const int ny_pkj, 
                      const int num_grpkj, const int *d_label_grpkj, const int *d_num_syn_gr,
                      const double *d_x, const double *d_z, curandState *S, const int *d_num_syn_pkj )
{    
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int n_gr = nx_gr * ny_gr;
  int n_pkj = nx_pkj * ny_pkj;
  if ( id < n_gr )
  {
    int l_num_syn = d_num_syn_gr [ id ]; // # of synapses from id-th GrC
    int s_syn_id = 0;                 // # of synapses from GrCs 0 to (id-1)
    for ( int j = 0; j < id; j++ ) { s_syn_id += d_num_syn_gr [ j ]; }
    
    for ( int i = s_syn_id; i < s_syn_id + l_num_syn; i++ )
    {
      int l_n_gr = id;
      int l_n_pkj = d_label_grpkj [ l_n_gr * n_pkj + i - s_syn_id ];

      int ix_gr = l_n_gr % nx_gr; int ix_pkj = l_n_pkj % nx_pkj;
      int iy_gr = l_n_gr / nx_gr; int iy_pkj = l_n_pkj / nx_pkj;

      double diff_x = ix_gr * 16.0 - ( 8.0 + ix_pkj * 32.0 ); // x Distance between GrC and GoC somas
      double diff_y = iy_gr * 16.0 - ( 8.0 + iy_pkj * 32.0 ); // y Distance between GrC and GoC somas

      int gr_ax_comp = 250 + 77 + ( int ) ( diff_y / 10.0 );
      int pkj_d_comp = -1; double min_x = 1000000.0;

      int guarantee = 1;
      //if ( ix_pkj < 3 || ( nx_pkj - ix_pkj ) < 4 ) { guarantee = 3; }
      //if ( iy_pkj < 3 || ( ny_pkj - iy_pkj ) < 4 ) { guarantee = 3; }


      for ( int i_cp = 0; i_cp < N_Syn_Per_GRPKJ; i_cp++ )
      {
        double f = curand_uniform ( & ( S [ i * N_Syn_Per_GRPKJ + i_cp ] ) ) - 0.5;
        for ( int i_comp = 0; i_comp < PKJ_COMP; i_comp++ )
        {
          if ( fabs ( d_x [ i_comp ] - diff_x + 10.0 * f ) < min_x )
          {
            min_x = fabs ( d_x [ i_comp ] - diff_x + 10.0 * f );
            pkj_d_comp = i_comp;
          } 
        }
        // Debug
        if ( pkj_d_comp < 0 || l_n_pkj < 0 || fabs ( diff_y ) > 220.0 || gr_ax_comp > 577 ) { printf ( "error in grpkj_initialize\n" ); }
        //if ( id < 2 )  printf ("%d -> %d \n", gr_ax_comp + GR_COMP  * l_n_gr, pkj_d_comp + PKJ_COMP * l_n_pkj );
    
        d_comp [ pre_comp  * num_grpkj + i * N_Syn_Per_GRPKJ + i_cp ] = gr_ax_comp + GR_COMP  * l_n_gr;
        d_comp [ post_comp * num_grpkj + i * N_Syn_Per_GRPKJ + i_cp ] = pkj_d_comp + PKJ_COMP * l_n_pkj;
         
        //if ( id < 2 )  printf ("%d -> %d \n",
        //  d_comp [ pre_comp  * num_grpkj + i * N_Syn_Per_GRPKJ + i_cp ],
        //  d_comp [ post_comp * num_grpkj + i * N_Syn_Per_GRPKJ + i_cp ] );
        d_elem [ grpkj_ampa   * num_grpkj + i * N_Syn_Per_GRPKJ + i_cp ] = 0.0;
        d_elem [ grpkj_weight * num_grpkj + i * N_Syn_Per_GRPKJ + i_cp ] = W_GRPKJ / ( 512.0 * d_num_syn_pkj [ l_n_pkj ] * 1.0 * guarantee / 10.0 );
        d_elem [ grpkj_val    * num_grpkj + i * N_Syn_Per_GRPKJ + i_cp ] = 0.0;
        d_elem [ grpkj_old_v  * num_grpkj + i * N_Syn_Per_GRPKJ + i_cp ] = 1000.0;
      }
    }
  }
}

__host__ 
synapse_t *grpkj_create ( const int nx_gr, const int ny_gr, const int nx_pkj, const int ny_pkj )
{
  int n_gr  = nx_gr  * ny_gr;
  int n_pkj = nx_pkj * ny_pkj;
  int max_n_grpkj = n_pkj * n_gr;

  synapse_t *d_grpkj = ( synapse_t * ) malloc ( sizeof ( synapse_t ) );

  if ( n_pkj * n_gr == 0 ) 
  { 
    d_grpkj -> n = 0;  
    printf ( "# of grpkj = 0\n" ); 
    return d_grpkj; 
  }

  // read PKJ_compartments location
  FILE *f_ = fopen ( "Pkj_location_info.csv", "r" );
  if ( ! f_ ) { fprintf ( stderr, "no such file %s\n", "Pkj_location_info.csv" ); exit ( 1 ); }
  int type [ PKJ_COMP ];
  double x [ PKJ_COMP ]; double z [ PKJ_COMP ];
  int i1, i5; double i2, i3, i4; 
  for ( int i = 0; i < PKJ_COMP; i++ ) 
  {
    if ( fscanf ( f_, "%d,%lf,%lf,%lf,%d", &i1, &i2, &i3, &i4, &i5 ) == ( EOF ) ) {
        printf ( "PARAM_FILE_READING_ERROR\n" );
        exit ( 1 );
    }
    //l_comp [ i ] = i1;
    x [ i ] = i2;
    //y [ i ] = i3;
    z [ i ] = i4;
    type [ i ] = i5;
  }
  fclose ( f_ );

  double *d_x, *d_z;
  cudaMalloc ( ( double ** ) & ( d_x ), PKJ_COMP * sizeof ( double ) );  
  cudaMalloc ( ( double ** ) & ( d_z ), PKJ_COMP * sizeof ( double ) );  
  cudaMemcpy ( d_x, x, PKJ_COMP * sizeof ( double ), cudaMemcpyHostToDevice );  
  cudaMemcpy ( d_z, z, PKJ_COMP * sizeof ( double ), cudaMemcpyHostToDevice );

  int *label_grpkj = ( int * ) malloc ( max_n_grpkj * sizeof ( int ) ); // PC labels are connected by each GrC
  int *num_syn_gr  = ( int * ) malloc ( n_gr        * sizeof ( int ) ); // # of synapses from each GrC
  int *num_syn_pkj = ( int * ) malloc ( n_pkj       * sizeof ( int ) ); // # of synapses from each PC
  reset_array ( label_grpkj, max_n_grpkj );
  reset_zero ( num_syn_gr,  n_gr  );   
  reset_zero ( num_syn_pkj, n_pkj );   
  
  int num_grpkj = 0;
  for ( int i_gr = 0; i_gr < n_gr; i_gr++ ) 
  {
    double lx_gr = ( int ) ( i_gr % nx_gr ) * 16.0; // i_gr's x-coordinate
    double ly_gr = ( int ) ( i_gr / nx_gr ) * 16.0; // i_gr's y-coordinate

    int l_count = 0;
    for ( int i_pkj = 0; i_pkj < n_pkj; i_pkj++ ) 
    {
      double lx_pkj = ( int ) ( i_pkj % nx_pkj ) * 32.0 + 8.0; // i_pkj's x-coordinate
      double ly_pkj = ( int ) ( i_pkj / nx_pkj ) * 32.0 + 8.0; // i_pkj's y-coordinate

      if ( abs ( lx_pkj - lx_gr ) < 120.0 && abs ( ly_pkj - ly_gr ) < 220.0 )
      {
        label_grpkj [ i_gr * n_pkj + l_count ] = i_pkj;
        l_count ++;
        num_syn_pkj [ i_pkj ]++;
      }
    }
    num_syn_gr [ i_gr ] = l_count;
    num_grpkj += l_count * N_Syn_Per_GRPKJ;
    // Debug
    if ( l_count / N_Syn_Per_GRPKJ > n_pkj ) { printf ( "Error in grpkj_create\n" ); exit ( 1 ); }
    //printf ("num_grpkj -> %d\n", num_grpkj);
  }

  //Debug
  int l_max = -1; int l_min = 100000;
  for ( int i_pkj = 0; i_pkj < n_pkj; i_pkj++ ) {
    if ( l_max < num_syn_pkj [ i_pkj ] ){
      l_max = num_syn_pkj [i_pkj];
    }
    if ( l_min > num_syn_pkj [ i_pkj ] ){
      l_min = num_syn_pkj [i_pkj];
    }
  }
  printf ("max grpkj per cell = %d, min = %d\n", l_max, l_min);

  d_grpkj -> n = num_grpkj;  
  cudaMalloc ( ( int    ** ) & ( d_grpkj -> comp ), syn_n_comp   * num_grpkj * sizeof ( int    ) );  
  cudaMalloc ( ( double ** ) & ( d_grpkj -> elem ), grpkj_n_elem * num_grpkj * sizeof ( double ) );   
  printf ( "# of grpkj = %d\n", d_grpkj -> n );

  // Copy host array to device array
  int *d_label_grpkj;
  int *d_num_syn_gr, *d_num_syn_pkj;
  cudaMalloc ( ( int ** ) & ( d_label_grpkj ), max_n_grpkj * sizeof ( int ) );  
  cudaMalloc ( ( int ** ) & ( d_num_syn_gr ),  n_gr        * sizeof ( int ) );  
  cudaMalloc ( ( int ** ) & ( d_num_syn_pkj ), n_pkj       * sizeof ( int ) );  
  cudaMemcpy ( d_label_grpkj, label_grpkj, max_n_grpkj * sizeof ( int ), cudaMemcpyHostToDevice );  
  cudaMemcpy ( d_num_syn_gr,   num_syn_gr,   n_gr  * sizeof ( int ),        cudaMemcpyHostToDevice );
  cudaMemcpy ( d_num_syn_pkj,  num_syn_pkj,  n_pkj * sizeof ( int ),        cudaMemcpyHostToDevice );

  curandState *l_state;  
  cudaMalloc ( ( void ** ) &( l_state ), num_grpkj * sizeof ( curandState ) );
  setCurand <<< ( num_grpkj + 127 ) / 128, 128 >>>  ( rand (), l_state, num_grpkj );

  grpkj_initialize <<< ( ( n_gr ) + 127 ) / 128, 128 >>> 
    ( d_grpkj -> comp, d_grpkj -> elem, nx_gr, ny_gr, nx_pkj, ny_pkj, 
      num_grpkj, d_label_grpkj, d_num_syn_gr, d_x, d_z, l_state, d_num_syn_pkj );
  cudaDeviceSynchronize();
  // Debug
  //printf ("\nDebug for grpkj\n");
  //printf_GPU <<< 1, 1 >>> ( d_grpkj -> comp, d_grpkj -> elem, num_grpkj, syn_n_comp, grpkj_n_elem );
  //cudaDeviceSynchronize();
  
  free ( label_grpkj );  
  free ( num_syn_gr );
  free ( num_syn_pkj );
  cudaFree ( d_label_grpkj );
  cudaFree ( d_num_syn_gr );
  cudaFree ( d_num_syn_pkj );
  cudaFree ( d_x );  
  cudaFree ( d_z );
  
  return d_grpkj;
}

__global__ static
void grpkj_update ( int *grpkj_comp, double *grpkj_elem, const int num_grpkj, neuron_t *d_gr ) 
{  
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < num_grpkj )
  {
    int firing_flag = 0;
    int pre_num = grpkj_comp [ pre_comp * num_grpkj + id ];
    double pre_comp_v = d_gr -> elem [ v ] [ pre_num ];
    if ( pre_comp_v > 0.0 && grpkj_elem [ grpkj_old_v * num_grpkj + id ] < 0.0 )
    {
      firing_flag = 1;
    }
    // Decay = exp(-0.125 / tau ) = 0.98505259729  //tau = 8.3 
    //grpkj_elem [ grpkj_ampa  * num_grpkj + id ] = grpkj_elem [ grpkj_ampa * num_grpkj + id ] * 0.98505259729  + firing_flag;
    // Decay = exp(-0.125 / tau ) = 0.81193634615  //tau = 0.6  = exp(-0.25 / tau ) = 0.6592406302
    grpkj_elem [ grpkj_ampa  * num_grpkj + id ] = grpkj_elem [ grpkj_ampa * num_grpkj + id ] * D_GRPKJ_AMPA + firing_flag;// tau = 0.6 (Llano et al., 1991) 
    grpkj_elem [ grpkj_val   * num_grpkj + id ] = grpkj_elem [ grpkj_ampa * num_grpkj + id ] * grpkj_elem [ grpkj_weight * num_grpkj + id ] * G_GRPKJ_AMPA;
    grpkj_elem [ grpkj_old_v * num_grpkj + id ] = pre_comp_v;
    // Debug
    //printf ("ampa = %f, val = %f\n",grpkj_elem [ grpkj_ampa  * num_grpkj + id ],grpkj_elem [ grpkj_weight * num_grpkj + id ]);
  }
}

__host__ 
void grpkj_finalize ( synapse_t *d_grpkj, const int n_grpkj )
{
  if ( n_grpkj > 0 )
  {
    cudaFree ( d_grpkj -> comp );
    cudaFree ( d_grpkj -> elem );
  }
  free ( d_grpkj );
}

//////////////////////////////// MLIPKJ //////////////////////////////
__global__ static 
void mlipkj_initialize ( int *d_comp, double *d_elem, const int num_mlipkj, 
                         const int n_pkj, const int *d_postcomp_mlipkj )
{    
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < n_pkj )
  {
    for( int j = 0; j < N_Syn_Per_MLIPKJ - 1; j++ )
    {
      d_comp [ pre_comp  * num_mlipkj + id * N_Syn_Per_MLIPKJ + j ] = id; // not use
      d_comp [ post_comp * num_mlipkj + id * N_Syn_Per_MLIPKJ + j ] = d_postcomp_mlipkj [ j ] + id * PKJ_COMP;
      d_elem [ mlipkj_gaba   * num_mlipkj +  id * N_Syn_Per_MLIPKJ + j ] = 0.0;
      d_elem [ mlipkj_weight * num_mlipkj +  id * N_Syn_Per_MLIPKJ + j ] = W_MLIPKJ / ( N_Syn_Per_MLIPKJ * 1.0);
      d_elem [ mlipkj_val    * num_mlipkj +  id * N_Syn_Per_MLIPKJ + j ] = 0.0;
    }
    int j = N_Syn_Per_MLIPKJ - 1;
    d_comp [ pre_comp  * num_mlipkj + id * N_Syn_Per_MLIPKJ + j ] = id; // not use
    d_comp [ post_comp * num_mlipkj + id * N_Syn_Per_MLIPKJ + j ] = d_postcomp_mlipkj [ j ] + id * PKJ_COMP;
    d_elem [ mlipkj_gaba   * num_mlipkj +  id * N_Syn_Per_MLIPKJ + j ] = 0.0;
    d_elem [ mlipkj_weight * num_mlipkj +  id * N_Syn_Per_MLIPKJ + j ] = W_MLIPKJ / ( 1.0 );
    d_elem [ mlipkj_val    * num_mlipkj +  id * N_Syn_Per_MLIPKJ + j ] = 0.0;
  }
}

__host__ 
synapse_t *mlipkj_create ( const int n_pkj, const int n_gr )
{
  int num_mlipkj = n_pkj * N_Syn_Per_MLIPKJ;
  synapse_t *d_mlipkj = ( synapse_t * ) malloc ( sizeof ( synapse_t ) );

  if ( n_gr == 0 || n_pkj == 0 ) 
  { 
    d_mlipkj -> n = 0; 
    printf ( "# of mlipkj = 0\n" ); 
    return d_mlipkj; 
  }
  else 
  { 
    d_mlipkj -> n = num_mlipkj;  
    printf ( "# of mlipkj = %d\n", num_mlipkj ); 
  }

  cudaMalloc ( ( int    ** ) & ( d_mlipkj -> comp ), syn_n_comp    * num_mlipkj * sizeof ( int    ) );  
  cudaMalloc ( ( double ** ) & ( d_mlipkj -> elem ), mlipkj_n_elem * num_mlipkj * sizeof ( double ) );    
  d_mlipkj -> f_out = fopen ( "MLI_RASTER.csv", "w" );

  int *h_postcomp_mlipkj = ( int * ) malloc ( N_Syn_Per_MLIPKJ * sizeof ( int ) );
  int *d_postcomp_mlipkj;
  cudaMalloc ( ( int ** ) & ( d_postcomp_mlipkj ), N_Syn_Per_MLIPKJ * sizeof ( int ) );
  reset_array ( h_postcomp_mlipkj, N_Syn_Per_MLIPKJ );

  for ( int i = 0; i < N_Syn_Per_MLIPKJ - 1; i++ )
  {
    while ( 1 )
    {
      int r = rand () % ( PKJ_COMP - 10 ); // pkj comps : 1590 ~ 1599 -> main dends or soma
      for ( int j = 0; j < i; j++ ) { if ( h_postcomp_mlipkj [ j ] == r ) { continue; } } 
      h_postcomp_mlipkj [ i ] = r;
      break;
    }
  }   
  h_postcomp_mlipkj [ N_Syn_Per_MLIPKJ - 1 ] = 1599;

  // Debug
  //printf ("Debug for mlipkj");
  //for ( int i = 0; i < N_Syn_Per_MLIPKJ; i++ ) { printf ("%d, ", h_postcomp_mlipkj [ i ]); }
  //printf ("\n");

  cudaMemcpy ( d_postcomp_mlipkj, h_postcomp_mlipkj, N_Syn_Per_MLIPKJ * sizeof ( int ), cudaMemcpyHostToDevice );
  mlipkj_initialize <<< ( ( n_pkj ) + 127 ) / 128, 128 >>> ( d_mlipkj -> comp, d_mlipkj -> elem, num_mlipkj, n_pkj, d_postcomp_mlipkj );
 
  // Set rand
  cudaMalloc ( ( void ** ) &( d_mlipkj -> cstate ), num_mlipkj * sizeof ( curandState ) );
  setCurand <<< ( num_mlipkj + 127 ) / 128, 128 >>>  ( rand (), d_mlipkj -> cstate, num_mlipkj );
  
  free ( h_postcomp_mlipkj );  cudaFree ( d_postcomp_mlipkj );
  return d_mlipkj;
}

__global__ 
void mlipkj_update ( int *mlipkj_comp, double *mlipkj_elem, const double t, 
                     const int num_mlipkj, curandState *S ) 
{  
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < num_mlipkj )
  {
    int firing_flag;
    double fr_mli = 30.0;
    //( S_Stimuli <= t && t < E_Stimuli + 200.0 )? fr_mli = 150.0 : fr_mli = 30.0;
    //( 0 <= t && t < 500.0 )? fr_mli = 0.0 : 
    ( S_Stimuli + 50.0 <= t && t < E_Stimuli + 50.0 )? fr_mli = 30.0:
    //( E_Stimuli <= t && t < E_Stimuli + 50.0 )? fr_mli = 80.0:
    fr_mli = 30.0;
    
    //( S_Stimuli + 100.0 <= t && t < E_Stimuli )? fr_mli = 50.0:
    //( E_Stimuli <= t && t < E_Stimuli + 100.0 )? fr_mli = 150.0:
    

    double f = curand_uniform ( & ( S [ id ] ) );
    ( fr_mli * 0.125 * 0.001 > f )? firing_flag = 1 : firing_flag = 0;
    
    // Decay :exp( - DT / tau ) = 0.98757780049 ( DT = 0.125 )//tau = 10.0
    mlipkj_elem [ mlipkj_gaba * num_mlipkj + id ] = mlipkj_elem [ mlipkj_gaba * num_mlipkj + id ] * 0.98757780049 + firing_flag;
    mlipkj_elem [ mlipkj_val  * num_mlipkj + id ] = mlipkj_elem [ mlipkj_weight * num_mlipkj + id ]  
                                           * ( G_MLIPKJ_GABA * mlipkj_elem [ mlipkj_gaba * num_mlipkj + id ] ); 
  }
}

__host__ 
void mlipkj_finalize ( synapse_t *d_mlipkj , const int n )
{
  if ( n > 0 )
  {
    cudaFree ( d_mlipkj -> comp );
    cudaFree ( d_mlipkj -> elem );
    cudaFree ( d_mlipkj -> cstate );
    fclose   ( d_mlipkj -> f_out );
  }
  free ( d_mlipkj );
}

__host__
void mli_output_file ( synapse_t *d_mlipkj, const double t, neuron_t *p_pkj )
{
  FILE *f = d_mlipkj -> f_out;
  double *ret = ( double * ) malloc ( sizeof ( double ) * mlipkj_n_elem * d_mlipkj -> n );
  cudaMemcpy ( ret, d_mlipkj -> elem, mlipkj_n_elem * d_mlipkj -> n * sizeof ( double ), cudaMemcpyDeviceToHost );
  double val = 0.0;
  fprintf ( f, "%lf,", t );
  for ( int j = 0; j < d_mlipkj -> n; j++ ) {
    val = G_MLIPKJ_GABA * ret [ mlipkj_gaba * d_mlipkj -> n + j ];
    val *= ret [ mlipkj_weight * d_mlipkj -> n + j ] *1000000;
    fprintf ( f, "%lf,", val );
  }
  fprintf ( f, "\n" );
  free ( ret ); 
}
/////////////////////////////////////////////////////////////////////////

__host__
void gr_synapse_update ( const double t, const double DT, 
                         synapse_t *d_mfgr, synapse_t *d_gogr, 
                         neuron_t *d_go, neuron_solve_t *p_gr_solve )
{
  //static int count = 0;
  
  if ( 0 == strncmp ( p_gr_solve -> type, "BE", 2 ) ) 
  { 
    //if ( count % 5 == 0 )
    //{
      mfgr_update <<< ( d_mfgr -> n + 127 ) / 128, 128  >>> ( d_mfgr -> comp, d_mfgr -> elem, t, d_mfgr -> n, d_mfgr -> cstate );
      if ( d_gogr -> n > 0 ) { gogr_update <<< ( d_gogr -> n + 127 ) / 128, 128  >>> ( d_gogr -> comp, d_gogr -> elem,    d_gogr -> n, d_go ); }
    //  count = 0;
    //}
    //count++;
  }
  else if ( 0 == strncmp ( p_gr_solve -> type, "CN", 2 ) )
  { 
    //if ( count % 5 == 0 )
    //{
      mfgr_update <<< ( d_mfgr -> n + 127 ) / 128, 128  >>> ( d_mfgr -> comp, d_mfgr -> elem, t, d_mfgr -> n, d_mfgr -> cstate );
      if ( d_gogr -> n > 0 ) { gogr_update <<< ( d_gogr -> n + 127 ) / 128, 128  >>> ( d_gogr -> comp, d_gogr -> elem,    d_gogr -> n, d_go ); }
    //  count = 0;
    //}
    //count++;
  }
  else if ( 0 == strncmp ( p_gr_solve -> type, "RKC", 3 ) )
  { 
    mfgr_update <<< ( d_mfgr -> n + 127 ) / 128, 128  >>> ( d_mfgr -> comp, d_mfgr -> elem, t, d_mfgr -> n, d_mfgr -> cstate );
    if ( d_gogr -> n > 0 ) { gogr_update <<< ( d_gogr -> n + 127 ) / 128, 128  >>> ( d_gogr -> comp, d_gogr -> elem,    d_gogr -> n, d_go ); }
  }
}

__host__
void go_synapse_update ( const double t, const double DT, 
                         synapse_t *d_grgo, 
                         neuron_t *d_gr, neuron_solve_t *p_go_solve )
{
  //static int count = 0;
  
  if ( 0 == strncmp ( p_go_solve -> type, "BE", 2 ) ) 
  { 
    //if ( count % 5 == 0 )
    //{
      if ( d_grgo -> n > 0 ) { grgo_update <<< ( d_grgo -> n + 127 ) / 128, 128  >>> ( d_grgo -> comp, d_grgo -> elem, d_grgo -> n, d_gr ); }
    //  count = 0;
    //}
    //count++;
  }
  else if ( 0 == strncmp ( p_go_solve -> type, "CN", 2 ) )
  { 
    //if ( count % 5 == 0 )
    //{
      if ( d_grgo -> n > 0 ) { grgo_update <<< ( d_grgo -> n + 127 ) / 128, 128  >>> ( d_grgo -> comp, d_grgo -> elem, d_grgo -> n, d_gr ); }
    //  count = 0;
    //}
    //count++;
  }
  else if ( 0 == strncmp ( p_go_solve -> type, "RKC", 3 ) )
  { 
    if ( d_grgo -> n > 0 ) { grgo_update <<< ( d_grgo -> n + 127 ) / 128, 128  >>> ( d_grgo -> comp, d_grgo -> elem, d_grgo -> n, d_gr ); }
  }
}

__host__
void pkj_synapse_update ( const double t, const double DT, 
                          synapse_t *d_grpkj, synapse_t *d_mlipkj, 
                          neuron_t *d_gr, neuron_solve_t *p_pkj_solve )
{
  //static int count = 0;
  
  if ( 0 == strncmp ( p_pkj_solve -> type, "BE", 2 ) ) 
  { 
    //if ( count % 5 == 0 )
    //{
      if ( d_grpkj -> n > 0 ) { grpkj_update <<< ( d_grpkj -> n + 127 ) / 128, 128  >>> ( d_grpkj -> comp, d_grpkj -> elem, d_grpkj -> n, d_gr ); }
    //  count = 0;
    //}
    //count++;
  }
  else if ( 0 == strncmp ( p_pkj_solve -> type, "CN", 2 ) )
  { 
    //if ( count % 5 == 0 )
    //{
      if ( d_grpkj  -> n > 0 ) { grpkj_update <<< ( d_grpkj -> n + 127 ) / 128, 128  >>> ( d_grpkj -> comp, d_grpkj -> elem, d_grpkj -> n, d_gr ); }
      //if ( d_mlipkj -> n > 0 ) { mlipkj_update <<< ( d_mlipkj -> n + 127 ) / 128, 128  >>> ( d_mlipkj -> comp, d_mlipkj -> elem, t, d_mlipkj -> n, d_mlipkj -> cstate ); }
    //  count = 0;
    //}
    //count++;
  }
  else if ( 0 == strncmp ( p_pkj_solve -> type, "RKC", 3 ) )
  { 
    if ( d_grpkj -> n > 0 ) { grpkj_update <<< ( d_grpkj -> n + 127 ) / 128, 128  >>> ( d_grpkj -> comp, d_grpkj -> elem, d_grpkj -> n, d_gr ); }
    //if ( d_mlipkj -> n > 0 ) { mlipkj_update <<< ( d_mlipkj -> n + 127 ) / 128, 128  >>> ( d_mlipkj -> comp, d_mlipkj -> elem, t, d_mlipkj -> n, d_mlipkj -> cstate ); }
  }
}
