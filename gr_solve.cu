#include "gr_solve.cuh"
__host__ static 
neuron_solve_t* set_host_gr_solve ( const char *type, neuron_t *gr )
{
  neuron_solve_t *gr_solve =  ( neuron_solve_t * ) malloc ( sizeof ( neuron_solve_t ) );
  int n_gr = gr -> n;
  //int gr_x = gr -> nx;
  //int gr_y = gr -> ny;

  // Matrix  
  double *mat = ( double * ) malloc ( GR_COMP * GR_COMP * sizeof ( double ) );
  int *l_connect = ( int * ) malloc ( GR_COMP * sizeof ( int ) );
  for ( int i = 0; i < GR_COMP * GR_COMP; i++ ) { mat [ i ] = 0.0; }

  // !!!DUPLICATE CODE!!!
  double rad [ GR_COMP ], len [ GR_COMP ], Ra [ GR_COMP ];
  FILE *file = fopen ( PARAM_FILE_GR, "r" );
  if ( ! file ) { fprintf ( stderr, "no such file %s\n", PARAM_FILE_GR ); exit ( 1 ); }

  for ( int i = 0; i < GR_COMP; i++ ) {
    int i1, i2, i3;
    double d1, d2, d3, d4, i_l1, i_l2, i_l3, i_Na, i_KV, i_KA, i_KCa, i_KM, i_KIR, i_Ca;
    
    if ( fscanf ( file, "%d %d %lf %lf %lf %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf ",
    &i1, &i2, &d1, &d2, &d3, &d4, &i3, &i_l1, &i_l2, &i_l3, &i_Na, &i_KV, &i_KA, &i_KCa, &i_KM, &i_KIR, &i_Ca ) == ( EOF ) ){
        printf ( "PARAM_FILE_READING_ERROR\n" );
        exit ( 1 );
    }
  
    rad [ i1 ] = 0.5 * d1 * 1e-4; // [mum -> cm]
    len [ i1 ] = d2 * 1e-4; // [mum -> cm]
    Ra  [ i1 ] = d3 * 1e-3; // [kohm-cm]

    l_connect [ i ] = i2;
  }

  for ( int i = 0; i < GR_COMP; i++ ) {
    int d = l_connect [ i ];
    if ( d >= 0 ) {
      mat [ d + GR_COMP * i ] = ( 2.0 / ( ( Ra [ i ] * len [ i ] ) / ( rad [ i ] * rad [ i ] * M_PI ) + ( Ra [ d ] * len [ d ] ) / ( rad [ d ] * rad [ d ] * M_PI ) ) ); // [mS]
      mat [ i + GR_COMP * d ] = mat [ d + GR_COMP * i ]; // i*NGR -> set Rows, +d -> set Columns      
    }
  }

  // Debug
  //for ( int i = 0; i < GR_COMP; i++ ) { printf ("mat [%d] = %f\n", i, mat [i]); }
      
  // count number of nonzero elements NNZ
  int nnz = GR_COMP;
  for ( int i = 0; i < GR_COMP * GR_COMP; i++ ) { nnz += ( mat [ i ] > 0.0 ); } 
  gr_solve -> nnz = nnz * ( gr -> n );  printf ( "GR -> nnz = %d\n", gr_solve -> nnz );
  gr_solve -> val     = ( double * ) malloc ( gr_solve -> nnz          * sizeof ( double ) ); // nonzero and diagrnal elements array
  gr_solve -> val_ori = ( double * ) malloc ( gr_solve -> nnz          * sizeof ( double ) );
  gr_solve -> b       = ( double * ) malloc ( GR_COMP * gr -> n         * sizeof ( double ) ); // b value
  gr_solve -> dammy   = ( double * ) malloc ( GR_COMP * gr -> n         * sizeof ( double ) ); // calculation vec
  gr_solve -> col     = ( int *    ) malloc ( gr_solve -> nnz          * sizeof ( int ) ); //column number of val's elements
  gr_solve -> row     = ( int *    ) malloc ( ( GR_COMP * gr -> n + 1 ) * sizeof ( int ) ); //row index
  gr_solve -> dig     = ( int *    ) malloc ( GR_COMP * gr -> n         * sizeof ( int ) ); //dig index
  sprintf ( gr_solve -> type, "%s", type ); 
  
  double *val     = gr_solve -> val;
  double *val_ori = gr_solve -> val_ori;
  double *b       = gr_solve -> b;
  double *dammy   = gr_solve -> dammy;
  int *col  = gr_solve -> col; 
  int *row  = gr_solve -> row;
  int *dig  = gr_solve -> dig;
  
  for ( int i = 0; i < gr_solve -> nnz; i++ )     { val [ i ] = val_ori [ i ] = 0.0; col [ i ] = 0; }
  for ( int i = 0; i < GR_COMP * gr -> n + 1; i++) { row [ i ] = 0; }
  for ( int i = 0; i < GR_COMP * gr -> n; i++)     { dig [ i ] = 0; b [ i ] = 0; dammy [ i ] = 0.0; }
        
  // Create CSR
  int num_row = 0, num_col = 0, num_dig = 0, num = 0, count_row = 0;
  //int count_gj = 0;
  
  for ( int i = 0; i < GR_COMP * GR_COMP; i++ ) 
  {
    if ( mat [ i ] > 0.0 ) {
      val [ num ] = - mat [ i ];
      count_row++;
      col [ num ] = num_col;
      num++;
    }
    else if ( num_row == num_col ) 
    {
      val [ num ] = 0.0;
      count_row++;
      col [ num ] = num_col;
      num_dig = num;
      num++;
    }
    num_col++;
    if ( num_col == GR_COMP ) 
    {
      for (int j = num_row * GR_COMP; j < num_row * GR_COMP + GR_COMP; j++ ) 
      {
        if ( j != num_row * GR_COMP + num_row ) {
          val [ num_dig ] += mat [ j ];
          dig [ num_row ] = num_dig;
        }
      }
      num_col = 0;
      num_row++;
      row [ num_row ] = count_row;
    }
  }
  
  for ( int i = 1; i < gr -> n; i++ ) 
  {
    for ( int j = 0; j < nnz; j++ ) 
    {
      val [ j + nnz * i ] = val [ j ];
      //val_ori [ j + nnz * i ] = val [ j ];
      col [ j + nnz * i ] = col [ j ] + GR_COMP * i;
    }
    for ( int j = 0; j < GR_COMP; j++ ) 
    {
      row [ j + GR_COMP * i ] = row [ j ] + nnz * i;
      dig [ j + GR_COMP * i ] = dig [ j ] + nnz * i;
    }
  }
  row [ GR_COMP * gr -> n ] = gr_solve -> nnz;      
  
  for ( int i = 0; i < gr_solve -> nnz; i++ ) 
  {  
    if ( ( 0 == strncmp ( gr_solve -> type, "FORWARD_EULER", 13 ) ) ||
         ( 0 == strncmp ( gr_solve -> type, "RUNGE_KUTTA_4", 13 ) ) ||
         ( 0 == strncmp ( gr_solve -> type, "RKC", 3 ) ) ) { val [ i ] *= -1; }
    if ( 0 == strncmp ( gr_solve -> type, "CN", 2 ) ) { val [ i ] /= 2.0; }

    val_ori [ i ] = val [ i ];
  } // nnz is OK.
    
  free ( mat );
  free ( l_connect );
  
  return gr_solve;
}

__global__
static void device_mem_allocation ( const int nc, const int l_nnz, neuron_solve_t* d_gr_solve,
     double *d_val, double *d_val_ori, double *d_b, int *d_col, int *d_row, int *d_dig, double *d_dammy ) 
{
  d_gr_solve -> nnz     = l_nnz;
  d_gr_solve -> val     = d_val;
  d_gr_solve -> val_ori = d_val_ori;
  d_gr_solve -> b       = d_b;
  d_gr_solve -> col     = d_col;
  d_gr_solve -> row     = d_row;   // # of neurons
  d_gr_solve -> dig     = d_dig; // # of all compartments
  d_gr_solve -> dammy   = d_dammy;
  d_gr_solve -> numThreadsPerBlock = 128;
  d_gr_solve -> numBlocks = ( int ) ( nc / d_gr_solve -> numThreadsPerBlock ) + 1;
  //Debug
  //printf ( "From GPU \n n = %d, nc = %d\n", d_gr -> n, d_gr -> nc );
}

__global__
static void device_mem_allocation2 (const int n, double ** dev, double *ptr )
{
  dev [ n ] = ptr;
}

__global__ static
void device_mem_allocation3 ( neuron_solve_t* d_gr_solve, double **d_vec ) 
{
  d_gr_solve -> vec = d_vec;
}

neuron_solve_t *gr_solve_initialize ( neuron_solve_t *p_gr_solve, const char *type, neuron_t *gr, neuron_t *d_gr ) // tentatively, type is ignored
{
  neuron_solve_t *d_gr_solve;
  cudaMalloc ( ( neuron_solve_t **) &d_gr_solve, sizeof ( neuron_solve_t ) );
  if ( gr -> nc == 0 ) { return d_gr_solve; }

  neuron_solve_t *h_gr_solve = set_host_gr_solve ( type, gr );  
  
  int l_nnz = h_gr_solve -> nnz;
  double *d_val, *d_val_ori, *d_b, *d_dammy;
  cudaMalloc ( ( double ** ) &d_val,     l_nnz * sizeof ( double ) );
  cudaMalloc ( ( double ** ) &d_val_ori, l_nnz * sizeof ( double ) );
  cudaMalloc ( ( double ** ) &d_b,       gr -> nc * sizeof ( double ) );
  cudaMalloc ( ( double ** ) &d_dammy,   gr -> nc * sizeof ( double ) );
  int *d_col, *d_row, *d_dig;
  cudaMalloc ( ( int ** ) &d_col, l_nnz * sizeof ( int ) );
  cudaMalloc ( ( int ** ) &d_row, ( GR_COMP * gr -> n + 1 ) * sizeof ( int ) );
  cudaMalloc ( ( int ** ) &d_dig, ( GR_COMP * gr -> n     ) * sizeof ( int ) );
  
  p_gr_solve -> val     = d_val;
  p_gr_solve -> val_ori = d_val_ori;
  p_gr_solve -> b       = d_b;
  p_gr_solve -> dammy   = d_dammy;
  p_gr_solve -> col = d_col;
  p_gr_solve -> row = d_row;
  p_gr_solve -> dig = d_dig;
  p_gr_solve -> nnz = l_nnz;
  p_gr_solve -> numThreadsPerBlock = 128;
  p_gr_solve -> numBlocks = ( int ) ( ( gr -> nc ) / ( p_gr_solve -> numThreadsPerBlock ) ) + 1;
  sprintf ( p_gr_solve -> type, "%s", type );

  cudaDeviceSynchronize ( );
  device_mem_allocation <<< 1, 1 >>> ( gr -> nc, l_nnz, d_gr_solve, d_val, d_val_ori, d_b, d_col, d_row, d_dig, d_dammy );
  cudaDeviceSynchronize ( );

  cudaMemcpy ( p_gr_solve ->     val, h_gr_solve ->     val, l_nnz * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_gr_solve -> val_ori, h_gr_solve -> val_ori, l_nnz * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_gr_solve ->       b, h_gr_solve ->       b, gr -> nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_gr_solve ->   dammy, h_gr_solve ->   dammy, gr -> nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_gr_solve ->     col, h_gr_solve ->     col, l_nnz * sizeof ( int ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_gr_solve ->     row, h_gr_solve ->     row, ( GR_COMP * gr -> n + 1 ) * sizeof ( int ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_gr_solve ->     dig, h_gr_solve ->     dig, ( GR_COMP * gr -> n     ) * sizeof ( int ), cudaMemcpyHostToDevice );

  free ( h_gr_solve -> val );  free ( h_gr_solve -> val_ori );  free ( h_gr_solve -> b   );
  free ( h_gr_solve -> col );  free ( h_gr_solve -> row );      free ( h_gr_solve -> dig );
  free ( h_gr_solve -> dammy );

  // set vec
  double **d_vec;
  if ( 0 == strncmp ( p_gr_solve -> type, "RKC", 3 ) ) { 
    cudaMalloc ( ( double *** ) &d_vec, n_vec_RKC * sizeof ( double * ) );
    p_gr_solve -> vec = ( double ** ) malloc ( n_vec_RKC * sizeof ( double * ) );
    for ( int i = 0; i < n_vec_RKC; i++ ) {
      cudaMalloc ( ( double ** ) ( & ( p_gr_solve -> vec [ i ] ) ), gr -> nc * sizeof ( double ) );
      device_mem_allocation2 <<< 1, 1 >>> ( i, d_vec, p_gr_solve -> vec [ i ] );
    }
    device_mem_allocation3 <<< 1, 1 >>> ( d_gr_solve, d_vec );
    RKC_vec_initialize ( d_gr, d_gr_solve, gr, p_gr_solve );
  }
  /**/
  else if ( 0 == strncmp ( p_gr_solve -> type, "CN", 2 ) ) { 
    cudaMalloc ( ( double *** ) &d_vec, n_vec_CNm * sizeof ( double * ) );
    p_gr_solve -> vec = ( double ** ) malloc ( n_vec_CNm * sizeof ( double * ) );
    for ( int i = 0; i < n_vec_CNm; i++ ) {
      cudaMalloc ( ( double ** ) ( & ( p_gr_solve -> vec [ i ] ) ), gr -> nc * sizeof ( double ) );
      device_mem_allocation2 <<< 1, 1 >>> ( i, d_vec, p_gr_solve -> vec [ i ] );
    }
    device_mem_allocation3 <<< 1, 1 >>> ( d_gr_solve, d_vec );
    //printf ( "pre gr_cnm_vec_initialize\n" ); // Debug
    gr_cnm_vec_initialize <<< p_gr_solve -> numBlocks, p_gr_solve -> numThreadsPerBlock >>> ( d_gr, d_gr_solve );
    gr_update_ion <<< p_gr_solve -> numBlocks, p_gr_solve -> numThreadsPerBlock >>> ( d_gr, d_gr_solve, CN_DT );
    double **l_ion  = gr -> ion;
    gr_Na_update <<< p_gr_solve -> numBlocks, p_gr_solve -> numThreadsPerBlock >>>
    ( gr -> nc, gr -> elem [ v ], CN_DT,  gr -> elem [ compart ],
      l_ion [ o_Na ],  l_ion [ c1_Na ], l_ion [ c2_Na ], l_ion [ c3_Na ], l_ion [ c4_Na ], l_ion [ c5_Na ],
      l_ion [ i1_Na ], l_ion [ i2_Na ], l_ion [ i3_Na ], l_ion [ i4_Na ], l_ion [ i5_Na ], l_ion [ i6_Na ] );
    // Debug
    //printf ("\n");
  }
  /*
  else if ( 0 == strncmp ( gr_solve -> type, "RUNGE_KUTTA_4", 13 ) ) { 
    gr_solve -> vec = ( double ** ) malloc ( gr_n_vec_RK4 * sizeof ( double *) );
    for ( int i = 0; i < gr_n_vec_RK4; i++ ) {
      gr_solve -> vec [ i ] = ( double * ) malloc ( gr -> nc * sizeof ( double ) );
    }
  }*/
  free ( h_gr_solve );
  return d_gr_solve;
}

void gr_solve_finalize ( const int n_gr, neuron_solve_t *d_gr_solve, neuron_solve_t *p_gr_solve )
{
  if ( n_gr > 0 ) {
    cudaFree ( p_gr_solve -> val );
    cudaFree ( p_gr_solve -> val_ori );
    cudaFree ( p_gr_solve -> b   );
    cudaFree ( p_gr_solve -> dammy );
    cudaFree ( p_gr_solve -> col );
    cudaFree ( p_gr_solve -> row );
    cudaFree ( p_gr_solve -> dig );
    if ( 0 == strncmp ( p_gr_solve -> type, "RKC", 3 ) ) 
    { 
      for ( int i = 0; i < n_vec_RKC; i++ ) { cudaFree ( p_gr_solve -> vec [ i ] ); }    
      free ( p_gr_solve -> vec );
      free ( p_gr_solve -> h_work );
      free ( p_gr_solve -> h_others );
      free ( p_gr_solve -> h_bool );
    } 
    else if ( 0 == strncmp ( p_gr_solve -> type, "CN", 2 ) ) 
    { 
      for ( int i = 0; i < n_vec_CNm; i++ ) { cudaFree ( p_gr_solve -> vec [ i ] ); }
      free ( p_gr_solve -> vec );
    }
    /*
    else if ( 0 == strncmp ( gr_solve -> type, "RUNGE_KUTTA_4", 13 ) ) { 
      for ( int i = 0; i < gr_n_vec_RK4; i++ ) { free ( gr_solve -> vec [ i ] ); }
    }*/
  }
  cudaFree ( d_gr_solve ); free ( p_gr_solve );
}

__host__
void gr_solve_update_v ( neuron_t *d_gr, neuron_solve_t *d_gr_solve, 
                         neuron_t *p_gr, neuron_solve_t *p_gr_solve, 
                         synapse_t *d_mfgr, synapse_t *d_gogr )
{
  if      ( 0 == strncmp ( p_gr_solve -> type, "BE", 2 ) ) { 
    gr_solve_by_bem ( d_gr, d_gr_solve, p_gr, p_gr_solve, d_mfgr, d_gogr );
  }
  else if ( 0 == strncmp ( p_gr_solve -> type, "CN", 2 ) ) { 
    gr_solve_by_cnm ( d_gr, d_gr_solve, p_gr, p_gr_solve, d_mfgr, d_gogr );
  }
  else if ( 0 == strncmp ( p_gr_solve -> type, "RKC", 3 ) ) { 
    gr_solve_by_rkc ( d_gr, d_gr_solve, p_gr, p_gr_solve, d_mfgr, d_gogr );
  }
  else { printf ( "solver Error\n" ); }
}