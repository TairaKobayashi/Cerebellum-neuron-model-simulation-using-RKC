#include "io_solve.cuh"
__host__ static 
neuron_solve_t* set_host_io_solve ( const char *type, neuron_t *io )
{
  neuron_solve_t *io_solve =  ( neuron_solve_t * ) malloc ( sizeof ( neuron_solve_t ) );
  //int n_io = io -> n;
  //int io_x = io -> nx;
  //int io_y = io -> ny;

  // Matrix  
  double *mat = ( double * ) malloc ( IO_COMP * IO_COMP * sizeof ( double ) );
  int *l_connect = ( int * ) malloc ( IO_COMP * sizeof ( int ) );
  int *l_compart = ( int * ) malloc ( IO_COMP * sizeof ( int ) );
  for ( int i = 0; i < IO_COMP * IO_COMP; i++ ) { mat [ i ] = 0.0; }

  // !!!DUPLICATE CODE!!!
  //double rad [ IO_COMP ], len [ IO_COMP ], Ra [ IO_COMP ];
  FILE *file = fopen ( PARAM_FILE_IO, "r" );
  if ( ! file ) { fprintf ( stderr, "no such file %s\n", PARAM_FILE_IO ); exit ( 1 ); }

  for ( int i = 0; i < IO_COMP; i++ ) {
    int i1, i2, i3;
    double d1, d2, d4;
    
    if ( fscanf ( file, "%d %d %lf %lf %lf %d ",
      &i1, &i2, &d1, &d2, &d4, &i3 ) == ( EOF ) ) {
      printf ( "IO_PARAM_FILE_READING_ERROR\n" );
      exit ( 1 );
    }
  
    //rad [ i1 ] = 0.5 * d1 * 1e-4; // [mum -> cm]
    //len [ i1 ] = d2 * 1e-4; // [mum -> cm]
    //Ra  [ i1 ] = d3 * 1e-3; // [kohm-cm]

    l_connect [ i ] = i2;
    l_compart [ i ] = i3;
  }

  for ( int i = 0; i < IO_COMP; i++ ) {
    int d  = l_connect [ i ];
    int i3 = l_compart [ i ];
    double l_g = 0.0; double l_g_rev = 0.0;
    if ( d >= 0 ) {
      if ( i3 == 0 && d == IO_COMP_DEND ) 
      { 
        l_g = ( G_INT_IO / ( 1.0 - G_P_DS_IO ) ); l_g_rev = ( G_INT_IO / ( G_P_DS_IO ) ); 
        //l_g = l_g_rev = ( G_INT_IO / ( 0.5 ) ); 
      }
      else if ( i3 == 1 && d == IO_COMP_DEND + IO_COMP_SOMA ) 
      { 
        l_g = ( G_INT_IO / ( G_P_SA_IO ) ); l_g_rev = ( G_INT_IO / ( 1.0 - G_P_SA_IO ) ); 
        //l_g = l_g_rev = ( G_INT_IO / ( 0.5 ) ); 
      }
      else if ( i3 == 0 && d < IO_COMP_DEND ) 
      { 
        //l_g = ( G_INT_IO / ( 1.0 - 0.5 ) ); l_g_rev = ( G_INT_IO / ( 0.5 ) ); 
        l_g = l_g_rev = ( G_INT_IO / ( 0.5 ) ); 
      }
      else { printf ("io_solve Error\n"); }

      mat [ d + IO_COMP * i ] = l_g; // [mS / cm^2]
      mat [ i + IO_COMP * d ] = l_g_rev; // i*NIO -> set Rows, +d -> set Columns      
      //mat [ d + IO_COMP * i ] = l_g_rev; // [mS / cm^2]
      //mat [ i + IO_COMP * d ] = l_g; // i*NIO -> set Rows, +d -> set Columns      
    }
  }

  // Debug
  //for ( int i = 0; i < IO_COMP; i++ ) { printf ("mat [%d] = %f\n", i, mat [i]); }
      
  // count number of nonzero elements NNZ
  int nnz = IO_COMP;
  for ( int i = 0; i < IO_COMP * IO_COMP; i++ ) { nnz += ( mat [ i ] > 0.0 ); } 
  io_solve -> nnz = nnz * ( io -> n );  printf ( "IO -> nnz = %d\n", io_solve -> nnz );
  io_solve -> val     = ( double * ) malloc ( io_solve -> nnz          * sizeof ( double ) ); // nonzero and diaional elements array
  io_solve -> val_ori = ( double * ) malloc ( io_solve -> nnz          * sizeof ( double ) );
  io_solve -> b       = ( double * ) malloc ( IO_COMP * io -> n         * sizeof ( double ) ); // b value
  io_solve -> dammy   = ( double * ) malloc ( IO_COMP * io -> n         * sizeof ( double ) ); // calculation vec
  io_solve -> col     = ( int *    ) malloc ( io_solve -> nnz          * sizeof ( int ) ); //column number of val's elements
  io_solve -> row     = ( int *    ) malloc ( ( IO_COMP * io -> n + 1 ) * sizeof ( int ) ); //row index
  io_solve -> dig     = ( int *    ) malloc ( IO_COMP * io -> n         * sizeof ( int ) ); //dig index
  sprintf ( io_solve -> type, "%s", type ); 
  
  double *val     = io_solve -> val;
  double *val_ori = io_solve -> val_ori;
  double *b       = io_solve -> b;
  double *dammy   = io_solve -> dammy;
  int *col  = io_solve -> col; 
  int *row  = io_solve -> row;
  int *dig  = io_solve -> dig;
  
  for ( int i = 0; i < io_solve -> nnz; i++ )     { val [ i ] = val_ori [ i ] = 0.0; col [ i ] = 0; }
  for ( int i = 0; i < IO_COMP * io -> n + 1; i++) { row [ i ] = 0; }
  for ( int i = 0; i < IO_COMP * io -> n; i++)     { dig [ i ] = 0; b [ i ] = 0; dammy [ i ] = 0.0; }
        
  // Create CSR
  int num_row = 0, num_col = 0, num_dig = 0, num = 0, count_row = 0;
  //int count_gj = 0;
  
  for ( int i = 0; i < IO_COMP * IO_COMP; i++ ) 
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
    if ( num_col == IO_COMP ) 
    {
      for (int j = num_row * IO_COMP; j < num_row * IO_COMP + IO_COMP; j++ ) 
      {
        if ( j != num_row * IO_COMP + num_row ) {
          val [ num_dig ] += mat [ j ];
          dig [ num_row ] = num_dig;
        }
      }
      num_col = 0;
      num_row++;
      row [ num_row ] = count_row;
    }
  }
  
  for ( int i = 1; i < io -> n; i++ ) 
  {
    for ( int j = 0; j < nnz; j++ ) 
    {
      val [ j + nnz * i ] = val [ j ];
      //val_ori [ j + nnz * i ] = val [ j ];
      col [ j + nnz * i ] = col [ j ] + IO_COMP * i;
    }
    for ( int j = 0; j < IO_COMP; j++ ) 
    {
      row [ j + IO_COMP * i ] = row [ j ] + nnz * i;
      dig [ j + IO_COMP * i ] = dig [ j ] + nnz * i;
    }
  }
  row [ IO_COMP * io -> n ] = io_solve -> nnz;      
  
  for ( int i = 0; i < io_solve -> nnz; i++ ) 
  {  
    //
    if ( ( 0 == strncmp ( io_solve -> type, "FORWARD_EULER", 13 ) ) ||
         ( 0 == strncmp ( io_solve -> type, "RUNGE_KUTTA_4", 13 ) ) ||
         ( 0 == strncmp ( io_solve -> type, "RKC", 3 ) ) ) { val [ i ] *= -1; }
    //
    if ( 0 == strncmp ( io_solve -> type, "CN", 2 ) ) { val [ i ] /= 2.0; }

    val_ori [ i ] = val [ i ];
  } // nnz is OK.
    
  free ( mat );
  free ( l_connect );
  
  return io_solve;
}

__global__
static void device_mem_allocation ( const int nc, const int l_nnz, neuron_solve_t* d_io_solve,
     double *d_val, double *d_val_ori, double *d_b, int *d_col, int *d_row, int *d_dig, double *d_dammy ) 
{
  d_io_solve -> nnz     = l_nnz;
  d_io_solve -> val     = d_val;
  d_io_solve -> val_ori = d_val_ori;
  d_io_solve -> b       = d_b;
  d_io_solve -> col     = d_col;
  d_io_solve -> row     = d_row;   // # of neurons
  d_io_solve -> dig     = d_dig; // # of all compartments
  d_io_solve -> dammy   = d_dammy;
  d_io_solve -> numThreadsPerBlock = 128;
  d_io_solve -> numBlocks = ( int ) ( nc / d_io_solve -> numThreadsPerBlock ) + 1;
  //Debug
  //printf ( "From GPU \n n = %d, nc = %d\n", d_io -> n, d_io -> nc );
}

__global__
static void device_mem_allocation2 (const int n, double ** dev, double *ptr )
{
  dev [ n ] = ptr;
}

__global__ static
void device_mem_allocation3 ( neuron_solve_t* d_io_solve, double **d_vec ) 
{
  d_io_solve -> vec = d_vec;
}

neuron_solve_t *io_solve_initialize ( neuron_solve_t *p_io_solve, const char *type, neuron_t *io, neuron_t *d_io ) // tentatively, type is ignored
{
  neuron_solve_t *d_io_solve;
  cudaMalloc ( ( neuron_solve_t **) &d_io_solve, sizeof ( neuron_solve_t ) );
  if ( io -> nc == 0 ) { return d_io_solve; }

  neuron_solve_t *h_io_solve = set_host_io_solve ( type, io );  
  
  int l_nnz = h_io_solve -> nnz;
  double *d_val, *d_val_ori, *d_b, *d_dammy;
  cudaMalloc ( ( double ** ) &d_val,     l_nnz * sizeof ( double ) );
  cudaMalloc ( ( double ** ) &d_val_ori, l_nnz * sizeof ( double ) );
  cudaMalloc ( ( double ** ) &d_b,       io -> nc * sizeof ( double ) );
  cudaMalloc ( ( double ** ) &d_dammy,   io -> nc * sizeof ( double ) );
  int *d_col, *d_row, *d_dig;
  cudaMalloc ( ( int ** ) &d_col, l_nnz * sizeof ( int ) );
  cudaMalloc ( ( int ** ) &d_row, ( IO_COMP * io -> n + 1 ) * sizeof ( int ) );
  cudaMalloc ( ( int ** ) &d_dig, ( IO_COMP * io -> n     ) * sizeof ( int ) );
  
  p_io_solve -> val     = d_val;
  p_io_solve -> val_ori = d_val_ori;
  p_io_solve -> b       = d_b;
  p_io_solve -> dammy   = d_dammy;
  p_io_solve -> col = d_col;
  p_io_solve -> row = d_row;
  p_io_solve -> dig = d_dig;
  p_io_solve -> nnz = l_nnz;
  p_io_solve -> numThreadsPerBlock = 128;
  p_io_solve -> numBlocks = ( int ) ( ( io -> nc ) / ( p_io_solve -> numThreadsPerBlock ) ) + 1;
  sprintf ( p_io_solve -> type, "%s", type );

  cudaDeviceSynchronize ( );
  device_mem_allocation <<< 1, 1 >>> ( io -> nc, l_nnz, d_io_solve, d_val, d_val_ori, d_b, d_col, d_row, d_dig, d_dammy );
  cudaDeviceSynchronize ( );

  cudaMemcpy ( p_io_solve ->     val, h_io_solve ->     val, l_nnz * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_io_solve -> val_ori, h_io_solve -> val_ori, l_nnz * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_io_solve ->       b, h_io_solve ->       b, io -> nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_io_solve ->   dammy, h_io_solve ->   dammy, io -> nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_io_solve ->     col, h_io_solve ->     col, l_nnz * sizeof ( int ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_io_solve ->     row, h_io_solve ->     row, ( IO_COMP * io -> n + 1 ) * sizeof ( int ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_io_solve ->     dig, h_io_solve ->     dig, ( IO_COMP * io -> n     ) * sizeof ( int ), cudaMemcpyHostToDevice );

  free ( h_io_solve -> val );  free ( h_io_solve -> val_ori );  free ( h_io_solve -> b   );
  free ( h_io_solve -> col );  free ( h_io_solve -> row );      free ( h_io_solve -> dig );
  free ( h_io_solve -> dammy );

  // set vec
  double **d_vec;
  if ( 0 == strncmp ( p_io_solve -> type, "RKC", 3 ) ) { 
    cudaMalloc ( ( double *** ) &d_vec, n_vec_RKC * sizeof ( double * ) );
    p_io_solve -> vec = ( double ** ) malloc ( n_vec_RKC * sizeof ( double * ) );
    for ( int i = 0; i < n_vec_RKC; i++ ) {
      cudaMalloc ( ( double ** ) ( & ( p_io_solve -> vec [ i ] ) ), io -> nc * sizeof ( double ) );
      device_mem_allocation2 <<< 1, 1 >>> ( i, d_vec, p_io_solve -> vec [ i ] );
    }
    device_mem_allocation3 <<< 1, 1 >>> ( d_io_solve, d_vec );
    RKC_vec_initialize ( d_io, d_io_solve, io, p_io_solve );
  }
  /**/
  else if ( 0 == strncmp ( p_io_solve -> type, "CN", 2 ) ) { 
    cudaMalloc ( ( double *** ) &d_vec, n_vec_CNm * sizeof ( double * ) );
    p_io_solve -> vec = ( double ** ) malloc ( n_vec_CNm * sizeof ( double * ) );
    for ( int i = 0; i < n_vec_CNm; i++ ) {
      cudaMalloc ( ( double ** ) ( & ( p_io_solve -> vec [ i ] ) ), io -> nc * sizeof ( double ) );
      device_mem_allocation2 <<< 1, 1 >>> ( i, d_vec, p_io_solve -> vec [ i ] );
    }
    device_mem_allocation3 <<< 1, 1 >>> ( d_io_solve, d_vec );
    //printf ( "pre io_cnm_vec_initialize\n" ); // Debug
    io_cnm_vec_initialize <<< p_io_solve -> numBlocks, p_io_solve -> numThreadsPerBlock >>> ( d_io, d_io_solve );
    io_update_ion <<< p_io_solve -> numBlocks, p_io_solve -> numThreadsPerBlock >>> ( d_io, d_io_solve, CN_DT );
  }
  free ( h_io_solve );
  return d_io_solve;
}

void io_solve_finalize ( const int n_io, neuron_solve_t *d_io_solve, neuron_solve_t *p_io_solve )
{
  if ( n_io > 0 ) {
    cudaFree ( p_io_solve -> val );
    cudaFree ( p_io_solve -> val_ori );
    cudaFree ( p_io_solve -> b   );
    cudaFree ( p_io_solve -> dammy );
    cudaFree ( p_io_solve -> col );
    cudaFree ( p_io_solve -> row );
    cudaFree ( p_io_solve -> dig );
    if ( 0 == strncmp ( p_io_solve -> type, "RKC", 3 ) ) 
    { 
      for ( int i = 0; i < n_vec_RKC; i++ ) { cudaFree ( p_io_solve -> vec [ i ] ); }   
      free ( p_io_solve -> vec );
      free ( p_io_solve -> h_work );
      free ( p_io_solve -> h_others );
      free ( p_io_solve -> h_bool );
    } 
    else if ( 0 == strncmp ( p_io_solve -> type, "CN", 2 ) ) 
    { 
      for ( int i = 0; i < n_vec_CNm; i++ ) { cudaFree ( p_io_solve -> vec [ i ] ); }
      free ( p_io_solve -> vec );
    }
  }
  cudaFree ( d_io_solve ); free ( p_io_solve );
}

__host__
void io_solve_update_v ( neuron_t *d_io, neuron_solve_t *d_io_solve, 
                         neuron_t *p_io, neuron_solve_t *p_io_solve, gap_t *d_io_gap )
{
  if      ( 0 == strncmp ( p_io_solve -> type, "BE", 2 ) ) { 
    io_solve_by_bem ( d_io, d_io_solve, p_io, p_io_solve, d_io_gap );
  }
  else if ( 0 == strncmp ( p_io_solve -> type, "CN", 2 ) ) { 
    io_solve_by_cnm ( d_io, d_io_solve, p_io, p_io_solve, d_io_gap );
  }
  else if ( 0 == strncmp ( p_io_solve -> type, "RKC", 3 ) ) { 
    io_solve_by_rkc ( d_io, d_io_solve, p_io, p_io_solve, d_io_gap );
  }
  else { printf ( "solver Error\n" ); }
}