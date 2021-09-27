#include "pkj_solve.cuh"

__host__
static void sort_csr ( double *val, int *col, int *row, int s, int pre_gj, int post_gj, int nc, int nnz )
{
  int l_c = post_gj;
  int m_c = col [ s ];
  double l_v = - G_GJ_PKJ;
  double m_v = val [ s ];
  for ( int i = s; i < nnz; i++ )
  {
    col [ i ] = l_c; 
    val [ i ] = l_v;
    l_c = m_c; 
    l_v = m_v; 
    if ( i < nnz - 1 ) 
    {
      m_c = col [ i + 1 ];
      m_v = val [ i + 1 ];
    }
  }
  for ( int i = pre_gj + 1; i < nc + 1; i++ )
  {
    row [ i ]++;
  }
}

__host__
static neuron_solve_t *set_host_pkj_solve ( const char *type, neuron_t *pkj )
{
  neuron_solve_t *pkj_solve =  ( neuron_solve_t * ) malloc ( sizeof ( neuron_solve_t ) );
  int n_pkj = pkj -> n;
  int pkj_x = pkj -> nx;
  int pkj_y = pkj -> ny;

  // Matrix  
  double *mat = ( double * ) malloc ( PKJ_COMP * PKJ_COMP * sizeof ( double ) );
  int *l_connect = ( int * ) malloc ( PKJ_COMP * sizeof ( int ) );
  for ( int i = 0; i < PKJ_COMP * PKJ_COMP; i++ ) { mat [ i ] = 0.0; }  

  // !!!DUPLICATE CODE!!!
  double rad [ PKJ_COMP ], len [ PKJ_COMP ], Ra [ PKJ_COMP ];
  FILE *file = fopen ( PARAM_FILE_PKJ, "r" );
  if ( ! file ) { fprintf ( stderr, "no such file %s\n", PARAM_FILE_PKJ ); exit ( 1 ); }

  for ( int i = 0; i < PKJ_COMP; i++ ) {
    int i1, i2, i3;
    double x, y, z, d;
    
    if ( fscanf ( file, "%d %d %lf %lf %lf %lf %d ", &i1, &i2, &x, &y, &z, &d, &i3 ) == ( EOF ) ){
        printf ( "PARAM_FILE_READING_ERROR\n" );
        exit ( 1 );
    }
    if ( i3 == PKJ_soma ){
      rad [ i1 ] = 0.5 * d * 1e-4; // [mum -> cm]
      len [ i1 ] = d * 1e-4; // [mum -> cm]
      Ra  [ i1 ] = PKJ_RI; // [kohm-cm]
    }
    else 
    { // pkj_dend // main, spiny, smooth
      rad [ i1 ] = 0.5 * d * 1e-4; // [mum -> cm]
      len [ i1 ] = sqrt ( pow ( x, 2 ) + pow ( y, 2 )+pow ( z, 2 ) ) * 1.0e-4; // [mum -> cm]
      Ra  [ i1 ] = PKJ_RI; // [kohm-cm]
    }   
    l_connect [ i ] = i2;
  }

  for ( int i = 0; i < PKJ_COMP; i++ ) {
    int d = l_connect [ i ];
    if ( d >= 0 ) {
      mat [ d + PKJ_COMP * i ] = ( 2.0 / ( ( Ra [ i ] * len [ i ] ) / ( rad [ i ] * rad [ i ] * M_PI ) + ( Ra [ d ] * len [ d ] ) / ( rad [ d ] * rad [ d ] * M_PI ) ) ); // [mS]
      mat [ i + PKJ_COMP * d ] = mat [ d + PKJ_COMP * i ]; // i*NPKJ -> set Rows, +d -> set Columns      
    }
  }

  // Debug
  //for ( int i = 0; i < PKJ_COMP; i++ ) { printf ("mat [%d] = %f\n", i, mat [i]); }
  
  // number of among_Pkjlgi_cells GJ
  int n_gj = 0; int c_gj = 0;
  if ( n_pkj > 1 ) n_gj = 2 * GJ_EACH_PKJ * ( pkj_x - 1 ) * pkj_y;
  int * pre_connect_gj  = ( int * ) calloc( n_gj, sizeof ( int ) );
  int * post_connect_gj = ( int * ) calloc( n_gj, sizeof ( int ) );

  // read GJ info
  FILE *f_gj = fopen ( "Pkj_GJ.csv", "r" );
  if ( ! f_gj ) { fprintf ( stderr, "no such file %s\n", "Pkj_GJ.csv" ); exit ( 1 ); }
  int l_pre [ GJ_EACH_PKJ ], l_post [ GJ_EACH_PKJ ], i1, i2;
  for ( int i = 0; i < GJ_EACH_PKJ; i++ ) 
  {
    if ( fscanf ( f_gj, "%d, %d", &i1, &i2 ) == ( EOF ) ) {
        printf ( "PARAM_FILE_READING_ERROR\n" );
        exit ( 1 );
    }
    l_pre [ i ] = i1;
    l_post [ i ] = i2;
  }
  fclose ( f_gj );

  if ( n_gj > 0 )
  {
    for( int y = 0; y < pkj_y; y++ ) { 
      for( int x = 0; x < pkj_x - 1; x++ ) {
        for ( int i = 0; i < GJ_EACH_PKJ; i++ )
        {
          pre_connect_gj [ c_gj * 2     ] = l_pre  [ i ] + ( ( x     ) + y * pkj_x ) * PKJ_COMP;
          pre_connect_gj [ c_gj * 2 + 1 ] = l_post [ i ] + ( ( x + 1 ) + y * pkj_x ) * PKJ_COMP;
          c_gj++;
        }
      }
    }
    for ( int i = 0; i < n_gj; i += 2 )
    {
      post_connect_gj [ i + 1 ] = pre_connect_gj [ i ];
      post_connect_gj [ i ] = pre_connect_gj [ i + 1 ];
    }
  }
  if ( c_gj == 0 ) { printf ( "without PKJ_gj\n" ); }
  else if ( c_gj * 2 == n_gj ) { printf( "PKJ_gj_num = success \n" ); }
  else { printf ( "PKJ_gj_num = error \n" ); exit ( 1 ); }  
  
  // count number of nonzero elements NNZ
  int nnz = PKJ_COMP;
  for ( int i = 0; i < PKJ_COMP * PKJ_COMP; i++ ) { nnz += ( mat [ i ] > 0.0 ); } 
  pkj_solve -> nnz = nnz * ( pkj -> n ) + n_gj;  printf ( "PKJ -> nnz = %d\n", pkj_solve -> nnz );
  pkj_solve -> val     = ( double * ) malloc ( pkj_solve -> nnz          * sizeof ( double ) ); // nonzero and diapkjnal elements array
  pkj_solve -> val_ori = ( double * ) malloc ( pkj_solve -> nnz          * sizeof ( double ) );
  pkj_solve -> b       = ( double * ) malloc ( PKJ_COMP * pkj -> n         * sizeof ( double ) ); // b value
  pkj_solve -> dammy   = ( double * ) malloc ( PKJ_COMP * pkj -> n         * sizeof ( double ) ); // calculation vec
  pkj_solve -> col     = ( int *    ) malloc ( pkj_solve -> nnz          * sizeof ( int ) ); //column number of val's elements
  pkj_solve -> row     = ( int *    ) malloc ( ( PKJ_COMP * pkj -> n + 1 ) * sizeof ( int ) ); //row index
  pkj_solve -> dig     = ( int *    ) malloc ( PKJ_COMP * pkj -> n         * sizeof ( int ) ); //dig index
  sprintf ( pkj_solve -> type, "%s", type ); 
  
  double *val     = pkj_solve -> val;
  double *val_ori = pkj_solve -> val_ori;
  double *b       = pkj_solve -> b;
  double *dammy   = pkj_solve -> dammy;
  int *col  = pkj_solve -> col; 
  int *row  = pkj_solve -> row;
  int *dig  = pkj_solve -> dig;
  
  for ( int i = 0; i < pkj_solve -> nnz; i++ )     { val [ i ] = val_ori [ i ] = 0.0; col [ i ] = 0; }
  for ( int i = 0; i < PKJ_COMP * pkj -> n + 1; i++) { row [ i ] = 0; }
  for ( int i = 0; i < PKJ_COMP * pkj -> n; i++)     { dig [ i ] = 0; b [ i ] = 0; dammy [ i ] = 0.0; }
        
  // Create CSR
  int num_row = 0, num_col = 0, num_dig = 0, num = 0, count_row = 0;
  //int count_gj = 0;
  
  for ( int i = 0; i < PKJ_COMP * PKJ_COMP; i++ ) 
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
    if ( num_col == PKJ_COMP ) 
    {
      for (int j = num_row * PKJ_COMP; j < num_row * PKJ_COMP + PKJ_COMP; j++ ) 
      {
        if ( j != num_row * PKJ_COMP + num_row ) {
          val [ num_dig ] += mat [ j ];
          dig [ num_row ] = num_dig;
        }
      }
      num_col = 0;
      num_row++;
      row [ num_row ] = count_row;
    }
  }
  
  for ( int i = 1; i < pkj -> n; i++ ) 
  {
    for ( int j = 0; j < nnz; j++ ) 
    {
      val [ j + nnz * i ] = val [ j ];
      //val_ori [ j + nnz * i ] = val [ j ];
      col [ j + nnz * i ] = col [ j ] + PKJ_COMP * i;
    }
    for ( int j = 0; j < PKJ_COMP; j++ ) 
    {
      row [ j + PKJ_COMP * i ] = row [ j ] + nnz * i;
      dig [ j + PKJ_COMP * i ] = dig [ j ] + nnz * i;
    }
  }
  //row [ PKJ_COMP * pkj -> n ] = pkj_solve -> nnz;  
  row [ PKJ_COMP * pkj -> n ] = pkj -> n * nnz;  

  // Adding GJ val
  for ( int i = 0; i < n_gj; i++ ) { val [ dig [ pre_connect_gj [ i ] ] ] += G_GJ_PKJ; }
  for ( int i = 0; i < n_gj; i++ )
  {
    int pre_gj  = pre_connect_gj  [ i ];
    int post_gj = post_connect_gj [ i ];
    int flag_gj = 0;
    for ( int j = row [ pre_gj ]; j < row [ pre_gj + 1 ]; j++ )
    {
      if ( col [ j ] == post_gj )
      {  
        printf (" ERROR!! DUPLICATE CONNECTION in PURKINJE!! %d -> %d\n ", pre_gj, post_gj );
        exit ( 1 );
      }
      if ( col [ j ] > post_gj )
      {
        sort_csr ( val, col, row, j, pre_gj, post_gj, pkj -> nc, pkj_solve -> nnz );
        flag_gj = 1;
        break;
      }
    }
    if ( flag_gj == 0 )
    {
      sort_csr ( val, col, row, row [ pre_gj + 1 ], pre_gj, post_gj, pkj -> nc, pkj_solve -> nnz );
      flag_gj = 0;
    }
  } 

  num_dig = 0;
  for ( int i = 0; i < pkj_solve -> nnz; i++ )
  {
    if ( val [ i ] > 0.0 ) 
    {
      dig [ num_dig ] = i;
      num_dig++;
    }
  }
  
  for ( int i = 0; i < pkj_solve -> nnz; i++ ) 
  {  
    if ( ( 0 == strncmp ( pkj_solve -> type, "FORWARD_EULER", 13 ) ) ||
         ( 0 == strncmp ( pkj_solve -> type, "RUNGE_KUTTA_4", 13 ) ) ||
         ( 0 == strncmp ( pkj_solve -> type, "RKC", 3 ) ) ) { val [ i ] *= -1; }
    if ( 0 == strncmp ( pkj_solve -> type, "CN", 2 ) ) { val [ i ] /= 2.0; }

    val_ori [ i ] = val [ i ];
  } // nnz is OK.
    
  free ( mat );
  free ( l_connect );
  free ( pre_connect_gj );
  free ( post_connect_gj );
  
  return pkj_solve;
}

__global__
static void device_mem_allocation ( const int nc, const int l_nnz, neuron_solve_t* d_pkj_solve,
     double *d_val, double *d_val_ori, double *d_b, int *d_col, int *d_row, int *d_dig, double *d_dammy ) 
{
  d_pkj_solve -> nnz     = l_nnz;
  d_pkj_solve -> val     = d_val;
  d_pkj_solve -> val_ori = d_val_ori;
  d_pkj_solve -> b       = d_b;
  d_pkj_solve -> col     = d_col;
  d_pkj_solve -> row     = d_row;   // # of neurons
  d_pkj_solve -> dig     = d_dig; // # of all compartments
  d_pkj_solve -> dammy   = d_dammy;
  d_pkj_solve -> numThreadsPerBlock = 128;
  d_pkj_solve -> numBlocks = ( int ) ( ( nc + d_pkj_solve -> numThreadsPerBlock - 1 ) / d_pkj_solve -> numThreadsPerBlock );
  //Debug
  //printf ( "From GPU \n n = %d, nc = %d\n", d_pkj -> n, d_pkj -> nc );
}

__global__
static void device_mem_allocation2 (const int n, double ** dev, double *ptr )
{
  dev [ n ] = ptr;
}

__global__ static
void device_mem_allocation3 ( neuron_solve_t* d_pkj_solve, double **d_vec ) 
{
  d_pkj_solve -> vec = d_vec;
}

neuron_solve_t *pkj_solve_initialize ( neuron_solve_t *p_pkj_solve, const char *type, neuron_t *pkj, neuron_t *d_pkj ) // tentatively, type is ignored
{
  neuron_solve_t *d_pkj_solve;
  cudaMalloc ( ( neuron_solve_t **) &d_pkj_solve, sizeof ( neuron_solve_t ) );
  if ( pkj -> nc == 0 ) { return d_pkj_solve; }

  neuron_solve_t *h_pkj_solve = set_host_pkj_solve ( type, pkj );  
  
  int l_nnz = h_pkj_solve -> nnz;
  double *d_val, *d_val_ori, *d_b, *d_dammy;
  cudaMalloc ( ( double ** ) &d_val,     l_nnz * sizeof ( double ) );
  cudaMalloc ( ( double ** ) &d_val_ori, l_nnz * sizeof ( double ) );
  cudaMalloc ( ( double ** ) &d_b,       pkj -> nc * sizeof ( double ) );
  cudaMalloc ( ( double ** ) &d_dammy,   pkj -> nc * sizeof ( double ) );
  int *d_col, *d_row, *d_dig;
  cudaMalloc ( ( int ** ) &d_col, l_nnz * sizeof ( int ) );
  cudaMalloc ( ( int ** ) &d_row, ( PKJ_COMP * pkj -> n + 1 ) * sizeof ( int ) );
  cudaMalloc ( ( int ** ) &d_dig, ( PKJ_COMP * pkj -> n     ) * sizeof ( int ) );
  
  p_pkj_solve -> val     = d_val;
  p_pkj_solve -> val_ori = d_val_ori;
  p_pkj_solve -> b       = d_b;
  p_pkj_solve -> dammy   = d_dammy;
  p_pkj_solve -> col = d_col;
  p_pkj_solve -> row = d_row;
  p_pkj_solve -> dig = d_dig;
  p_pkj_solve -> nnz = l_nnz;
  p_pkj_solve -> numThreadsPerBlock = 128;
  p_pkj_solve -> numBlocks = ( int ) ( ( pkj -> nc + p_pkj_solve -> numThreadsPerBlock - 1 ) / p_pkj_solve -> numThreadsPerBlock );
  sprintf ( p_pkj_solve -> type, "%s", type );

  cudaDeviceSynchronize ( );
  device_mem_allocation <<< 1, 1 >>> ( pkj -> nc, l_nnz, d_pkj_solve, d_val, d_val_ori, d_b, d_col, d_row, d_dig, d_dammy );
  cudaDeviceSynchronize ( );

  cudaMemcpy ( p_pkj_solve ->     val, h_pkj_solve ->     val, l_nnz * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_pkj_solve -> val_ori, h_pkj_solve -> val_ori, l_nnz * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_pkj_solve ->       b, h_pkj_solve ->       b, pkj -> nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_pkj_solve ->   dammy, h_pkj_solve ->   dammy, pkj -> nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_pkj_solve ->     col, h_pkj_solve ->     col, l_nnz * sizeof ( int ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_pkj_solve ->     row, h_pkj_solve ->     row, ( PKJ_COMP * pkj -> n + 1 ) * sizeof ( int ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_pkj_solve ->     dig, h_pkj_solve ->     dig, ( PKJ_COMP * pkj -> n     ) * sizeof ( int ), cudaMemcpyHostToDevice );

  //Debug
  //for ( int i = 0; i < l_nnz; i++ ) 
  //  printf ( "host val [ %d ] = %f\n", i, h_pkj_solve -> val [ i ] );
  //test_mem_allocation  <<< 1, 1 >>> ( d_pkj_solve );  
  cudaDeviceSynchronize ( );

  free ( h_pkj_solve -> val );  free ( h_pkj_solve -> val_ori );  free ( h_pkj_solve -> b   );
  free ( h_pkj_solve -> col );  free ( h_pkj_solve -> row );      free ( h_pkj_solve -> dig );
  free ( h_pkj_solve -> dammy );

  
  // set vec
  double **d_vec;
  if ( 0 == strncmp ( p_pkj_solve -> type, "RKC", 3 ) ) { 
    cudaMalloc ( ( double *** ) &d_vec, n_vec_RKC * sizeof ( double * ) );
    p_pkj_solve -> vec = ( double ** ) malloc ( n_vec_RKC * sizeof ( double * ) );
    for ( int i = 0; i < n_vec_RKC; i++ ) {
      cudaMalloc ( ( double ** ) ( & ( p_pkj_solve -> vec [ i ] ) ), pkj -> nc * sizeof ( double ) );
      device_mem_allocation2 <<< 1, 1 >>> ( i, d_vec, p_pkj_solve -> vec [ i ] );
    }
    device_mem_allocation3 <<< 1, 1 >>> ( d_pkj_solve, d_vec );
    RKC_vec_initialize (  d_pkj, d_pkj_solve, pkj, p_pkj_solve );
  }
  else if ( 0 == strncmp ( p_pkj_solve -> type, "CN", 2 ) ) { 
    cudaMalloc ( ( double *** ) &d_vec, n_vec_CNm * sizeof ( double * ) );
    p_pkj_solve -> vec = ( double ** ) malloc ( n_vec_CNm * sizeof ( double * ) );
    for ( int i = 0; i < n_vec_CNm; i++ ) {
      cudaMalloc ( ( double ** ) ( & ( p_pkj_solve -> vec [ i ] ) ), pkj -> nc * sizeof ( double ) );
      device_mem_allocation2 <<< 1, 1 >>> ( i, d_vec, p_pkj_solve -> vec [ i ] );
    }
    device_mem_allocation3 <<< 1, 1 >>> ( d_pkj_solve, d_vec );
    pkj_cnm_vec_initialize <<< p_pkj_solve -> numBlocks, p_pkj_solve -> numThreadsPerBlock >>> ( d_pkj, d_pkj_solve );
    //pkj_update_ion_2nd <<< p_pkj_solve -> numBlocks, p_pkj_solve -> numThreadsPerBlock >>> ( d_pkj, d_pkj_solve, CN_DT );
  }
  free ( h_pkj_solve );

  return d_pkj_solve;
}

void pkj_solve_finalize ( const int n_pkj, neuron_solve_t *d_pkj_solve, neuron_solve_t *p_pkj_solve )
{
  if ( n_pkj > 0 ) {
    cudaFree ( p_pkj_solve -> val );
    cudaFree ( p_pkj_solve -> val_ori );
    cudaFree ( p_pkj_solve -> b   );
    cudaFree ( p_pkj_solve -> dammy );
    cudaFree ( p_pkj_solve -> col );
    cudaFree ( p_pkj_solve -> row );
    cudaFree ( p_pkj_solve -> dig );
    if ( 0 == strncmp ( p_pkj_solve -> type, "RKC", 3 ) ) 
    { 
      for ( int i = 0; i < n_vec_RKC; i++ ) { cudaFree ( p_pkj_solve -> vec [ i ] ); }   
      free ( p_pkj_solve -> vec );
      free ( p_pkj_solve -> h_work );
      free ( p_pkj_solve -> h_others );
      free ( p_pkj_solve -> h_bool );
    } 
    else if ( 0 == strncmp ( p_pkj_solve -> type, "CN", 2 ) ) 
    { 
      for ( int i = 0; i < n_vec_CNm; i++ ) { cudaFree ( p_pkj_solve -> vec [ i ] ); }
      free ( p_pkj_solve -> vec );
    }
  }
  cudaFree ( d_pkj_solve ); free ( p_pkj_solve );
}
__host__
void pkj_solve_update_v ( neuron_t *d_pkj, neuron_solve_t *d_pkj_solve,
                          neuron_t *p_pkj, neuron_solve_t *p_pkj_solve,
                          synapse_t *d_grpkj, synapse_t *d_mlipkj )
{
  if      ( 0 == strncmp ( p_pkj_solve -> type, "BE", 2 ) ) { 
    pkj_solve_by_bem ( d_pkj, d_pkj_solve, p_pkj, p_pkj_solve );
  }
  else if ( 0 == strncmp ( p_pkj_solve -> type, "CN", 2 ) ) { 
    pkj_solve_by_cnm ( d_pkj, d_pkj_solve, p_pkj, p_pkj_solve, d_grpkj, d_mlipkj );
  }
  else if ( 0 == strncmp ( p_pkj_solve -> type, "RKC", 3 ) ) { 
    pkj_solve_by_rkc ( d_pkj, d_pkj_solve, p_pkj, p_pkj_solve, d_grpkj, d_mlipkj );
  }
  else { printf ( "solver Error\n" ); }
}
