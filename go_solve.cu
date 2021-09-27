#include "go_solve.cuh"

__host__
static neuron_solve_t *set_host_go_solve ( const char *type, neuron_t *go )
{
  neuron_solve_t *go_solve =  ( neuron_solve_t * ) malloc ( sizeof ( neuron_solve_t ) );
  int n_go = go -> n;
  int go_x = go -> nx;
  int go_y = go -> ny;

  // Matrix  
  double *mat = ( double * ) malloc ( GO_COMP * GO_COMP * sizeof ( double ) );
  int *l_connect = ( int * ) malloc ( GO_COMP * sizeof ( int ) );
  for ( int i = 0; i < GO_COMP * GO_COMP; i++ ) { mat [ i ] = 0.0; }  

  // !!!DUPLICATE CODE!!!
  double rad [ GO_COMP ], len [ GO_COMP ], Ra [ GO_COMP ];
  FILE *file = fopen ( PARAM_FILE_GO, "r" );
  if ( ! file ) { fprintf ( stderr, "no such file %s\n", PARAM_FILE_GO ); exit ( 1 ); }

  for ( int i = 0; i < GO_COMP; i++ ) {
    int i1, i2, i3;
    double d1, d2, d3, d4, i_leak, i_NaT, i_NaR, i_NaP, i_KV, i_KA, i_KC;
    double i_Kslow, i_CaHVA, i_CaLVA, i_HCN1, i_HCN2, i_KAHP;
    
    if ( fscanf ( file, "%d %d %lf %lf %lf %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
        &i1, &i2, &d1, &d2, &d3, &d4, &i3, &i_leak, &i_NaT, &i_NaR, &i_NaP, &i_KV, &i_KA, &i_KC,
        &i_Kslow, &i_CaHVA, &i_CaLVA, &i_HCN1, &i_HCN2, &i_KAHP ) == ( EOF ) ){
        printf ( "PARAM_FILE_READING_ERROR\n" );
        exit ( 1 );
    }
  
    rad [ i1 ] = 0.5 * d1 * 1e-4; // [mum -> cm]
    len [ i1 ] = d2 * 1e-4; // [mum -> cm]
    Ra  [ i1 ] = d3 * 1e-3; // [kohm-cm]

    l_connect [ i ] = i2;
  }

  for ( int i = 0; i < GO_COMP; i++ ) {
    int d = l_connect [ i ];
    if ( d >= 0 ) {
      mat [ d + GO_COMP * i ] = ( 2.0 / ( ( Ra [ i ] * len [ i ] ) / ( rad [ i ] * rad [ i ] * M_PI ) + ( Ra [ d ] * len [ d ] ) / ( rad [ d ] * rad [ d ] * M_PI ) ) ); // [mS]
      mat [ i + GO_COMP * d ] = mat [ d + GO_COMP * i ]; // i*NGO -> set Rows, +d -> set Columns      
    }
  }

  // Debug
  //for ( int i = 0; i < GO_COMP; i++ ) { printf ("mat [%d] = %f\n", i, mat [i]); }
    
  // number of among_Golgi_cells GJ
  int n_gj = 0; int c_gj = 0;
  if ( n_go > 1 ) n_gj = 2 * ( ( go_x - 1 ) * go_y + ( go_y - 1 ) * go_x );
  int * pre_connect_gj  = ( int * ) calloc( n_gj, sizeof ( int ) );
  int * post_connect_gj = ( int * ) calloc( n_gj, sizeof ( int ) );
  if ( n_gj > 0 )
  {
    for( int y = 0; y < go_y; y++ ) { 
      for( int x = 0; x < go_x - 1 ; x++ ) {
        pre_connect_gj [ c_gj * 2     ] = ( y * ( go_x ) + ( x     ) ) * GO_COMP;
        pre_connect_gj [ c_gj * 2 + 1 ] = ( y * ( go_x ) + ( x + 1 ) ) * GO_COMP;  //( y * ( go_x - 1 ) + x )
        c_gj++;  
      }
    }
    //c_gj = ( ( go_x - 1 ) * go_y ) * 2 - 1;
    for( int x = 0; x < go_x; x++ ) {
      for( int y = 0; y < go_y - 1 ; y++ ) { 
        pre_connect_gj [ c_gj * 2     ] = ( ( y     ) * go_x + x ) * GO_COMP;
        pre_connect_gj [ c_gj * 2 + 1 ] = ( ( y + 1 ) * go_x + x ) * GO_COMP;
        c_gj++;
       }
    }
    for ( int i = 0; i < n_gj; i += 2 )
    {
      post_connect_gj [ i + 1 ] = pre_connect_gj [ i ];
      post_connect_gj [ i ] = pre_connect_gj [ i + 1 ];
    }
  }
  if ( c_gj == 0 ) { printf ( "without GO gj\n" ); }
  else if ( c_gj * 2 == n_gj ) { printf( "GO gj_num = success \n" ); }
  else { printf ( "GO gj_num = error \n" ); exit ( 1 ); }  
  
  // count number of nonzero elements NNZ
  int nnz = GO_COMP;
  for ( int i = 0; i < GO_COMP * GO_COMP; i++ ) { nnz += ( mat [ i ] > 0.0 ); } 
  go_solve -> nnz = nnz * ( go -> n ) + n_gj;  printf ( "GO -> nnz = %d\n", go_solve -> nnz );
  go_solve -> val     = ( double * ) malloc ( go_solve -> nnz          * sizeof ( double ) ); // nonzero and diagonal elements array
  go_solve -> val_ori = ( double * ) malloc ( go_solve -> nnz          * sizeof ( double ) );
  go_solve -> b       = ( double * ) malloc ( GO_COMP * go -> n         * sizeof ( double ) ); // b value
  go_solve -> dammy   = ( double * ) malloc ( GO_COMP * go -> n         * sizeof ( double ) ); // calculation vec
  go_solve -> col     = ( int *    ) malloc ( go_solve -> nnz          * sizeof ( int ) ); //column number of val's elements
  go_solve -> row     = ( int *    ) malloc ( ( GO_COMP * go -> n + 1 ) * sizeof ( int ) ); //row index
  go_solve -> dig     = ( int *    ) malloc ( GO_COMP * go -> n         * sizeof ( int ) ); //dig index
  sprintf ( go_solve -> type, "%s", type ); 
  
  double *val     = go_solve -> val;
  double *val_ori = go_solve -> val_ori;
  double *b       = go_solve -> b;
  double *dammy   = go_solve -> dammy;
  int *col  = go_solve -> col; 
  int *row  = go_solve -> row;
  int *dig  = go_solve -> dig;
  
  for ( int i = 0; i < go_solve -> nnz; i++ )     { val [ i ] = val_ori [ i ] = 0.0; col [ i ] = 0; }
  for ( int i = 0; i < GO_COMP * go -> n + 1; i++) { row [ i ] = 0; }
  for ( int i = 0; i < GO_COMP * go -> n; i++)     { dig [ i ] = 0; b [ i ] = 0; dammy [ i ] = 0.0; }
        
  // Create CSR
  int num_row = 0, num_col = 0, num_dig = 0, num = 0, count_row = 0;
  //int count_gj = 0;
  
  for ( int i = 0; i < GO_COMP * GO_COMP; i++ ) 
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
    if ( num_col == GO_COMP ) 
    {
      for (int j = num_row * GO_COMP; j < num_row * GO_COMP + GO_COMP; j++ ) 
      {
        if ( j != num_row * GO_COMP + num_row ) {
          val [ num_dig ] += mat [ j ];
          dig [ num_row ] = num_dig;
        }
      }
      num_col = 0;
      num_row++;
      row [ num_row ] = count_row;
    }
  }
  
  for ( int i = 1; i < go -> n; i++ ) 
  {
    for ( int j = 0; j < nnz; j++ ) 
    {
      val [ j + nnz * i ] = val [ j ];
      //val_ori [ j + nnz * i ] = val [ j ];
      col [ j + nnz * i ] = col [ j ] + GO_COMP * i;
    }
    for ( int j = 0; j < GO_COMP; j++ ) 
    {
      row [ j + GO_COMP * i ] = row [ j ] + nnz * i;
      dig [ j + GO_COMP * i ] = dig [ j ] + nnz * i;
    }
  }
  row [ GO_COMP * go -> n ] = go_solve -> nnz;  
    
  // Adding GJ val
  for ( int i = 0; i < n_gj; i++ ) 
  { 
    val [ ( go_solve -> nnz ) - n_gj + i ] = - G_GJ_GO; 
    val [ dig [ pre_connect_gj [ i ] ] ] += G_GJ_GO;
    col [ ( go_solve -> nnz ) - n_gj + i ] = post_connect_gj [ i ]; 
  }
    
  // sort
  for ( int i = 0; i < n_gj; i++ ) 
  {
    int target = ( go_solve -> nnz ) - n_gj + i;
    int post_gj = post_connect_gj [ i ];//52
    int pre_gj  =  pre_connect_gj [ i ];//2
    int point = -1;
  
    if ( col [ row [ pre_gj + 1 ] - 1 ] < post_gj ) { point = row [ pre_gj + 1 ]; }
    else for ( int j = row [ pre_gj ]; j < row [ pre_gj + 1 ]; j++ ) {
      if ( col [ j ] > post_gj ) { point = j; break; }      
    }
    
    if ( point < 0 ) { printf(" GO gj [ %d ] sort error \n ", i );  exit(1);  }
  
    int    col_target = col [ target ]; 
    double val_target = val [ target ];
    for (int j = target; j >= point; j-- ) 
    {
      col [ j ] = col [ j - 1 ];
      val [ j ] = val [ j - 1 ];
    }
    col [ point ] = col_target;
    val [ point ] = val_target;
  
    if ( dig [ pre_gj ] >= point ) { for ( int j = pre_gj; j < go -> nc; j++ ) { dig [ j ] ++; } }
    else { for ( int j = pre_gj + 1; j < go -> nc; j++ ) { dig [ j ] ++; } }  
    for ( int j = pre_gj + 1; j < go -> nc; j++ ) row [ j ] ++;  
  }
  
  for ( int i = 0; i < go_solve -> nnz; i++ ) 
  {  
    //
    if ( ( 0 == strncmp ( go_solve -> type, "FORWARD_EULER", 13 ) ) ||
         ( 0 == strncmp ( go_solve -> type, "RUNGE_KUTTA_4", 13 ) ) ||
         ( 0 == strncmp ( go_solve -> type, "RKC", 3 ) ) ) { val [ i ] *= -1; }
    //
    if ( 0 == strncmp ( go_solve -> type, "CN", 2 ) ) { val [ i ] /= 2.0; }

    val_ori [ i ] = val [ i ];
  } // nnz is OK.
    
  free ( mat );
  free ( l_connect );
  free ( pre_connect_gj );
  free ( post_connect_gj );
  
  return go_solve;
}

__global__
static void device_mem_allocation ( const int nc, const int l_nnz, neuron_solve_t* d_go_solve,
     double *d_val, double *d_val_ori, double *d_b, int *d_col, int *d_row, int *d_dig, double *d_dammy ) 
{
  d_go_solve -> nnz     = l_nnz;
  d_go_solve -> val     = d_val;
  d_go_solve -> val_ori = d_val_ori;
  d_go_solve -> b       = d_b;
  d_go_solve -> col     = d_col;
  d_go_solve -> row     = d_row;   // # of neurons
  d_go_solve -> dig     = d_dig; // # of all compartments
  d_go_solve -> dammy   = d_dammy;
  d_go_solve -> numThreadsPerBlock = 128;
  d_go_solve -> numBlocks = ( int ) ( nc / d_go_solve -> numThreadsPerBlock ) + 1;
  //Debug
  //printf ( "From GPU \n n = %d, nc = %d\n", d_go -> n, d_go -> nc );
}

__global__
static void device_mem_allocation2 (const int n, double ** dev, double *ptr )
{
  dev [ n ] = ptr;
}

__global__ static
void device_mem_allocation3 ( neuron_solve_t* d_go_solve, double **d_vec ) 
{
  d_go_solve -> vec = d_vec;
}

neuron_solve_t *go_solve_initialize ( neuron_solve_t *p_go_solve, const char *type, neuron_t *go, neuron_t *d_go ) // tentatively, type is ignored
{
  neuron_solve_t *d_go_solve;
  cudaMalloc ( ( neuron_solve_t **) &d_go_solve, sizeof ( neuron_solve_t ) );
  if ( go -> nc == 0 ) { return d_go_solve; }

  neuron_solve_t *h_go_solve = set_host_go_solve ( type, go );  
  
  int l_nnz = h_go_solve -> nnz;
  double *d_val, *d_val_ori, *d_b, *d_dammy;
  cudaMalloc ( ( double ** ) &d_val,     l_nnz * sizeof ( double ) );
  cudaMalloc ( ( double ** ) &d_val_ori, l_nnz * sizeof ( double ) );
  cudaMalloc ( ( double ** ) &d_b,       go -> nc * sizeof ( double ) );
  cudaMalloc ( ( double ** ) &d_dammy,   go -> nc * sizeof ( double ) );
  int *d_col, *d_row, *d_dig;
  cudaMalloc ( ( int ** ) &d_col, l_nnz * sizeof ( int ) );
  cudaMalloc ( ( int ** ) &d_row, ( GO_COMP * go -> n + 1 ) * sizeof ( int ) );
  cudaMalloc ( ( int ** ) &d_dig, ( GO_COMP * go -> n     ) * sizeof ( int ) );
  
  p_go_solve -> val     = d_val;
  p_go_solve -> val_ori = d_val_ori;
  p_go_solve -> b       = d_b;
  p_go_solve -> dammy   = d_dammy;
  p_go_solve -> col = d_col;
  p_go_solve -> row = d_row;
  p_go_solve -> dig = d_dig;
  p_go_solve -> nnz = l_nnz;
  p_go_solve -> numThreadsPerBlock = 128;
  p_go_solve -> numBlocks = ( int ) ( ( go -> nc ) / p_go_solve -> numThreadsPerBlock ) + 1;
  sprintf ( p_go_solve -> type, "%s", type );

  cudaDeviceSynchronize ( );
  device_mem_allocation <<< 1, 1 >>> ( go -> nc, l_nnz, d_go_solve, d_val, d_val_ori, d_b, d_col, d_row, d_dig, d_dammy );
  cudaDeviceSynchronize ( );

  cudaMemcpy ( p_go_solve ->     val, h_go_solve ->     val, l_nnz * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_go_solve -> val_ori, h_go_solve -> val_ori, l_nnz * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_go_solve ->       b, h_go_solve ->       b, go -> nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_go_solve ->   dammy, h_go_solve ->   dammy, go -> nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_go_solve ->     col, h_go_solve ->     col, l_nnz * sizeof ( int ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_go_solve ->     row, h_go_solve ->     row, ( GO_COMP * go -> n + 1 ) * sizeof ( int ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_go_solve ->     dig, h_go_solve ->     dig, ( GO_COMP * go -> n     ) * sizeof ( int ), cudaMemcpyHostToDevice );

  //Debug
  //for ( int i = 0; i < l_nnz; i++ ) 
  //  printf ( "host val [ %d ] = %f\n", i, h_go_solve -> val [ i ] );
  //test_mem_allocation  <<< 1, 1 >>> ( d_go_solve );  
  cudaDeviceSynchronize ( );

  free ( h_go_solve -> val );  free ( h_go_solve -> val_ori );  free ( h_go_solve -> b   );
  free ( h_go_solve -> col );  free ( h_go_solve -> row );      free ( h_go_solve -> dig );
  free ( h_go_solve -> dammy );

  // set vec
  double **d_vec;
  if ( 0 == strncmp ( p_go_solve -> type, "RKC", 3 ) ) { 
    cudaMalloc ( ( double *** ) &d_vec, n_vec_RKC * sizeof ( double * ) );
    p_go_solve -> vec = ( double ** ) malloc ( n_vec_RKC * sizeof ( double * ) );
    for ( int i = 0; i < n_vec_RKC; i++ ) {
      cudaMalloc ( ( double ** ) ( & ( p_go_solve -> vec [ i ] ) ), go -> nc * sizeof ( double ) );
      device_mem_allocation2 <<< 1, 1 >>> ( i, d_vec, p_go_solve -> vec [ i ] );
    }
    device_mem_allocation3 <<< 1, 1 >>> ( d_go_solve, d_vec );
    RKC_vec_initialize (  d_go, d_go_solve, go, p_go_solve );
  }
  /**/
  else if ( 0 == strncmp ( p_go_solve -> type, "CN", 2 ) ) { 
    cudaMalloc ( ( double *** ) &d_vec, n_vec_CNm * sizeof ( double * ) );
    p_go_solve -> vec = ( double ** ) malloc ( n_vec_CNm * sizeof ( double * ) );
    for ( int i = 0; i < n_vec_CNm; i++ ) {
      cudaMalloc ( ( double ** ) ( & ( p_go_solve -> vec [ i ] ) ), go -> nc * sizeof ( double ) );
      device_mem_allocation2 <<< 1, 1 >>> ( i, d_vec, p_go_solve -> vec [ i ] );
    }
    device_mem_allocation3 <<< 1, 1 >>> ( d_go_solve, d_vec );
    go_cnm_vec_initialize <<< p_go_solve -> numBlocks, p_go_solve -> numThreadsPerBlock >>> ( d_go, d_go_solve );
    go_update_ion <<< p_go_solve -> numBlocks, p_go_solve -> numThreadsPerBlock >>> ( d_go, d_go_solve, CN_DT );
    double **l_ion  = go -> ion;
    go_KAHP_update <<< p_go_solve -> numBlocks, p_go_solve -> numThreadsPerBlock >>> 
     ( go -> n, go -> elem [ Ca ], l_ion [ o1_KAHP_go ], l_ion [ o2_KAHP_go ], 
      l_ion [ c1_KAHP_go ], l_ion [ c2_KAHP_go ], l_ion [ c3_KAHP_go ], l_ion [ c4_KAHP_go ], CN_DT );  // Debug
    //printf ("\n");
  }
  free ( h_go_solve );

  return d_go_solve;
}

void go_solve_finalize ( const int n_go, neuron_solve_t *d_go_solve, neuron_solve_t *p_go_solve )
{
  if ( n_go > 0 ) {
    cudaFree ( p_go_solve -> val );
    cudaFree ( p_go_solve -> val_ori );
    cudaFree ( p_go_solve -> b   );
    cudaFree ( p_go_solve -> dammy );
    cudaFree ( p_go_solve -> col );
    cudaFree ( p_go_solve -> row );
    cudaFree ( p_go_solve -> dig );
    if ( 0 == strncmp ( p_go_solve -> type, "RKC", 3 ) ) 
    { 
      for ( int i = 0; i < n_vec_RKC; i++ ) { cudaFree ( p_go_solve -> vec [ i ] ); }   
      free ( p_go_solve -> vec );
      free ( p_go_solve -> h_work );
      free ( p_go_solve -> h_others );
      free ( p_go_solve -> h_bool );
    } 
    else if ( 0 == strncmp ( p_go_solve -> type, "CN", 2 ) ) 
    { 
      for ( int i = 0; i < n_vec_CNm; i++ ) { cudaFree ( p_go_solve -> vec [ i ] ); }
      free ( p_go_solve -> vec );
    }
  }
  cudaFree ( d_go_solve ); free ( p_go_solve );
}
__host__
void go_solve_update_v ( neuron_t *d_go, neuron_solve_t *d_go_solve, neuron_t *p_go, neuron_solve_t *p_go_solve, synapse_t *d_grgo )
{
  if      ( 0 == strncmp ( p_go_solve -> type, "BE", 2 ) ) { 
    go_solve_by_bem ( d_go, d_go_solve, p_go, p_go_solve, d_grgo );
  }
  else if ( 0 == strncmp ( p_go_solve -> type, "CN", 2 ) ) { 
    go_solve_by_cnm ( d_go, d_go_solve, p_go, p_go_solve, d_grgo );
  }
  else if ( 0 == strncmp ( p_go_solve -> type, "RKC", 3 ) ) { 
    go_solve_by_rkc ( d_go, d_go_solve, p_go, p_go_solve, d_grgo );
  }
  else { printf ( "solver Error\n" ); }
}
