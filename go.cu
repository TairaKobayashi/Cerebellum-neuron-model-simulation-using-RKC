#include "go.cuh"

__host__ static void initialize_host_go ( neuron_t *go )
{ 
  int nc = go -> nc; // # of all compartments
  go -> elem = ( double ** ) malloc ( n_elem * sizeof ( double *) );
  for ( int i = 0; i < n_elem; i++ ) {
    go -> elem [ i ] = ( double * ) malloc ( nc * sizeof ( double ) );
  }

  go -> cond = ( double ** ) malloc ( go_n_cond * sizeof ( double *) );
  for ( int i = 0; i < go_n_cond; i++ ) {
    go -> cond [ i ] = ( double * ) malloc ( nc * sizeof ( double ) );
  }

  go -> ion = ( double ** ) malloc ( go_n_ion * sizeof ( double *) );
  for ( int i = 0; i < go_n_ion; i++ ) {
    go -> ion [ i ] = ( double * ) malloc ( nc * sizeof ( double ) );
  }
  
  go -> ca2 = ( double * ) malloc ( nc * sizeof ( double ) );
  go -> rev_ca2 = ( double * ) malloc ( nc * sizeof ( double ) );
  go -> ca_old  = ( double * ) malloc ( nc * sizeof ( double ) );

  double rad [ GO_COMP ], len [ GO_COMP ];// Ra [ GO_COMP ];

  FILE *file = fopen ( PARAM_FILE_GO, "r" );
  if ( ! file ) { fprintf ( stderr, "no such file %s\n", PARAM_FILE_GO ); exit ( 1 ); }

  for ( int i = 0; i < GO_COMP; i++ ) {
    int i1, i2, i3;
    double d1, d2, d3, d4, i_leak, i_NaT, i_NaR, i_NaP, i_KV, i_KA, i_KC;
    double i_Kslow, i_CaHVA, i_CaLVA, i_HCN1, i_HCN2, i_KAHP;

    if ( fscanf ( file, "%d %d %lf %lf %lf %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
        &i1, &i2, &d1, &d2, &d3, &d4, &i3, &i_leak, &i_NaT, &i_NaR, &i_NaP, &i_KV, &i_KA, &i_KC,
        &i_Kslow, &i_CaHVA, &i_CaLVA, &i_HCN1, &i_HCN2, &i_KAHP ) == ( EOF ) ){
        printf ( "GO_PARAM_FILE_READING_ERROR\n" );
        exit ( 1 );
    }

    rad [ i1 ] = 0.5 * d1 * 1e-4; // [mum -> cm]
    len [ i1 ] = d2 * 1e-4; // [mum -> cm]

    go -> elem [ connect ] [ i1 ] = i2;
    go -> elem [ compart ] [ i1 ] = i3;
    go -> elem [ area    ] [ i1 ] = 2.0 * M_PI * rad [ i1 ] * len [ i1 ]; // [cm^2]
    double area_val = go -> elem [ area ] [ i1 ];
    go -> elem [ Cm      ] [ i1 ] = d4      * area_val; // [muF]
    go -> cond [ g_leak_go  ] [ i1 ] = i_leak  * area_val;    
    go -> cond [ g_NaT_go   ] [ i1 ] = i_NaT   * area_val;
    go -> cond [ g_NaR_go   ] [ i1 ] = i_NaR   * area_val;
    go -> cond [ g_NaP_go   ] [ i1 ] = i_NaP   * area_val;
    go -> cond [ g_KV_go    ] [ i1 ] = i_KV    * area_val;
    go -> cond [ g_KA_go    ] [ i1 ] = i_KA    * area_val;
    go -> cond [ g_KC_go    ] [ i1 ] = i_KC    * area_val;
    go -> cond [ g_Kslow_go ] [ i1 ] = i_Kslow * area_val;
    go -> cond [ g_CaHVA_go ] [ i1 ] = i_CaHVA * area_val;
    go -> cond [ g_CaLVA_go ] [ i1 ] = i_CaLVA * area_val;
    go -> cond [ g_HCN1_go  ] [ i1 ] = i_HCN1  * area_val;
    go -> cond [ g_HCN2_go  ] [ i1 ] = i_HCN2  * area_val;
    go -> cond [ g_KAHP_go  ] [ i1 ] = i_KAHP  * area_val;
  }
  fclose ( file );

  for ( int i = 1; i < go -> n; i++ ) {
    for ( int j = 0; j < GO_COMP; j++ ) {
      go -> elem [ connect ] [ j + GO_COMP * i ] = ( go -> elem [ connect] [ j + GO_COMP * i ] > 0 ?
						    go -> elem [ connect ] [ j ] + GO_COMP * i : -1 );
      go -> elem [ compart ] [ j + GO_COMP * i ] = go -> elem [ compart ] [ j ];
      go -> elem [ Cm      ] [ j + GO_COMP * i ] = go -> elem [ Cm      ] [ j ];
      go -> elem [ area    ] [ j + GO_COMP * i ] = go -> elem [ area    ] [ j ];
      go -> cond [ g_leak_go  ] [ j + GO_COMP * i ] = go -> cond [ g_leak_go  ] [ j ];
      go -> cond [ g_NaT_go   ] [ j + GO_COMP * i ] = go -> cond [ g_NaT_go   ] [ j ];
      go -> cond [ g_NaR_go   ] [ j + GO_COMP * i ] = go -> cond [ g_NaR_go   ] [ j ];
      go -> cond [ g_NaP_go   ] [ j + GO_COMP * i ] = go -> cond [ g_NaP_go   ] [ j ];
      go -> cond [ g_KV_go    ] [ j + GO_COMP * i ] = go -> cond [ g_KV_go    ] [ j ];
      go -> cond [ g_KA_go    ] [ j + GO_COMP * i ] = go -> cond [ g_KA_go    ] [ j ];
      go -> cond [ g_KC_go    ] [ j + GO_COMP * i ] = go -> cond [ g_KC_go    ] [ j ];
      go -> cond [ g_Kslow_go ] [ j + GO_COMP * i ] = go -> cond [ g_Kslow_go ] [ j ];
      go -> cond [ g_CaHVA_go ] [ j + GO_COMP * i ] = go -> cond [ g_CaHVA_go ] [ j ];
      go -> cond [ g_CaLVA_go ] [ j + GO_COMP * i ] = go -> cond [ g_CaLVA_go ] [ j ];
      go -> cond [ g_HCN1_go  ] [ j + GO_COMP * i ] = go -> cond [ g_HCN1_go  ] [ j ];
      go -> cond [ g_HCN2_go  ] [ j + GO_COMP * i ] = go -> cond [ g_HCN2_go  ] [ j ];
      go -> cond [ g_KAHP_go  ] [ j + GO_COMP * i ] = go -> cond [ g_KAHP_go  ] [ j ];
    }
  }
}

__host__
static void finalize_host_go ( neuron_t *go )
{  
  if ( ( go -> n ) > 0 )
  {
    for ( int i = 0; i < n_elem; i++ ) { free ( go -> elem [ i ] ); }
    free ( go -> elem );

    for ( int i = 0; i < go_n_cond; i++ ) { free ( go -> cond [ i ] ); }
    free ( go -> cond );

    for ( int i = 0; i < go_n_ion;  i++ ) { free ( go -> ion [ i ] ); }
    free ( go -> ion );

    free ( go -> ca2 );
    free ( go -> rev_ca2 );
    free ( go -> ca_old );
  }
  free ( go );
}

__global__
static void device_mem_allocation ( const double DT, const int nx, const int ny, neuron_t* d_go, double **d_elem, double **d_cond, double **d_ion,
     double *d_ca2, double *d_rev, double *d_ca_old ) 
{
  d_go -> elem    = d_elem;
  d_go -> cond    = d_cond;
  d_go -> ion     = d_ion;
  d_go -> ca2     = d_ca2;
  d_go -> rev_ca2 = d_rev;
  d_go -> ca_old  = d_ca_old;
  d_go -> nx = nx; // GO_X 
  d_go -> ny = ny; // GO_Y
  int n = nx * ny; int nc = n * GO_COMP;
  d_go -> n = n;   // # of neurons
  d_go -> nc = nc; // # of all compartments
  d_go -> DT = DT;
  //Debug
  printf ( "go -> n = %d, nc = %d\n", d_go -> n, d_go -> nc );
}

__global__
static void device_mem_allocation2 (const int n, double ** dev, double *ptr )
{
  dev [ n ] = ptr;
}

__host__ static void memcpy_neuron ( neuron_t *p_go, neuron_t *h_go )
{
  int nc = h_go -> nc;
  cudaMemcpy ( p_go -> elem [ v     ], h_go -> elem [ v     ], nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_go -> elem [ Ca    ], h_go -> elem [ Ca    ], nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_go -> elem [ Cm    ], h_go -> elem [ Cm    ], nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_go -> elem [ area  ], h_go -> elem [ area  ], nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_go -> elem [ i_ext ], h_go -> elem [ i_ext ], nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_go -> elem [ connect ], h_go -> elem [ connect ], nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_go -> elem [ compart ], h_go -> elem [ compart ], nc * sizeof ( double ), cudaMemcpyHostToDevice );
  
  for ( int i = 0; i < go_n_ion; i++ )
    cudaMemcpy ( p_go -> ion [ i ], h_go -> ion [ i ], nc * sizeof ( double ), cudaMemcpyHostToDevice );
  for ( int i = 0; i < go_n_cond; i++ )
    cudaMemcpy ( p_go -> cond [ i ], h_go -> cond [ i ], nc * sizeof ( double ), cudaMemcpyHostToDevice );

  cudaMemcpy ( p_go -> ca2,     h_go -> ca2,     nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_go -> rev_ca2, h_go -> rev_ca2, nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_go -> ca_old , h_go -> ca_old , nc * sizeof ( double ), cudaMemcpyHostToDevice );

  cudaDeviceSynchronize ();
} 

__host__
neuron_t *go_initialize ( const int nx, const int ny, const char *type, neuron_t *p_go )
{
  p_go -> nx = nx; // GO_X 
  p_go -> ny = ny; // GO_Y
  int n = nx * ny; int nc = n * GO_COMP;
  p_go -> n = n; // # of neurons
  p_go -> nc = nc; // # of all compartments
  p_go -> neuron_type = N_TYPE_GO;  
  
  // set DT
  if      ( 0 == strncmp ( type, "BE", 2 ) ) { p_go -> DT = BE_DT;  p_go -> n_solver = kBE; }
  else if ( 0 == strncmp ( type, "CN", 2 ) ) { p_go -> DT = CN_DT;  p_go -> n_solver = kCN; }
  else if ( 0 == strncmp ( type, "RKC", 3 ) ) { p_go -> DT = H_MAX; p_go -> n_solver = kRKC; }
  else { printf ("error in go_initialize\n"); exit ( 1 ); }

  neuron_t *d_go;  
  cudaMalloc ( ( neuron_t ** ) &d_go, sizeof ( neuron_t ) );

  if ( n == 0 ) { printf ( "go -> n = 0\n" ); return d_go; }
  
  /**/
  double **d_elem;
  cudaMalloc ( ( double *** ) &d_elem, n_elem * sizeof ( double * ) );
  double **d_cond;
  cudaMalloc ( ( double *** ) &d_cond, go_n_cond * sizeof ( double * ) );
  double **d_ion;
  cudaMalloc ( ( double *** ) &d_ion , go_n_ion  * sizeof ( double * ) );
  double *d_ca2;
  cudaMalloc ( ( double **  ) &d_ca2 , nc * sizeof ( double ) );
  double *d_rev;
  cudaMalloc ( ( double **  ) &d_rev , nc * sizeof ( double ) );  
  double *d_ca_old;
  cudaMalloc ( ( double **  ) &d_ca_old , nc * sizeof ( double ) );  

  p_go -> ca2  = d_ca2;
  p_go -> rev_ca2 = d_rev;
  p_go -> ca_old  = d_ca_old;

  p_go -> elem = ( double ** ) malloc ( n_elem * sizeof ( double * ) );
  for ( int i = 0; i < n_elem; i++ ) {
    cudaMalloc ( ( double ** ) ( & ( p_go -> elem [ i ] ) ), nc * sizeof ( double ) );
    device_mem_allocation2 <<< 1, 1 >>> ( i, d_elem, p_go -> elem [ i ] );
  }
  p_go -> cond = ( double ** ) malloc ( go_n_cond * sizeof ( double * ) );
  for ( int i = 0; i < go_n_cond; i++ ) {
    cudaMalloc ( ( double ** ) ( & ( p_go -> cond [ i ] ) ), nc * sizeof ( double ) );
    device_mem_allocation2 <<< 1, 1 >>> ( i, d_cond, p_go -> cond [ i ] );
  }
  p_go -> ion = ( double ** ) malloc ( go_n_ion * sizeof ( double * ) );
  for ( int i = 0; i < go_n_ion; i++ ) {
    cudaMalloc ( ( double ** ) ( & ( p_go -> ion [ i ] ) ), nc * sizeof ( double ) );
    device_mem_allocation2 <<< 1, 1 >>> ( i, d_ion, p_go -> ion [ i ] );
  }

  device_mem_allocation <<< 1, 1 >>> ( p_go -> DT, nx, ny, d_go, d_elem, d_cond, d_ion, d_ca2, d_rev, d_ca_old );
  cudaDeviceSynchronize();
  
  // set temporary host go
  neuron_t *h_go = ( neuron_t * ) malloc ( sizeof ( neuron_t ) );
  h_go -> nx = nx; h_go -> ny = ny;
  h_go -> n = n; h_go -> nc = nc;
  initialize_host_go ( h_go );
  go_initialize_ion  ( h_go );
  
  // copy host go -> device go
  memcpy_neuron ( p_go, h_go );
  finalize_host_go ( h_go );
 
  return d_go;
}

__host__
void go_finalize ( neuron_t *p_go, neuron_t *d_go, FILE *f_out, FILE *f_out_raster )
{
  if ( p_go -> n > 0 )
  {
    for ( int i = 0; i < n_elem; i++ ) { cudaFree ( p_go -> elem [ i ] ); }
    for ( int i = 0; i < go_n_cond; i++ ) { cudaFree ( p_go -> cond [ i ] ); }
    for ( int i = 0; i < go_n_ion;  i++ ) { cudaFree ( p_go -> ion [ i ] ); }
    free ( p_go -> elem  );
    free ( p_go -> cond  );
    free ( p_go -> ion  );
    cudaFree ( p_go -> ca2 );
    cudaFree ( p_go -> rev_ca2 );
    cudaFree ( p_go -> ca_old  );

    cudaMemcpy ( p_go, d_go, sizeof ( neuron_t ), cudaMemcpyDeviceToHost );
    cudaFree ( p_go -> elem );
    cudaFree ( p_go -> cond );
    cudaFree ( p_go -> ion  );
  }
  fclose ( f_out );
  fclose ( f_out_raster );
  free ( p_go );
  cudaFree ( d_go );
}

__global__ void go_set_current ( neuron_t *d_go, const double t )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < d_go -> n )
  { 
    double *i_ext_val = & d_go -> elem [ i_ext ] [ GO_COMP_DEND * 3 + id * GO_COMP ];
    *i_ext_val = 0;
    // fig6
    if ( ( 800 <= t && t < 1100 )  ) { *i_ext_val = 0.2e-3; }
  }
}

__host__
void go_output_file ( neuron_t *go, double *memv_test, double *memv_raster, const double t, FILE *f_out, FILE *f_out_raster )
{
  
  FILE *f = f_out;
  FILE *f_raster = f_out_raster;
  
  
  fprintf ( f, "%lf,", t );
  for ( int j = 0; j < go -> n; j++ ) {
    fprintf ( f, "%lf,", memv_test [ GO_COMP_DEND * 3 + j * GO_COMP ] ); //soma
  }
  fprintf ( f, "\n" );
  
  //fprintf ( f_raster, "%lf,", t );
  for ( int j = 0; j < go -> n; j++ ) {
    if ( memv_test [ GO_COMP_DEND * 3 + j * GO_COMP ] > 10.0 && memv_raster [ j ] <= 10.0 )
      fprintf ( f_raster, "%lf,%d\n", t, j );
    //else 
    //  fprintf ( f_raster, "," );
    memv_raster [ j ] = memv_test [ GO_COMP_DEND * 3 + j * GO_COMP ];
  }
  //fprintf ( f_raster, "\n" );
}
