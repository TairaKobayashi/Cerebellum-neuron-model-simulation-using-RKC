#include "gr.cuh"

__host__ static void initialize_host_gr ( neuron_t *gr )
{ 
  int nc = gr -> nc; // # of all compartments
  gr -> elem = ( double ** ) malloc ( n_elem * sizeof ( double *) );
  for ( int i = 0; i < n_elem; i++ ) {
    gr -> elem [ i ] = ( double * ) malloc ( nc * sizeof ( double ) );
  }

  gr -> cond = ( double ** ) malloc ( gr_n_cond * sizeof ( double *) );
  for ( int i = 0; i < gr_n_cond; i++ ) {
    gr -> cond [ i ] = ( double * ) malloc ( nc * sizeof ( double ) );
  }

  gr -> ion = ( double ** ) malloc ( gr_n_ion * sizeof ( double *) );
  for ( int i = 0; i < gr_n_ion; i++ ) {
    gr -> ion [ i ] = ( double * ) malloc ( nc * sizeof ( double ) );
  }
  
  gr -> ca2 = ( double * ) malloc ( 1 * sizeof ( double ) );
  gr -> rev_ca2 = ( double * ) malloc ( 1 * sizeof ( double ) );
  
  double rad [ GR_COMP ], len [ GR_COMP ];// Ra [ gr_COMP ];

  FILE *file = fopen ( PARAM_FILE_GR, "r" );
  if ( ! file ) { fprintf ( stderr, "no such file %s\n", PARAM_FILE_GR ); exit ( 1 ); }

  for ( int i = 0; i < GR_COMP; i++ ) {
    int i1, i2, i3;
    double d1, d2, d3, d4, i_l1, i_l2, i_l3, i_Na, i_KV, i_KA, i_KCa, i_KM, i_KIR, i_Ca;

    if ( fscanf ( file, "%d %d %lf %lf %lf %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf ",
      &i1, &i2, &d1, &d2, &d3, &d4, &i3, &i_l1, &i_l2, &i_l3, 
      &i_Na, &i_KV, &i_KA, &i_KCa, &i_KM, &i_KIR, &i_Ca ) == ( EOF ) ) {
      printf ( "GR_PARAM_FILE_READING_ERROR\n" );
      exit ( 1 );
    }
    rad [ i1 ] = 0.5 * d1 * 1e-4; // [mum -> cm]
    len [ i1 ] = d2 * 1e-4; // [mum -> cm]
    
    gr -> elem [ connect ] [ i1 ] = ( double ) i2;
    gr -> elem [ compart ] [ i1 ] = ( double ) i3;
    gr -> elem [ area    ] [ i1 ] = 2.0 * M_PI * rad [ i1 ] * len [ i1 ]; // [cm^2]
    double area_val = gr -> elem [ area ] [ i1 ];
    gr -> elem [ Cm      ] [ i1 ] = d4    * area_val; // [muF]
    //gr -> elem [ i_ext   ] [ i1 ] = 0.0;
    gr -> cond [ g_leak1 ] [ i1 ] = i_l1  * area_val;
    gr -> cond [ g_leak2 ] [ i1 ] = i_l2  * area_val;
    gr -> cond [ g_leak3 ] [ i1 ] = i_l3  * area_val;
    gr -> cond [ g_Na    ] [ i1 ] = i_Na  * area_val;
    gr -> cond [ g_KV    ] [ i1 ] = i_KV  * area_val;
    gr -> cond [ g_KA    ] [ i1 ] = i_KA  * area_val;
    gr -> cond [ g_KCa   ] [ i1 ] = i_KCa * area_val;
    gr -> cond [ g_KM    ] [ i1 ] = i_KM  * area_val;
    gr -> cond [ g_KIR   ] [ i1 ] = i_KIR * area_val;
    gr -> cond [ g_Ca    ] [ i1 ] = i_Ca  * area_val;
  }
  fclose ( file );

  for ( int i = 1; i < gr -> n; i++ ) {
    for ( int j = 0; j < GR_COMP; j++ ) {
      gr -> elem [ connect ] [ j + GR_COMP * i ] 
            = ( gr -> elem [ connect ] [ j + GR_COMP * i ] > 0 ?
						    gr -> elem [ connect ] [ j ] + GR_COMP * i : -1 );
      gr -> elem [ compart ] [ j + GR_COMP * i ] = gr -> elem [ compart ] [ j ];
      gr -> elem [ Cm      ] [ j + GR_COMP * i ] = gr -> elem [ Cm      ] [ j ];
      gr -> elem [ area    ] [ j + GR_COMP * i ] = gr -> elem [ area    ] [ j ];
      //gr -> elem [ i_ext   ] [ j + GR_COMP * i ] = gr -> elem [ i_ext   ] [ j ];
      gr -> cond [ g_leak1 ] [ j + GR_COMP * i ] = gr -> cond [ g_leak1 ] [ j ];
      gr -> cond [ g_leak2 ] [ j + GR_COMP * i ] = gr -> cond [ g_leak2 ] [ j ];
      gr -> cond [ g_leak3 ] [ j + GR_COMP * i ] = gr -> cond [ g_leak3 ] [ j ];
      gr -> cond [ g_Na  ] [ j + GR_COMP * i ] = gr -> cond [ g_Na  ] [ j ];
      gr -> cond [ g_KV  ] [ j + GR_COMP * i ] = gr -> cond [ g_KV  ] [ j ];
      gr -> cond [ g_KA  ] [ j + GR_COMP * i ] = gr -> cond [ g_KA  ] [ j ];
      gr -> cond [ g_KCa ] [ j + GR_COMP * i ] = gr -> cond [ g_KCa ] [ j ];
      gr -> cond [ g_KM  ] [ j + GR_COMP * i ] = gr -> cond [ g_KM  ] [ j ];
      gr -> cond [ g_KIR ] [ j + GR_COMP * i ] = gr -> cond [ g_KIR ] [ j ];
      gr -> cond [ g_Ca  ] [ j + GR_COMP * i ] = gr -> cond [ g_Ca  ] [ j ];
    }
  }
}

__host__
static void finalize_host_gr ( neuron_t *gr )
{  
  if ( ( gr -> n ) > 0 )
  {
    for ( int i = 0; i < n_elem; i++ ) { free ( gr -> elem [ i ] ); }
    free ( gr -> elem );

    for ( int i = 0; i < gr_n_cond; i++ ) { free ( gr -> cond [ i ] ); }
    free ( gr -> cond );

    for ( int i = 0; i < gr_n_ion;  i++ ) { free ( gr -> ion [ i ] ); }
    free ( gr -> ion );

    free ( gr -> ca2 );
    free ( gr -> rev_ca2 );
  }
  free ( gr );
}

__global__
static void device_mem_allocation ( const double DT, const int nx, const int ny, neuron_t* d_gr, double **d_elem, double **d_cond, double **d_ion,
                                    double *d_ca2, double *d_rev ) 
{
  d_gr -> elem    = d_elem;
  d_gr -> cond    = d_cond;
  d_gr -> ion     = d_ion;
  d_gr -> ca2     = d_ca2;
  d_gr -> rev_ca2 = d_rev;
  d_gr -> nx = nx; // gr_X 
  d_gr -> ny = ny; // gr_Y
  int n = nx * ny; int nc = n * GR_COMP;
  d_gr -> n = n;   // # of neurons
  d_gr -> nc = nc; // # of all compartments
  d_gr -> DT = DT;
  //Debug
  printf ( "gr -> n = %d, nc = %d\n", d_gr -> n, d_gr -> nc );
}

__global__
static void device_mem_allocation2 (const int n, double ** dev, double *ptr )
{
  dev [ n ] = ptr;
}

__host__ static void memcpy_neuron ( neuron_t *p_gr, neuron_t *h_gr )
{
  int nc = h_gr -> nc;
  cudaMemcpy ( p_gr -> elem [ v     ], h_gr -> elem [ v     ], nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_gr -> elem [ Ca    ], h_gr -> elem [ Ca    ], nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_gr -> elem [ Cm    ], h_gr -> elem [ Cm    ], nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_gr -> elem [ area  ], h_gr -> elem [ area  ], nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_gr -> elem [ i_ext ], h_gr -> elem [ i_ext ], nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_gr -> elem [ connect ], h_gr -> elem [ connect ], nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_gr -> elem [ compart ], h_gr -> elem [ compart ], nc * sizeof ( double ), cudaMemcpyHostToDevice );
  
  for ( int i = 0; i < gr_n_ion; i++ )
    cudaMemcpy ( p_gr -> ion [ i ], h_gr -> ion [ i ], nc * sizeof ( double ), cudaMemcpyHostToDevice );
  for ( int i = 0; i < gr_n_cond; i++ )
    cudaMemcpy ( p_gr -> cond [ i ], h_gr -> cond [ i ], nc * sizeof ( double ), cudaMemcpyHostToDevice );

  cudaMemcpy ( p_gr -> rev_ca2, h_gr -> rev_ca2, 1 * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_gr -> ca2,     h_gr -> ca2,     1 * sizeof ( double ), cudaMemcpyHostToDevice );

  cudaDeviceSynchronize ();
} 

__host__
neuron_t *gr_initialize ( const int nx, const int ny, const char *type, neuron_t *p_gr )
{
  p_gr -> nx = nx; // gr_X 
  p_gr -> ny = ny; // gr_Y
  int n = nx * ny; int nc = n * GR_COMP;
  p_gr -> n = n; // # of neurons
  p_gr -> nc = nc; // # of all compartments
  p_gr -> neuron_type = N_TYPE_GR;
  
  // set DT
  if      ( 0 == strncmp ( type, "BE", 2 ) ) { p_gr -> DT = BE_DT;  p_gr -> n_solver = kBE; }
  else if ( 0 == strncmp ( type, "CN", 2 ) ) { p_gr -> DT = CN_DT;  p_gr -> n_solver = kCN; }
  else if ( 0 == strncmp ( type, "RKC", 3 ) ) { p_gr -> DT = H_MAX; p_gr -> n_solver = kRKC; }
  else { printf ("error in gr_initialize\n"); exit ( 1 ); }

  neuron_t *d_gr;  
  cudaMalloc ( ( neuron_t ** ) &d_gr, sizeof ( neuron_t ) );

  if ( n == 0 ) { printf ( "gr -> n = 0\n" ); return d_gr; }
  
  /**/
  double **d_elem;
  cudaMalloc ( ( double *** ) &d_elem, n_elem * sizeof ( double * ) );
  double **d_cond;
  cudaMalloc ( ( double *** ) &d_cond, gr_n_cond * sizeof ( double * ) );
  double **d_ion;
  cudaMalloc ( ( double *** ) &d_ion , gr_n_ion  * sizeof ( double * ) );
  double *d_ca2; // don't use
  cudaMalloc ( ( double **  ) &d_ca2 , 1 * sizeof ( double ) );
  double *d_rev; // don't use
  cudaMalloc ( ( double **  ) &d_rev , 1 * sizeof ( double ) );  
  
  p_gr -> ca2  = d_ca2;
  p_gr -> rev_ca2 = d_rev;
  
  p_gr -> elem = ( double ** ) malloc ( n_elem * sizeof ( double * ) );
  for ( int i = 0; i < n_elem; i++ ) {
    cudaMalloc ( ( double ** ) ( & ( p_gr -> elem [ i ] ) ), nc * sizeof ( double ) );
    device_mem_allocation2 <<< 1, 1 >>> ( i, d_elem, p_gr -> elem [ i ] );
  }
  p_gr -> cond = ( double ** ) malloc ( gr_n_cond * sizeof ( double * ) );
  for ( int i = 0; i < gr_n_cond; i++ ) {
    cudaMalloc ( ( double ** ) ( & ( p_gr -> cond [ i ] ) ), nc * sizeof ( double ) );
    device_mem_allocation2 <<< 1, 1 >>> ( i, d_cond, p_gr -> cond [ i ] );
  }
  p_gr -> ion = ( double ** ) malloc ( gr_n_ion * sizeof ( double * ) );
  for ( int i = 0; i < gr_n_ion; i++ ) {
    cudaMalloc ( ( double ** ) ( & ( p_gr -> ion [ i ] ) ), nc * sizeof ( double ) );
    device_mem_allocation2 <<< 1, 1 >>> ( i, d_ion, p_gr -> ion [ i ] );
  }

  device_mem_allocation <<< 1, 1 >>> ( p_gr -> DT, nx, ny, d_gr, d_elem, d_cond, d_ion, d_ca2, d_rev );
  cudaDeviceSynchronize();
  
  // set temporary host gr
  neuron_t *h_gr = ( neuron_t * ) malloc ( sizeof ( neuron_t ) );
  h_gr -> nx = nx; h_gr -> ny = ny;
  h_gr -> n = n; h_gr -> nc = nc;
  initialize_host_gr ( h_gr );
  gr_initialize_ion  ( h_gr );
  
  // copy host gr -> device gr
  memcpy_neuron ( p_gr, h_gr );
  finalize_host_gr ( h_gr );
 
  return d_gr;
}

__host__
void gr_finalize ( neuron_t *p_gr, neuron_t *d_gr, FILE *f_out, FILE *f_out_raster )
{
  if ( p_gr -> n > 0 )
  {
    for ( int i = 0; i < n_elem; i++ ) { cudaFree ( p_gr -> elem [ i ] ); }
    for ( int i = 0; i < gr_n_cond; i++ ) { cudaFree ( p_gr -> cond [ i ] ); }
    for ( int i = 0; i < gr_n_ion;  i++ ) { cudaFree ( p_gr -> ion [ i ] ); }
    free ( p_gr -> elem  );
    free ( p_gr -> cond  );
    free ( p_gr -> ion  );
    cudaFree ( p_gr -> ca2 );
    cudaFree ( p_gr -> rev_ca2 );
    
    cudaMemcpy ( p_gr, d_gr, sizeof ( neuron_t ), cudaMemcpyDeviceToHost );
    cudaFree ( p_gr -> elem );
    cudaFree ( p_gr -> cond );
    cudaFree ( p_gr -> ion  );        
  }
  fclose ( f_out );
  fclose ( f_out_raster );
  free ( p_gr );
  cudaFree ( d_gr );
}

__global__ void gr_set_current ( neuron_t *d_gr, const double t )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < d_gr -> n )
  { 
    double *i_ext_vals = & d_gr -> elem [ i_ext ] [ 16 + id * GR_COMP ];
    *i_ext_vals = 0;
    if ( 50 <= t && t < 100 ) { *i_ext_vals = 0.01e-3; }     
  }
}

__host__
void gr_output_file ( neuron_t *gr, double *memv_test, double *memv_raster, const double t, FILE *f_out, FILE *f_out_raster )
{
  FILE *f = f_out;
  FILE *f_raster = f_out_raster;

  fprintf ( f, "%lf,", t );
  for ( int j = 0; j < gr -> n; j++ ) {
    fprintf ( f, "%lf,", memv_test [ 16 + j * GR_COMP ] ); // soma
  }
  fprintf ( f, "\n" );
  
  //fprintf ( f_raster, "%lf,", t );
  for ( int j = 0; j < gr -> n; j++ ) {
    if ( memv_test [ 77 + j * GR_COMP ] > 10.0 && memv_raster [ j ] <= 10.0 )
      fprintf ( f_raster, "%lf,%d\n", t, j );
    //else 
      //fprintf ( f_raster, "," );
    memv_raster [ j ] = memv_test [ 77 + j * GR_COMP ];
  }
  //fprintf ( f_raster, "\n" );
}
