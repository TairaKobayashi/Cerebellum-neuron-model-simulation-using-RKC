#include "io.cuh"

__host__ static void initialize_host_io ( neuron_t *io )
{ 
  int nc = io -> nc; // # of all compartments
  io -> elem = ( double ** ) malloc ( n_elem * sizeof ( double *) );
  for ( int i = 0; i < n_elem; i++ ) {
    io -> elem [ i ] = ( double * ) malloc ( nc * sizeof ( double ) );
  }

  io -> cond = ( double ** ) malloc ( io_n_cond * sizeof ( double *) );
  for ( int i = 0; i < io_n_cond; i++ ) {
    io -> cond [ i ] = ( double * ) malloc ( nc * sizeof ( double ) );
  }

  io -> ion = ( double ** ) malloc ( io_n_ion * sizeof ( double *) );
  for ( int i = 0; i < io_n_ion; i++ ) {
    io -> ion [ i ] = ( double * ) malloc ( nc * sizeof ( double ) );
  }
    
  double rad [ IO_COMP ], len [ IO_COMP ];// Ra [ io_COMP ];

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
    rad [ i1 ] = 0.5 * d1 * 1e-4; // [mum -> cm]
    len [ i1 ] = d2 * 1e-4; // [mum -> cm]
    
    io -> elem [ connect ] [ i1 ] = ( double ) i2;
    io -> elem [ compart ] [ i1 ] = ( double ) i3;
    io -> elem [ area    ] [ i1 ] = 2.0 * M_PI * rad [ i1 ] * len [ i1 ]; // [cm^2]
    //double area_val = io -> elem [ area ] [ i1 ];
    io -> elem [ Cm      ] [ i1 ] = d4; // [muF / cm^2]
    //io -> elem [ i_ext   ] [ i1 ] = 0.0;
    if ( i3 == 0 )  // dend
    {
      io -> cond [ g_leak_io ] [ i1 ] = G_LEAK_IO / ( 1 * 1.0 ); // no need area
      io -> cond [ g_CaL_io  ] [ i1 ] = 0.0;
      io -> cond [ g_Na_io   ] [ i1 ] = 0.0;
      io -> cond [ g_Kdr_io  ] [ i1 ] = 0.0;
      io -> cond [ g_K_io    ] [ i1 ] = 0.0;
      io -> cond [ g_CaH_io  ] [ i1 ] = DNDR_G_CAH_IO / ( 1 * 1.0 );
      io -> cond [ g_KCa_io  ] [ i1 ] = DNDR_G_KCA_IO / ( 1 * 1.0 );
      io -> cond [ g_H_io    ] [ i1 ] = DNDR_G_H_IO / ( 1 * 1.0 );
      //io -> cond [ g_H_io    ] [ i1 ] = 0.0;
    }    
    else if ( i3 == 1 )  // soma
    {
      io -> cond [ g_leak_io ] [ i1 ] = G_LEAK_IO; // no need area
      io -> cond [ g_CaL_io  ] [ i1 ] = SOMA_G_CAL_IO;
      io -> cond [ g_Na_io   ] [ i1 ] = SOMA_G_NA_IO;
      io -> cond [ g_Kdr_io  ] [ i1 ] = SOMA_G_KDR_IO;
      io -> cond [ g_K_io    ] [ i1 ] = SOMA_G_K_IO;
      io -> cond [ g_CaH_io  ] [ i1 ] = 0.0;
      io -> cond [ g_KCa_io  ] [ i1 ] = 0.0;
      io -> cond [ g_H_io    ] [ i1 ] = 0.0;
    }    
    else if ( i3 == 2 )  // axon
    {
      io -> cond [ g_leak_io ] [ i1 ] = G_LEAK_IO; // no need area
      io -> cond [ g_CaL_io  ] [ i1 ] = 0.0;
      io -> cond [ g_Na_io   ] [ i1 ] = AXON_G_NA_IO;
      io -> cond [ g_Kdr_io  ] [ i1 ] = 0.0;
      io -> cond [ g_K_io    ] [ i1 ] = AXON_G_K_IO;
      io -> cond [ g_CaH_io  ] [ i1 ] = 0.0;
      io -> cond [ g_KCa_io  ] [ i1 ] = 0.0;
      io -> cond [ g_H_io    ] [ i1 ] = 0.0;
      //io -> cond [ g_H_io    ] [ i1 ] = DNDR_G_H_IO;
    }
    else { printf ( "\n\nio.cu error!!!!! \n\n\n" ); }
  }
  fclose ( file );

  for ( int i = 1; i < io -> n; i++ ) {
    for ( int j = 0; j < IO_COMP; j++ ) {
      io -> elem [ connect ] [ j + IO_COMP * i ] 
            = ( io -> elem [ connect ] [ j + IO_COMP * i ] > 0 ?
						    io -> elem [ connect ] [ j ] + IO_COMP * i : -1 );
      io -> elem [ compart ] [ j + IO_COMP * i ] = io -> elem [ compart ] [ j ];
      io -> elem [ Cm      ] [ j + IO_COMP * i ] = io -> elem [ Cm      ] [ j ];
      io -> elem [ area    ] [ j + IO_COMP * i ] = io -> elem [ area    ] [ j ];
      //io -> elem [ i_ext   ] [ j + IO_COMP * i ] = io -> elem [ i_ext   ] [ j ];
      io -> cond [ g_leak_io ] [ j + IO_COMP * i ] = io -> cond [ g_leak_io ] [ j ];
      io -> cond [ g_CaL_io  ] [ j + IO_COMP * i ] = io -> cond [ g_CaL_io  ] [ j ];
      io -> cond [ g_Na_io   ] [ j + IO_COMP * i ] = io -> cond [ g_Na_io   ] [ j ];
      io -> cond [ g_Kdr_io  ] [ j + IO_COMP * i ] = io -> cond [ g_Kdr_io  ] [ j ];
      io -> cond [ g_K_io    ] [ j + IO_COMP * i ] = io -> cond [ g_K_io    ] [ j ];
      io -> cond [ g_CaH_io  ] [ j + IO_COMP * i ] = io -> cond [ g_CaH_io  ] [ j ];
      io -> cond [ g_KCa_io  ] [ j + IO_COMP * i ] = io -> cond [ g_KCa_io  ] [ j ];
      io -> cond [ g_H_io    ] [ j + IO_COMP * i ] = io -> cond [ g_H_io   ] [ j ];
    }
  }
}

__host__
static void finalize_host_io ( neuron_t *io )
{  
  if ( ( io -> n ) > 0 )
  {
    for ( int i = 0; i < n_elem; i++ ) { free ( io -> elem [ i ] ); }
    free ( io -> elem );

    for ( int i = 0; i < io_n_cond; i++ ) { free ( io -> cond [ i ] ); }
    free ( io -> cond );

    for ( int i = 0; i < io_n_ion;  i++ ) { free ( io -> ion [ i ] ); }
    free ( io -> ion );
  }
  free ( io );
}

__global__
static void device_mem_allocation ( const double DT, const int nx, const int ny, neuron_t* d_io, double **d_elem, double **d_cond, double **d_ion ) 
{
  d_io -> elem    = d_elem;
  d_io -> cond    = d_cond;
  d_io -> ion     = d_ion;
  d_io -> nx = nx; // io_X 
  d_io -> ny = ny; // io_Y
  int n = nx * ny; int nc = n * IO_COMP;
  d_io -> n = n;   // # of neurons
  d_io -> nc = nc; // # of all compartments
  d_io -> DT = DT;
  //Debug
  printf ( "io -> n = %d, nc = %d\n", d_io -> n, d_io -> nc );
}

__global__
static void device_mem_allocation2 (const int n, double ** dev, double *ptr )
{
  dev [ n ] = ptr;
}

__host__ static void memcpy_neuron ( neuron_t *p_io, neuron_t *h_io )
{
  int nc = h_io -> nc;
  cudaMemcpy ( p_io -> elem [ v     ], h_io -> elem [ v     ], nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_io -> elem [ Ca    ], h_io -> elem [ Ca    ], nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_io -> elem [ Cm    ], h_io -> elem [ Cm    ], nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_io -> elem [ area  ], h_io -> elem [ area  ], nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_io -> elem [ i_ext ], h_io -> elem [ i_ext ], nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_io -> elem [ connect ], h_io -> elem [ connect ], nc * sizeof ( double ), cudaMemcpyHostToDevice );
  cudaMemcpy ( p_io -> elem [ compart ], h_io -> elem [ compart ], nc * sizeof ( double ), cudaMemcpyHostToDevice );
  
  for ( int i = 0; i < io_n_ion; i++ )
    cudaMemcpy ( p_io -> ion [ i ], h_io -> ion [ i ], nc * sizeof ( double ), cudaMemcpyHostToDevice );
  for ( int i = 0; i < io_n_cond; i++ )
    cudaMemcpy ( p_io -> cond [ i ], h_io -> cond [ i ], nc * sizeof ( double ), cudaMemcpyHostToDevice );

  cudaDeviceSynchronize ();
} 

__host__
neuron_t *io_initialize ( const int nx, const int ny, const char *type, neuron_t *p_io )
{
  p_io -> nx = nx; // io_X 
  p_io -> ny = ny; // io_Y
  int n = nx * ny; int nc = n * IO_COMP;
  p_io -> n = n; // # of neurons
  p_io -> nc = nc; // # of all compartments
  p_io -> neuron_type = N_TYPE_IO;
  
  // set DT
  if      ( 0 == strncmp ( type, "BE", 2 ) ) { p_io -> DT = BE_DT;  p_io -> n_solver = kBE; }
  else if ( 0 == strncmp ( type, "CN", 2 ) ) { p_io -> DT = CN_DT;  p_io -> n_solver = kCN; }
  else if ( 0 == strncmp ( type, "RKC", 3 ) ) { p_io -> DT = H_MAX; p_io -> n_solver = kRKC; }
  else { printf ("error in io_initialize\n"); exit ( 1 ); }

  neuron_t *d_io;  
  cudaMalloc ( ( neuron_t ** ) &d_io, sizeof ( neuron_t ) );

  if ( n == 0 ) { printf ( "io -> n = 0\n" ); return d_io; }
  
  /**/
  double **d_elem;
  cudaMalloc ( ( double *** ) &d_elem, n_elem * sizeof ( double * ) );
  double **d_cond;
  cudaMalloc ( ( double *** ) &d_cond, io_n_cond * sizeof ( double * ) );
  double **d_ion;
  cudaMalloc ( ( double *** ) &d_ion , io_n_ion  * sizeof ( double * ) );
     
  p_io -> elem = ( double ** ) malloc ( n_elem * sizeof ( double * ) );
  for ( int i = 0; i < n_elem; i++ ) {
    cudaMalloc ( ( double ** ) ( & ( p_io -> elem [ i ] ) ), nc * sizeof ( double ) );
    device_mem_allocation2 <<< 1, 1 >>> ( i, d_elem, p_io -> elem [ i ] );
  }
  p_io -> cond = ( double ** ) malloc ( io_n_cond * sizeof ( double * ) );
  for ( int i = 0; i < io_n_cond; i++ ) {
    cudaMalloc ( ( double ** ) ( & ( p_io -> cond [ i ] ) ), nc * sizeof ( double ) );
    device_mem_allocation2 <<< 1, 1 >>> ( i, d_cond, p_io -> cond [ i ] );
  }
  p_io -> ion = ( double ** ) malloc ( io_n_ion * sizeof ( double * ) );
  for ( int i = 0; i < io_n_ion; i++ ) {
    cudaMalloc ( ( double ** ) ( & ( p_io -> ion [ i ] ) ), nc * sizeof ( double ) );
    device_mem_allocation2 <<< 1, 1 >>> ( i, d_ion, p_io -> ion [ i ] );
  }

  device_mem_allocation <<< 1, 1 >>> ( p_io -> DT, nx, ny, d_io, d_elem, d_cond, d_ion );
  cudaDeviceSynchronize();
  
  // set temporary host io
  neuron_t *h_io = ( neuron_t * ) malloc ( sizeof ( neuron_t ) );
  h_io -> nx = nx; h_io -> ny = ny;
  h_io -> n = n; h_io -> nc = nc;
  initialize_host_io ( h_io );
  io_initialize_ion  ( h_io );
  
  // copy host io -> device io
  memcpy_neuron ( p_io, h_io );
  finalize_host_io ( h_io );
 
  return d_io;
}

__host__
void io_finalize ( neuron_t *p_io, neuron_t *d_io, FILE *f_out, FILE *f_out_raster )
{
  if ( p_io -> n > 0 )
  {
    for ( int i = 0; i < n_elem; i++ ) { cudaFree ( p_io -> elem [ i ] ); }
    for ( int i = 0; i < io_n_cond; i++ ) { cudaFree ( p_io -> cond [ i ] ); }
    for ( int i = 0; i < io_n_ion;  i++ ) { cudaFree ( p_io -> ion [ i ] ); }
    free ( p_io -> elem  );
    free ( p_io -> cond  );
    free ( p_io -> ion  );
    
    cudaMemcpy ( p_io, d_io, sizeof ( neuron_t ), cudaMemcpyDeviceToHost );
    cudaFree ( p_io -> elem );
    cudaFree ( p_io -> cond );
    cudaFree ( p_io -> ion  );        
  }
  fclose ( f_out );
  fclose ( f_out_raster );
  free ( p_io );
  cudaFree ( d_io );
}

__global__ void io_set_current ( neuron_t *d_io, const double t )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < d_io -> n )
  { 
    double *i_ext_vals = & d_io -> elem [ i_ext ] [ 0 + id * IO_COMP ];
    *i_ext_vals = 0; 
    // fig6
    if ( ( 800 <= t && t < 820 )  ) { *i_ext_vals = 5.5; }
  }
}

__host__
void io_output_file ( neuron_t *io, double *memv_test, double *memv_raster, const double t, FILE *f_out, FILE *f_out_raster )
{
  FILE *f = f_out;
  FILE *f_raster = f_out_raster;

  fprintf ( f, "%lf,", t );
  
  // Debug
  //fprintf ( f, "%lf,", memv_test [ IO_COMP_DEND + IO_COMP_SOMA + ( io -> n / 2 ) * IO_COMP ] );
 
  for ( int j = 0; j < io -> n; j++ ) {
    fprintf ( f, "%lf,", memv_test [ 0 + j * IO_COMP ] );
    fprintf ( f, "%lf,", memv_test [ IO_COMP_DEND + j * IO_COMP ] );
    fprintf ( f, "%lf,", memv_test [ IO_COMP_DEND + IO_COMP_SOMA + j * IO_COMP ] );
  }
  fprintf ( f, "\n" );
  
  //fprintf ( f_raster, "%lf,", t );
  for ( int j = 0; j < io -> n; j++ ) {
    if ( memv_test [ IO_COMP_DEND + IO_COMP_SOMA + j * IO_COMP ] > 10.0 && memv_raster [ j ] <= 10.0 )
      fprintf ( f_raster, "%lf,%d\n", t, j );
    //else 
      //fprintf ( f_raster, "," );
    memv_raster [ j ] = memv_test [ IO_COMP_DEND + IO_COMP_SOMA + j * IO_COMP ];
  }
  //fprintf ( f_raster, "\n" );
}
