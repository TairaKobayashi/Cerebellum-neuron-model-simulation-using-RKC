#include "solve_rkc.cuh"
#include "reduction.h"

//#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <cusparse.h>
#include <math.h>
#include <assert.h>

#define FMAX_ORI(x,y) (((x) >(y))?(x):(y))
#define ERR_ESTIMATE_C (1)

#define NumOfWork (8)

__global__ static 
void reset_b ( double *d_, int id_max )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < id_max ) { d_ [ id ] = 0.0; }
}

__global__ static
void rand_elements_change ( double *v, double *y, int id )
{
  // vec [ vtemp1 ] [ index ] = vec [ yn_r ] [ index ] - (vec [ vtemp1 ] [ index ] - vec [ yn_r ] [ index ] );
  v [ id ] = 2.0 * y [ id ] - v [ id ];
}


__host__ __device__ static 
int nint ( const double x ) {//四捨五入
	if ( x - ( double ) ( int ) x < 0.5) { return ( ( double ) ( int ) x ); }
	else { return ( ( double ) ( int ) ( x + 0.9 ) ); }
}

__host__ __device__ static 
double fmax_ori( const double a, const double b ) {
    return ( a > b )? a: b;
}

__global__ static
void err_estimate_atomic ( double *result, const double l_at, const double l_rt, const double *d_y,
                           const double *d_yn_r, const double *d_fn, const double *d_vtemp1, double h, int nc )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < nc) 
  {
    double wt = l_at + l_rt * FMAX_ORI ( fabs ( d_y [ id ] ), fabs ( d_yn_r [ id ] ) );
    double est = 0.8 *  ( d_yn_r[ id ] - d_y [ id ] ) + 0.4 * h * ( d_fn [ id ] + d_vtemp1 [ id ] );
    if ( wt == 0.0 ) {
        printf ( "assert: from err_estimate: idid = 3, global_threadId = %d\n", id );
        assert ( 0 );
    }
    atomicAdd ( result, ( est / wt ) * ( est / wt )  );
  }
}

__global__ static
void err_estimate_first_half ( double *d_input, const double l_at, const double l_rt, const double *d_y, 
                               const double *d_yn_r, const double *d_fn, const double *d_vtemp1, double h, int nc )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < nc) 
  {
    double wt = l_at + l_rt * FMAX_ORI( fabs ( d_y [ id ] ), fabs ( d_yn_r [ id ] ) );
    double est = 0.8 *  ( d_yn_r[ id ] - d_y [ id ] ) + 0.4 * h * ( d_fn [ id ] + d_vtemp1 [ id ] );
    if ( wt == 0.0 ) {
        printf ( "assert: from err_estimate: idid = 3, global_threadId = %d\n", id );
        assert ( 0 );
    }
    d_input [ id ] = ( est / wt ) * ( est / wt );  
  }
}


__host__ static
void err_estimate_pr ( double *result, double *h_others, double h, neuron_solve_t *p_solve, int nc ) 
{
  static double *d_input, *d_output;
  static int size = 0, count = 0;
  double **vec = p_solve -> vec;
  if ( nc > size ) {
    if ( size != 0 ) {
      cudaFree ( d_output );
      cudaFree ( d_input );
    }
    fprintf ( stderr,"count: %d\n", ++count );
    cudaMalloc ( &d_input, sizeof ( double ) * nc );
    cudaMalloc ( &d_output, sizeof ( double ) * nc );
    size = nc;
  }
  err_estimate_first_half <<< p_solve -> numBlocks, p_solve -> numThreadsPerBlock >>> 
      ( d_input, h_others [ atol_r ], h_others [ rtol ], vec [ y ], vec [ yn_r ], vec [ fn ], vec [ vtemp1 ], h, nc );
  //( d_input, p_go_solve -> h_others [ atol_r ], p_go_solve -> h_others [ rtol ], vec [ y ], vec [ yn_r ], vec [ fn ], vec [ vtemp1 ], h, nc );

   ParallelReduction < double > ( nc, d_input, d_output, result );
   //ParallelReduction < double > ( size, d_input, d_output, result );
  return;
}

__host__ static
double err_estimate ( double *h_others, double h, neuron_solve_t *p_solve, int nc )
{
  double **vec = p_solve -> vec;
  static double *result;
  double h_result = 0;
  static bool firstCall = true;
  if(firstCall){
    //cudaMallocManaged( &result, sizeof(double));
    cudaMallocHost( &result, sizeof(double));
    firstCall = false;
  }
  //result[0] = 0;
  switch ( ERR_ESTIMATE_C )
  {
    case 0:
          err_estimate_atomic <<< p_solve -> numBlocks, p_solve -> numThreadsPerBlock >>> 
            ( result, h_others [ atol_r ], h_others [ rtol ], vec [ y ], vec [ yn_r ], vec [ fn ], vec [ vtemp1 ], h, nc );
            //( result, p_go_solve -> h_others [ atol_r ], p_go_solve -> h_others [ rtol ], vec [ y ], vec [ yn_r ], vec [ fn ], vec [ vtemp1 ], h, nc );
          cudaDeviceSynchronize(); // cudaMallocManaged + cudaDeviceSynchronize or cudaMemcpy( 1 element ). Which is faster than the other?
          break;
    case 1:
          err_estimate_pr ( result, h_others, h, p_solve, nc );
          break;
  }
  //cudaDeviceSynchronize();
  return result [ 0 ];
}

__global__ static 
void Hadamard_product ( neuron_t *d_neuron, neuron_solve_t *d_solve )
{
  double **vec  = d_solve -> vec;  
  double *dammy = d_solve -> dammy;
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < d_neuron -> nc ) 
  {
    double wt = dammy [ i ];
    dammy [ i ] = ( vec [ vtemp2 ] [ i ] - vec [ fn ] [ i ] ) * wt;    
  }
}

__global__ static
void vec_sA_plus_B ( const double s, const double *A, const double *B, double *C, const int nc )
{
  int id = threadIdx.x + blockDim.x*blockIdx.x;
  if ( id < nc ) { C [ id ] = s * A [ id ] + B [ id ]; }
}

__global__
void vec_sA ( const double s, const double *A, double *C, const int nc )
{
  int id = threadIdx.x + blockDim.x*blockIdx.x;
  if( id < nc ) { C [ id ] =  s * A [ id ]; }
}

__global__
void set_s2A ( const double s, double *A, const int nc )
{
  int id = threadIdx.x + blockDim.x*blockIdx.x;
  if( id < nc ) { A [ id ] = s; }
}

__global__
void calcY ( const double mu, const double *d_vtemp1, const double nu, const double *d_vtemp2, 
             const double one_mu_nu, const double *d_yn_r, const double h_mus, double *d_y, 
            const double ajm1, const double *d_fn, const int nc )
{
  int id = threadIdx.x + blockDim.x*blockIdx.x;
  if( id < nc ) 
  { 
    d_y [ id ] = mu * d_vtemp1 [ id ] + nu * d_vtemp2 [ id ] + one_mu_nu * d_yn_r [ id ]
              + h_mus * ( d_y [ id ] - ajm1 * d_fn [ id ] ); 
  }
}

__global__ static
void dydt ( const double* __restrict__ y, double*  dy, neuron_t *d_neuron, neuron_solve_t *d_solve )
{ 
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < d_neuron -> nc) 
  {
    double *val = d_solve -> val;
    int *col  = d_solve -> col;
    int *row  = d_solve -> row;
    //double *cm  = d_neuron -> elem [ Cm ];
    double d_ = ( - ( d_solve -> b [ id ] ) );
    const double y_id = y [ id ];

    for ( int j = row [ id ]; j < row [ id + 1 ]; j++ ) 
    {
      d_ += val [ j ] * ( y [ col [ j ] ]  - y_id );
    }     
    dy [ id ] = d_ / d_neuron -> elem [ Cm ] [ id ]; 
  }
}

/////////////////////////****************** GRANULE **********************/////////////////////////

__global__ static
void add_mfgr_val ( neuron_solve_t *d_gr_solve, double *elem_v, int *mfgr_comp, 
                    double *mfgr_elem, const int num_mfgr )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < num_mfgr ) 
  {    
    double l_val = mfgr_elem [ mfgr_val  * num_mfgr + id ];
    int post_num = mfgr_comp [ post_comp * num_mfgr + id ];    
    atomicAdd ( & ( d_gr_solve -> b [ post_num ] ), ( l_val * ( elem_v [ post_num ] - E_MFGR ) ) );
  }
}

__global__ static
void add_gogr_val ( neuron_solve_t *d_gr_solve, double *elem_v, int *gogr_comp, 
                    double *gogr_elem, const int num_gogr )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < num_gogr ) 
  {    
    double l_val = gogr_elem [ gogr_val  * num_gogr + id ];
    int post_num = gogr_comp [ post_comp * num_gogr + id ];    
    atomicAdd ( & ( d_gr_solve -> b [ post_num ] ), ( l_val * ( elem_v [ post_num ] - E_GOGR ) ) );
  }
}

__global__ static
void gr_update_matrix ( neuron_t *d_gr, neuron_solve_t *d_gr_solve, double *elem_v )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  double **elem = d_gr -> elem;
  double **cond = d_gr -> cond;
  double **ion  = d_gr -> ion;
  if ( id < d_gr -> nc) 
  {
    double l_v =  elem_v [ id ];
    d_gr_solve -> b [ id ]    = (  
      + cond [ g_leak1 ] [ id ] * ( l_v - V_LEAK1_GR )
      + cond [ g_leak2 ] [ id ] * ( l_v - V_LEAK2_GR )
      + cond [ g_leak3 ] [ id ] * ( l_v - V_LEAK3_GR )
      - elem [ i_ext ] [ id ] + 
      + cond [ g_Na    ] [ id ] * ( l_v - V_Na_GR ) * ion [ o_Na   ] [ id ]
      + cond [ g_Ca    ] [ id ] * ( l_v - V_Ca_GR ) * ion [ ch_Ca  ] [ id ] * ion [ ch_Ca ] [ id ] * ion [ ci_Ca ] [ id ]
      + cond [ g_KV    ] [ id ] * ( l_v - V_K_GR )  * ion [ n_KV   ] [ id ] * ion [ n_KV  ] [ id ] * ion [ n_KV  ] [ id ] * ion [ n_KV ] [ id ]
      + cond [ g_KIR   ] [ id ] * ( l_v - V_K_GR )  * ion [ ir_KIR ] [ id ]
      + cond [ g_KA    ] [ id ] * ( l_v - V_K_GR )  * ion [ a_KA   ] [ id ] * ion [ a_KA  ] [ id ] * ion [ a_KA  ] [ id ] * ion [ b_KA ] [ id ]
      + cond [ g_KCa   ] [ id ] * ( l_v - V_K_GR )  * ion [ c_KCa  ] [ id ]
      + cond [ g_KM    ] [ id ] * ( l_v - V_K_GR )  * ion [ s_KM   ] [ id ] );
  }
}

/////////////////////////****************** GOLGI **********************/////////////////////////

__global__ static
void add_grgo_val ( neuron_solve_t *d_go_solve, double *elem_v, int *grgo_comp, 
                    double *grgo_elem, const int num_grgo )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < num_grgo ) 
  {    
    double l_val = grgo_elem [ grgo_val  * num_grgo + id ];
    int post_num = grgo_comp [ post_comp * num_grgo + id ];    
    atomicAdd ( & ( d_go_solve -> b [ post_num ] ), ( l_val * ( elem_v [ post_num ] - E_GRGO ) ) );
  }
}

__global__ static
void add_grgo_val_new ( double *vec_v_new, double *elem_v, int *grgo_comp, 
                    double *grgo_elem, const int num_grgo )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < num_grgo ) 
  {    
    double l_val = grgo_elem [ grgo_val  * num_grgo + id ];
    int post_num = grgo_comp [ post_comp * num_grgo + id ];    
    atomicAdd ( & ( vec_v_new [ post_num ] ), ( l_val * ( elem_v [ post_num ] - E_GRGO ) ) );
  }
}

__global__ static
void go_update_matrix_new ( neuron_solve_t *d_go_solve, double *vec_v_new, int nc )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < nc ) 
  {
    d_go_solve -> b [ id ] += vec_v_new [ id ];
  }
}
__global__ static
void go_update_matrix ( neuron_t *d_go, neuron_solve_t *d_go_solve, double *elem_v )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < d_go -> nc) 
  {
    double **cond = d_go -> cond;
    double **ion  = d_go -> ion;
    double l_v =  elem_v [ id ];
    d_go_solve -> b [ id ]    = (  
      - d_go -> elem [ i_ext ] [ id ] 
      + cond [ g_leak_go  ] [ id ] * ( l_v - V_LEAK_GO )
			+ cond [ g_NaT_go   ] [ id ] * ( l_v - V_Na_GO  ) * ion [ m_NaT_go ] [ id ] * ion [ m_NaT_go ] [ id ] * ion [ m_NaT_go ] [ id ] * ion [ h_NaT_go ] [ id ]
			+ cond [ g_NaR_go   ] [ id ] * ( l_v - V_Na_GO  ) * ion [ r_NaR_go ] [ id ] * ion [ s_NaR_go ] [ id ]
			+ cond [ g_NaP_go   ] [ id ] * ( l_v - V_Na_GO  ) * ion [ p_NaP_go ] [ id ]
			+ cond [ g_CaHVA_go ] [ id ] * ( l_v - V_Ca_GO  ) * ion [ ch_CaHVA_go ] [ id ] * ion [ ch_CaHVA_go ] [ id ] * ion [ ci_CaHVA_go ] [ id ]
			+ cond [ g_CaLVA_go ] [ id ] * ( l_v - d_go -> rev_ca2 [ id ] ) * ion [ cl_CaLVA_go ] [ id ] * ion [ cl_CaLVA_go ] [ id ] * ion [ cm_CaLVA_go ] [ id ]
			+ cond [ g_KAHP_go  ] [ id ] * ( l_v - V_K_GO   ) * ( ion [ o1_KAHP_go ] [ id ] + ion [ o2_KAHP_go ] [ id ] )
			+ cond [ g_KV_go    ] [ id ] * ( l_v - V_K_GO   ) * ion [ n_KV_go ] [ id ] * ion [ n_KV_go ] [ id ] * ion [ n_KV_go ] [ id ] * ion [ n_KV_go ] [ id ]
			+ cond [ g_KA_go    ] [ id ] * ( l_v - V_K_GO   ) * ion [ a_KA_go ] [ id ] * ion [ a_KA_go ] [ id ] * ion [ a_KA_go ] [ id ] * ion [ b_KA_go ] [ id ]
			+ cond [ g_KC_go    ] [ id ] * ( l_v - V_K_GO   ) * ion [ c_KC_go ] [ id ]
			+ cond [ g_Kslow_go ] [ id ] * ( l_v - V_K_GO   ) * ion [ sl_Kslow_go ] [ id ]
			+ cond [ g_HCN1_go  ] [ id ] * ( l_v - V_H_GO   ) * ( ion [ hf_HCN1_go ] [ id ] + ion [ hs_HCN1_go ] [ id ] )
			+ cond [ g_HCN2_go  ] [ id ] * ( l_v - V_H_GO   ) * ( ion [ hf_HCN2_go ] [ id ] + ion [ hs_HCN2_go ] [ id ] )
			);
  }
}

/////////////////////////****************** Purkinje **********************/////////////////////////

__global__ static
void add_mlipkj_val ( neuron_solve_t *d_pkj_solve, double *elem_v, int *mlipkj_comp, 
                    double *mlipkj_elem, const int num_mlipkj )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < num_mlipkj ) 
  {    
    double l_val = mlipkj_elem [ mlipkj_val * num_mlipkj + id ];
    int post_num = mlipkj_comp [ post_comp  * num_mlipkj + id ];    
    atomicAdd ( & ( d_pkj_solve -> b [ post_num ] ), ( l_val * ( elem_v [ post_num ] - E_MLIPKJ ) ) );
  }
}

__global__ static
void add_grpkj_val ( neuron_solve_t *d_pkj_solve, double *elem_v, int *grpkj_comp, 
                    double *grpkj_elem, const int num_grpkj )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < num_grpkj ) 
  {    
    double l_val = grpkj_elem [ grpkj_val * num_grpkj + id ];
    int post_num = grpkj_comp [ post_comp * num_grpkj + id ];    
    atomicAdd ( & ( d_pkj_solve -> b [ post_num ] ), ( l_val * ( elem_v [ post_num ] - E_GRPKJ ) ) );
  }
}

__global__ static
void add_grpkj_val_new ( double *vec_v_new, int *grpkj_comp, 
                    double *grpkj_elem, double *elem_v, const int num_grpkj )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < num_grpkj ) 
  {    
    double l_val = grpkj_elem [ grpkj_val * num_grpkj + id ];
    int post_num = grpkj_comp [ post_comp * num_grpkj + id ];    
    atomicAdd ( & ( vec_v_new [ post_num ] ), ( l_val  ) * ( elem_v [ post_num ] - E_GRPKJ ) );
    //atomicAdd ( & ( vec_v_new [ post_num ] ), ( l_val  ) );//* ( elem_v [ post_num ] - E_GRPKJ )
  }
}

__global__ static
void pkj_update_matrix_new (  neuron_solve_t *d_pkj_solve, const double *vec_v_new,
                              const int nc, double *elem_v, const double sum_decay )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < nc ) 
  {
    d_pkj_solve -> b [ id ] += vec_v_new [ id ] * sum_decay;// * ( elem_v [ id ] - E_GRPKJ );
  }
}

__global__ static
void pkj_update_matrix ( neuron_t *d_pkj, neuron_solve_t *d_pkj_solve, double *elem_v )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < d_pkj -> nc) 
  {
    double **cond = d_pkj -> cond;
    double **ion  = d_pkj -> ion;
    double v_ca2_pkj = d_pkj -> rev_ca2 [ id ];
    double l_v =  elem_v [ id ];
     
    d_pkj_solve -> b [ id ]    = ( 
      - d_pkj -> elem [ i_ext ] [ id ]
      + cond [ g_leak_pkj ] [ id ] * ( l_v - V_LEAK_PKJ )
      + cond [ g_NaF_pkj ] [ id ] * ion [ m_NaF_pkj ] [ id ] * ion [ m_NaF_pkj ] [ id ] * ion [ m_NaF_pkj ] [ id ] * ion [ h_NaF_pkj ] [ id ] * ( l_v - V_Na_PKJ )
      + cond [ g_NaP_pkj ] [ id ] * ion [ m_NaP_pkj ] [ id ] * ion [ m_NaP_pkj ] [ id ] * ion [ m_NaP_pkj ] [ id ] * ( l_v - V_Na_PKJ )
      + cond [ g_CaP_pkj ] [ id ] * ion [ m_CaP_pkj ] [ id ] * ion [ h_CaP_pkj ] [ id ] * ( l_v - v_ca2_pkj )
      + cond [ g_CaT_pkj ] [ id ] * ion [ m_CaT_pkj ] [ id ] * ion [ h_CaT_pkj ] [ id ] * ( l_v - v_ca2_pkj )
      + cond [ g_Kh_pkj ] [ id ] * ion [ m_Kh1_pkj ] [ id ] * ( l_v - V_KH_PKJ ) //KH????
      + cond [ g_Kh_pkj ] [ id ] * ion [ m_Kh2_pkj ] [ id ] * ( l_v - V_KH_PKJ ) //KH???
      + cond [ g_Kdr_pkj ] [ id ] * ion [ m_Kdr_pkj ] [ id ] * ion [ m_Kdr_pkj ] [ id ] * ion [ h_Kdr_pkj ] [ id ] * ( l_v - V_K_PKJ )
      + cond [ g_KM_pkj ] [ id ] * ion [ m_KM_pkj ] [ id ] * ( l_v - V_K_PKJ )
      + cond [ g_KA_pkj ] [ id ] * ion [ m_KA_pkj ] [ id ] * ion [ m_KA_pkj ] [ id ] * ion [ m_KA_pkj ] [ id ] * ion [ m_KA_pkj ] [ id ] * ion [ h_KA_pkj ] [ id ] * ( l_v - V_K_PKJ )
      + cond [ g_KC_pkj ] [ id ] * ion [ m_KC_pkj ] [ id ] * ion [ z_KC_pkj ] [ id ] * ion [ z_KC_pkj ] [ id ] * ( l_v - V_K_PKJ )
      + cond [ g_K2_pkj ] [ id ] * ion [ m_K2_pkj ] [ id ] * ion [ z_K2_pkj ] [ id ] * ion [ z_K2_pkj ] [ id ]  * ( l_v - V_K_PKJ ));
  }
}


/////////////////////////****************** IO **********************/////////////////////////

__global__ static
void add_io_gap_val ( neuron_solve_t *d_io_solve, int *io_gap_comp, double *io_gap_elem, const int num_io_gap )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < num_io_gap ) 
  {    
    double l_val = io_gap_elem [ gap_current * num_io_gap + id ];
    int post_num = io_gap_comp [ post_comp_gap  * num_io_gap + id ];    
    atomicAdd ( & ( d_io_solve -> b [ post_num ] ), ( l_val ) );
  }
}

__global__ static
void io_update_matrix ( neuron_t *d_io, neuron_solve_t *d_io_solve, double *elem_v )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < d_io -> nc) 
  {    
    double **elem = d_io -> elem;
    double **cond = d_io -> cond;
    double **ion  = d_io -> ion;
    double l_v =  elem_v [ id ];
    d_io_solve -> b [ id ]    = (  
      - elem [ i_ext ] [ id ] 
      + cond [ g_leak_io ] [ id ] * ( l_v - V_LEAK_IO )
			+ cond [ g_CaL_io  ] [ id ] * ( l_v - V_Ca_IO ) * ion [ k_CaL_io ] [ id ] * ion [ k_CaL_io ] [ id ] * ion [ k_CaL_io ] [ id ] * ion [ l_CaL_io ] [ id ]
      + cond [ g_Na_io   ] [ id ] * ( l_v - V_Na_IO ) * ion [ m_Na_io  ] [ id ] * ion [ m_Na_io  ] [ id ] * ion [ m_Na_io  ] [ id ] * ion [ h_Na_io  ] [ id ]
      + cond [ g_Kdr_io  ] [ id ] * ( l_v - V_K_IO  ) * ion [ n_Kdr_io ] [ id ] * ion [ p_Kdr_io ] [ id ] 
      + cond [ g_K_io    ] [ id ] * ( l_v - V_K_IO  ) * ion [ x_K_io   ] [ id ] * ion [ x_K_io   ] [ id ] * ion [ x_K_io   ] [ id ] * ion [ x_K_io   ] [ id ] 
      + cond [ g_CaH_io  ] [ id ] * ( l_v - V_Ca_IO ) * ion [ r_CaH_io ] [ id ] * ion [ r_CaH_io ] [ id ]
      + cond [ g_KCa_io  ] [ id ] * ( l_v - V_K_IO  ) * ion [ s_KCa_io ] [ id ]
      + cond [ g_H_io    ] [ id ] * ( l_v - V_H_IO  ) * ion [ q_H_io   ] [ id ]
    );
  }
}

//////////////////////*********** RKC ***************/////////////////////////
__host__ static
void set_initial_vec ( neuron_t *p_, neuron_solve_t *p_solve, 
                          neuron_t *d_, neuron_solve_t *d_solve,
                          cublasStatus_t stat1, cublasHandle_t handle1 )
{
  int nc = p_ -> nc;
  double **elem = p_ -> elem;
  double **vec  = p_solve -> vec;
  int numThreadsPerBlock = p_solve -> numThreadsPerBlock;
  int numBlocks = p_solve -> numBlocks;

  double l_h_min [ 4 ] = { H_MIN_GR, H_MIN_GO, H_MIN_PKJ, H_MIN_IO };
  double l_tol [ 4 ] = { TOL_GR, TOL_GO, TOL_PKJ, TOL_IO };

  double *h_nc;
  h_nc   = ( double * ) malloc ( nc * sizeof ( double ) );
  p_solve -> h_work   = ( double * ) malloc ( 8 * sizeof ( double ) );
  p_solve -> h_others = ( double * ) malloc ( n_vec_RKC_others * sizeof ( double ) );
  p_solve -> h_bool = ( bool * ) malloc ( n_h_bool * sizeof ( bool ) );
  for ( int i = 0; i < nc; i++ ) { h_nc   [ i ] = 0.0; }
  for ( int i = 0; i < 8; i++ ) { p_solve -> h_work [ i ] = 0.0; }
  for ( int i = 0; i < n_vec_RKC_others; i++ ) { p_solve -> h_others [ i ] = 0.0; }

  stat1 = cublasDcopy ( handle1, nc, elem [ v ], 1,  vec [ v_new  ], 1);
  stat1 = cublasDcopy ( handle1, nc, elem [ v ], 1,  vec [ y      ], 1);
  stat1 = cublasDcopy ( handle1, nc, elem [ v ], 1,  vec [ yn_r   ], 1);
  stat1 = cublasSetVector ( nc, sizeof ( double ), h_nc, 1, vec [ fn  ], 1 );
  stat1 = cublasSetVector ( nc, sizeof ( double ), h_nc, 1, vec [ ede ], 1 );
  stat1 = cublasSetVector ( nc, sizeof ( double ), h_nc, 1, vec [ vtemp1 ], 1 );
  stat1 = cublasSetVector ( nc, sizeof ( double ), h_nc, 1, vec [ vtemp2 ], 1 );
  p_solve -> h_work [ 2 ] = nc;            // The number of equations, neqn, is work(3).
  p_solve -> h_work [ 3 ] = UROUND; // 2.22e-16 // The unit roundoff, UROUND, is work(4).
  p_solve -> h_work [ 4 ] = sqrt ( UROUND );   // The square root of uround, sqrtu, is work(5).
  p_solve -> h_work [ 5 ] = H_MAX; //  The maximum step size, hmax, is work(6).8
  p_solve -> h_work [ 6 ] = l_h_min [ p_ -> neuron_type ]; // The minimum step size, hmin  // The base address for the solution is ptr1 = nint(work(7)).
  
  //強い刺激に対しては1.0e-7;
  //p_pkj_solve -> h_others [ atol_r ] = p_pkj_solve -> h_others [ rtol ] = 1.0e-8;    
  p_solve -> h_others [ atol_r ] = p_solve -> h_others [ rtol ] = l_tol [ p_ -> neuron_type ];
  p_solve -> h_others [ tend ] = H_MAX;
  
  p_solve -> h_bool [ last ] = true;
  p_solve -> h_bool [ newspc ] = true;
  p_solve -> h_bool [ jacatt ] = false;

  dydt <<< numBlocks, numThreadsPerBlock >>> ( vec [ yn_r ], vec [ fn ], d_, d_solve );

  stat1 = cublasDcopy ( handle1, nc, vec [ fn ], 1, vec [ ede ], 1);
  
  free ( h_nc );
}

__host__ static
void set_initial_absh ( neuron_t *p_, neuron_solve_t *p_solve,
                        neuron_t *d_, neuron_solve_t *d_solve,
                        cublasStatus_t stat1, cublasHandle_t handle1 )
{
  int nc = p_ -> nc;
  //double **elem = p_pkj -> elem;
  double **vec  = p_solve -> vec;  
  double *dammy = p_solve -> dammy;
  int numThreadsPerBlock = p_solve -> numThreadsPerBlock;
  int numBlocks = p_solve -> numBlocks;
  double absh_init = H_MAX; // vec [ work ] [ 5 ]

  if ( p_solve -> h_others [ sprad ] * absh_init > 1.0 ){
    absh_init = 1.0 /  p_solve -> h_others [ sprad ];
  }
  absh_init = fmax_ori( absh_init, p_solve -> h_work [ 6 ] );
   
    
  /* vec [ vtemp1 ] [ i ] = vec [ yn_r ] [ i ] + absh_init * vec [ fn ] [ i ]; */
  // vtemp1 [i] = yn_r [i]
  stat1 = cublasDcopy ( handle1, nc, vec [ yn_r ], 1, vec [ vtemp1 ], 1);
  // vtemp1 [i] += absh_init * fn [i]
  stat1 = cublasDaxpy ( handle1, nc, &absh_init, vec [ fn ], 1, vec [ vtemp1 ], 1 ); 
    
  dydt <<< numBlocks, numThreadsPerBlock >>> ( vec [ vtemp1 ], vec [ vtemp2 ], d_, d_solve );
  //cudaDeviceSynchronize ( );

  double est = 0.0;
  double at = p_solve -> h_others [ atol_r ];
  double wt;
  double *h_yn_r, *h_wt;
  h_yn_r   = ( double * ) malloc ( nc * sizeof ( double ) );
  h_wt   = ( double * ) malloc ( nc * sizeof ( double ) );
  stat1 = cublasGetVector ( nc, sizeof ( double ), vec [ yn_r ], 1, h_yn_r, 1 );

  for ( int i = 0; i < nc; i++ ) {
    wt = at + p_solve -> h_others [ rtol ] * fabs ( h_yn_r [ i ] );
    if ( wt == 0.0 ) { 
      printf( " idid = 3 \n " );
      p_solve -> h_others [ idid ] = 3.0;
      return;
    }
    /* est += ( ( vec [ vtemp2 ] [ i ] - vec [ fn ] [ i ] ) / wt ) 
         * ( ( vec [ vtemp2 ] [ i ] - vec [ fn ] [ i ] ) / wt ); */
    h_wt [ i ] = 1.0 / wt;
  }
  /* est += ( ( vec [ vtemp2 ] [ i ] - vec [ fn ] [ i ] ) * h_wt [i] ) ^ 2; */
  // dammy [i] = h_wt [i]
  stat1 = cublasSetVector ( nc, sizeof ( double ), h_wt, 1, dammy, 1 );
  // dammy [i] = ( ( vec [ vtemp2 ] [ i ] - vec [ fn ] [ i ] ) * dammy [i] );
  //fprintf( stderr, "d_pkj->nc: %d\n", d_pkj -> nc);
  fprintf(stderr, "p_->nc:%d\n", p_ -> nc);
  //Hadamard_product <<< 1, 1 >>> ( d_pkj, d_pkj_solve );
  Hadamard_product <<< (p_ -> nc + 127)/128, 128 >>> ( d_, d_solve );
  //cudaDeviceSynchronize ( );
  // est = || dammy ||
  stat1 = cublasDdot  ( handle1, nc, dammy, 1, dammy, 1, &est ); 

  est = absh_init * sqrt ( est / nc );
  if ( 0.1 * absh_init < H_MAX * sqrt ( est ) ) {
    absh_init = fmax_ori ( 0.1 * absh_init / sqrt ( est ) , p_solve -> h_work [ 6 ] );
  }
  else {
    absh_init = H_MAX;
  }

  p_solve -> h_others[ absh ] = absh_init;
  p_solve -> h_others [ mmax ] = nint ( sqrt ( p_solve -> h_others [ rtol ] /  ( 10.0 * UROUND ) ) );

  free ( h_yn_r ); free ( h_wt );
}

__host__
static void rkcrho ( neuron_t *p_, neuron_solve_t *p_solve, 
                     neuron_t *d_, neuron_solve_t *d_solve,
                     cublasStatus_t stat1, cublasHandle_t handle1 )
{ 
  double **vec  = p_solve -> vec;  
  double *dammy = p_solve -> dammy;
  int numThreadsPerBlock = p_solve -> numThreadsPerBlock;
  int numBlocks = p_solve -> numBlocks;
  int nc = p_ -> nc;
  int itmax = 1000;
  double sprad_old [ 3 ] = { 0.0, 0.0, 0.0 };
  double uround = UROUND;
  double sqrtu = sqrt ( UROUND );
  double small = 1.0 / 0.125;//H_MAX;
  double ynrm, sigma, sigmal, dynrm, dfnrm, vnrm;
  
  //ede : The estimated dominant eigenvector
  //for (int i = 0; i < nc; i++) { vec [ vtemp1 ] [ i ] = vec [ ede ] [ i ]; }
  //stat1 = cublasDcopy ( handle1, nc, vec [ ede ], 1, vec [ vtemp1 ], 1);
  ynrm = 0.0; 
  vnrm = 0.0;
  
  //ynrm = ynrm + vec [ yn_r ] [ i ] * vec [ yn_r ] [ i ] ;
  //vnrm = vnrm + vec [ vtemp1 ] [ i ] * vec [ vtemp1 ] [ i ];
  stat1 = cublasDdot ( handle1, nc, vec [ yn_r ], 1, vec [ yn_r ], 1, &ynrm );

  //stat1 = cublasDdot ( handle1, nc, vec [ vtemp1 ], 1, vec [ vtemp1 ], 1, &vnrm );  
  stat1 = cublasDdot ( handle1, nc, vec [ ede ], 1, vec [ ede ], 1, &vnrm );  

  ynrm = sqrt ( ynrm ); 
  vnrm = sqrt ( vnrm );
  
  if ( ynrm != 0.0 && vnrm != 0.0 ) {
    dynrm = ynrm * sqrtu;
    /* vec [ vtemp1 ] [ i ] = vec [ yn_r ] [ i ]  + vec [ vtemp1 ] [ i ] * ( dynrm / vnrm ); */
    double alpha = dynrm / vnrm;
    vec_sA_plus_B <<< numBlocks, numThreadsPerBlock >>> ( alpha, vec [ ede ], vec [ yn_r ], vec [ vtemp1 ], nc );
    /*
    double one = 1.0;
    // vtemp1 [i] = alpha * vtemp1[i]
    stat1 = cublasDscal ( handle1, nc, &alpha, vec [ vtemp1 ], 1 );
    // vtemp1 [i] += yn_r[i]
    stat1 = cublasDaxpy ( handle1, nc, &one, vec [ yn_r ], 1, vec [ vtemp1 ], 1 ); 
    */
  }
  else if ( ynrm != 0.0 ) {
    dynrm = ynrm * sqrtu;
    /* vec [ vtemp1 ] [ i ] = vec [ yn_r ] [ i ] + vec [ yn_r ] [ i ] * sqrtu; */
    vec_sA_plus_B <<< numBlocks, numThreadsPerBlock >>> ( sqrtu, vec [ yn_r ], vec [ yn_r ], vec [ vtemp1 ], nc );
    // vtemp1 [i] = yn_r [i]
    /*
    stat1 = cublasDcopy ( handle1, nc, vec [ yn_r ], 1, vec [ vtemp1 ], 1);
    // vtemp1 [i] += sqrtu * yn_r[i]
    stat1 = cublasDaxpy ( handle1, nc, &sqrtu, vec [ yn_r ], 1, vec [ vtemp1 ], 1 ); 
    */
  }
  else if ( vnrm != 0.0 ) {
    dynrm = uround;
    /* vec [ vtemp1 ] [ i ] = vec [ vtemp1 ] [ i ] * (dynrm / vnrm); */
    double alpha = dynrm / vnrm;
    vec_sA <<< numBlocks, numThreadsPerBlock >>> ( alpha, vec [ ede ], vec [ vtemp1 ], nc );
    //stat1 = cublasDscal ( handle1, nc, &alpha, vec [ vtemp1 ], 1 );    
  }
  else { // both ynrm and vnrm are zero vectors ( rare )
    dynrm = uround;
    /* vec [ vtemp1 ] [ i ] = dynrm; */
    set_s2A <<< numBlocks, numThreadsPerBlock >>> ( dynrm, vec [ vtemp1 ], nc);
    /*
    double *h_;
    h_ = ( double * ) malloc ( nc * sizeof ( double ) );
    for ( int i = 0; i < nc; i++ ) { h_ [ i ] = dynrm; }
    // from host h_ to device vtemp1
    stat1 = cublasSetVector ( nc, sizeof ( double ), h_, 1, vec [ vtemp1 ], 1 );
    free ( h_ );
    */
  }

  // Now iterate with a nonlinear power method.
  sigma = 0.0;

  for ( int iter = 0; iter < itmax; iter++ ) 
  {
    dydt <<< numBlocks, numThreadsPerBlock >>> ( vec [ vtemp1 ], vec [ vtemp2 ], d_, d_solve );
    //cudaDeviceSynchronize ( );
    dfnrm = 0.0;
    /* dfnrm += ( vec [ vtemp2 ] [ i ] - vec [ fn ] [ i ] ) * ( vec [ vtemp2 ] [ i ] - vec [ fn ] [ i ] ); */
    /* -> dammy = vec[vtemp2] - vec[fn]; dfnrm = cublasDdot(dammy);*/
    vec_sA_plus_B <<< numBlocks, numThreadsPerBlock >>> ( -1.0, vec [ fn ], vec [ vtemp2 ], dammy, nc );

    /*
    // dammy [i] = vtemp2 [i]
    stat1 = cublasDcopy ( handle1, nc, vec [ vtemp2 ], 1, dammy, 1);
    // dammy [i] -= fn [i]
    //double mone = -1.0;
    stat1 = cublasDaxpy ( handle1, nc, &mone, vec [ fn ], 1, dammy, 1 ); 
    // dfnrm = || dammy [i] ||
    */

    stat1 = cublasDdot ( handle1, nc, dammy, 1, dammy, 1, &dfnrm );  

    dfnrm = sqrt ( dfnrm );
    sigmal = sigma;
    sigma = dfnrm / dynrm;

    // sprad is a little bigger than the estimate sigma of the
    // spectral radius, so is more likely to be an upper bound.
    //stat1 = cublasGetVector ( n_vec_RKC_others, sizeof ( double ), vec [ others ], 1, h_temp_vec, 1 );
    sprad_old [ 0 ] = sprad_old [ 1 ];
    sprad_old [ 1 ] = sprad_old [ 2 ];
    sprad_old [ 2 ] =  ( p_solve -> h_others [ sprad ] );// vec [ others ] [ sprad ]
    
    //h_temp_vec [ sprad ] = 1.2 * sigma;
    p_solve -> h_others [ sprad ] = 1.2 * sigma;

    //stat1 = cublasSetVector ( n_vec_RKC_others, sizeof ( double ), h_temp_vec, 1, vec [ others ], 1 );

    if ( iter >= 1 && fabs ( sigma - sigmal ) <= fmax_ori ( sigma, small ) * 0.001 ) 
    {
      /* vec [ ede ] [ i ] = vec [ vtemp1 ] [ i ] - vec [ yn_r ] [ i ]; */
      vec_sA_plus_B <<< numBlocks, numThreadsPerBlock >>> ( -1.0, vec [ yn_r ], vec [ vtemp1 ], vec [ ede ], nc );
      /*
      // ede [i] = vtemp1 [i]
      stat1 = cublasDcopy ( handle1, nc, vec [ vtemp1 ], 1, vec [ ede ], 1);
      // ede [i] -= yn_r [i]
      //double mone = -1.0;
      stat1 = cublasDaxpy ( handle1, nc, &mone, vec [ yn_r ], 1, vec [ ede ], 1 ); 
      */
      return;
    }
  
    //振動している場合はspradを振幅の平均で計算しリターン
    if ( iter > itmax / 2 ) 
    //if ( iter > 10 ) 
    {
      if ( ( fabs ( sprad_old [ 2 ] - sprad_old [ 0 ] ) < 0.00001 )
        && ( fabs ( sprad_old [ 1 ] - p_solve -> h_others [ sprad ] ) < 0.00001 ) ) 
      {
        // vec [ others ] [ sprad ] = (vec [ others ] [ sprad ] + sprad_old [ 2 ] ) / 2.0;
        //h_temp_vec [ sprad ] = ( h_temp_vec [ sprad ]  + sprad_old [ 2 ] ) / 2.0;
        p_solve -> h_others [ sprad ] = ( p_solve -> h_others [ sprad ]  + sprad_old [ 2 ] ) / 2.0;

        //stat1 = cublasSetVector ( n_vec_RKC_others, sizeof ( double ), h_temp_vec, 1, vec [ others ], 1 );
        /* vec [ ede ] [ i ] = vec [ vtemp1 ] [ i ] - vec [ yn_r ] [ i ]; */
        vec_sA_plus_B <<< numBlocks, numThreadsPerBlock >>> ( -1.0, vec [ yn_r ], vec [ vtemp1 ], vec [ ede ], nc );
        /*
        // ede [i] = vtemp1 [i]
        stat1 = cublasDcopy ( handle1, nc, vec [ vtemp1 ], 1, vec [ ede ], 1);
        // ede [i] -= yn_r [i]
        //double mone = -1.0;
        stat1 = cublasDaxpy ( handle1, nc, &mone, vec [ yn_r ], 1, vec [ ede ], 1 );
        printf("sprad calculation didn't converge \n ");
        */
        return;
      }
    }

    // The next v(*) is the change in dydt
    // scaled so that norm(v - yn) = dynrm.
    if ( dfnrm != 0.0 ) 
    {      
      /* vec [ vtemp1 ] [ i ] = vec [ yn_r ] [ i ] + ( vec [ vtemp2 ] [ i ] - vec [ fn ] [ i ]) * ( dynrm / dfnrm ); */
      double alpha = dynrm / dfnrm;
      vec_sA_plus_B <<< numBlocks, numThreadsPerBlock >>> ( -1.0,  vec [ vtemp2 ], vec [ fn ],   vec [ vtemp1 ], nc );
      vec_sA_plus_B <<< numBlocks, numThreadsPerBlock >>> ( alpha, vec [ vtemp1 ], vec [ yn_r ], vec [ vtemp1 ], nc );

      /*
      // dammy [i] = vtemp2 [i]
      stat1 = cublasDcopy ( handle1, nc, vec [ vtemp2 ], 1, dammy, 1 );
      // dammy [i] -= fn [i]
      //double mone = -1.0;
      stat1 = cublasDaxpy ( handle1, nc, &mone, vec [ fn ], 1, dammy, 1 ); 
      // vtemp1 [i] = yn_r [i]
      stat1 = cublasDcopy ( handle1, nc, vec [ yn_r ], 1, vec [ vtemp1 ], 1);
      // vec [ vtemp1 ] [ i ] += dammy [ i ] * (dynrm / dfnrm); 
      double alpha = dynrm / dfnrm;
      stat1 = cublasDaxpy ( handle1, nc, &alpha, dammy, 1, vec [ vtemp1 ], 1 ); 
      */
    }
    else 
    {
      /* vec [ vtemp1 ] [ index ] = vec [ yn_r ] [ index ] - (vec [ vtemp1 ] [ index ] - vec [ yn_r ] [ index ] ); */
      int index = iter % nc;
      rand_elements_change <<< 1, 1 >>> ( vec [ vtemp1 ], vec [ yn_r ], index );
    }
  }
  
  //h_temp_vec [ idid ] = 6;
  p_solve -> h_others [idid] = 6;
  //stat1 = cublasSetVector ( n_vec_RKC_others, sizeof ( double ), h_temp_vec, 1, vec [ others ], 1 );
  
  return;
}

__host__ 
void RKC_vec_initialize ( neuron_t *d_, neuron_solve_t *d_solve, 
                             neuron_t *p_, neuron_solve_t *p_solve )
{
  // set cublas
  cublasStatus_t stat1;
  cublasHandle_t handle1;
  stat1 = cublasCreate_v2 ( &handle1 );
  
  // set b // global
  switch ( p_ -> neuron_type ) {
    case N_TYPE_GR:
      gr_update_matrix <<< p_solve -> numBlocks, p_solve -> numThreadsPerBlock >>>  ( d_, d_solve, p_ -> elem [ v ] );
      break;
    case N_TYPE_GO:
      go_update_matrix <<< p_solve -> numBlocks, p_solve -> numThreadsPerBlock >>>  ( d_, d_solve, p_ -> elem [ v ] );
      break;
    case N_TYPE_PKJ:
      pkj_update_matrix <<< p_solve -> numBlocks, p_solve -> numThreadsPerBlock >>> ( d_, d_solve, p_ -> elem [ v ] );
      break;
    case N_TYPE_IO:
      io_update_matrix <<< p_solve -> numBlocks, p_solve -> numThreadsPerBlock >>>  ( d_, d_solve, p_ -> elem [ v ] );
      break;
  }  

  // initialize vectors
  set_initial_vec ( p_, p_solve, d_, d_solve, stat1, handle1 );
  // Compute an initial step size.
  rkcrho ( p_, p_solve, d_, d_solve, stat1, handle1 );
  set_initial_absh ( p_, p_solve, d_, d_solve, stat1, handle1 );

  if ( p_solve -> h_others [ idid ] == 3.0 ) exit ( 1 );

  // destroy cublas
  cublasDestroy_v2 ( handle1 );
}


__global__ static
void rkcint ( double *d_v_new, const double hlast, const double lt, 
                 const double d_gt, const double *d_vtemp1, const double *d_vtemp2, 
                 const double *d_yn_r, const double *d_fn, int nc )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < nc ) 
  {
    double a1, a2, b1, b2, s, /*hlast, lt,*/ tlast;
    tlast = lt - hlast;
    s = ( d_gt - tlast ) / hlast;
    a1 = ( 1.0 + 2.0 * s ) * ( s - 1.0 ) * ( s - 1.0 );
    a2 = ( 3.0 - 2.0 * s ) * s * s;
    b1 = hlast * s * ( s - 1.0 ) * ( s - 1.0 );
    b2 = hlast * ( s - 1.0 ) * s * s;
    
    d_v_new [ id ] = a1 *  d_vtemp1 [ id ] + a2 *  d_yn_r [ id ]
                   + b1 * d_vtemp2 [ id ] + b2 * d_fn [ id ] ;    
  }
}

__host__
static void step ( neuron_t *p_, neuron_solve_t *p_solve, 
                   neuron_t *d_, neuron_solve_t *d_solve,
                   cublasStatus_t stat1, cublasHandle_t handle1, double h, int m ) 
{
  double **vec  = p_solve -> vec;
  int nc = p_ -> nc;
  int numThreadsPerBlock = p_solve -> numThreadsPerBlock;
  int numBlocks = p_solve -> numBlocks;
  //  Take a step of size H from T to T + H to get Y(*).
  double ajm1, arg, bj, bjm1, bjm2, dzj, dzjm1, dzjm2,
	d2zj, d2zjm1, d2zjm2, mu, mus, nu,
	temp1, temp2, thj, thjm1, thjm2, w0, w1,
	zj, zjm1, zjm2;

  double *tmp;

  w0 = 1.0 + 2.0 / ( 13.0 * m * m );
  temp1 = w0 * w0 - 1.0;
  temp2 = sqrt ( temp1 );
  arg = m * log ( w0 + temp2 );
  w1 = sinh ( arg ) * temp1 / ( cosh ( arg ) * m * temp2 - w0 * sinh ( arg ) );
  bjm1 = 1.0 / ( ( 2.0 * w0 ) * ( 2.0 * w0 ) );
  bjm2 = bjm1;
  mus = w1 * bjm1;

  //Evaluate the first stage  
  // vec [ vtemp2 ] [ i ] = vec [ yn_r ] [ i ];
  stat1 = cublasDcopy ( handle1, nc, vec [ yn_r ], 1, vec [ vtemp2 ], 1);  
  
  //vec [ vtemp1 ] [ i ] = vec [ yn_r ] [ i ] + h * mus * vec [ fn ] [ i ];
  double l_hmus = h * mus;
  vec_sA_plus_B <<< numBlocks, numThreadsPerBlock>>> ( l_hmus, vec [ fn ], vec [ yn_r ], vec [ vtemp1 ], nc );
  /*
  stat1 = cublasDcopy ( handle1, nc, vec [ yn_r ], 1, vec [ vtemp1 ], 1);
  stat1 = cublasDaxpy ( handle1, nc, &l_hmus, vec [ fn ], 1, vec [ vtemp1 ], 1 );
  */
  thjm2 = 0.0;  thjm1 = mus;  zjm1 = w0;
  zjm2 = 1.0;  dzjm1 = 1.0;  dzjm2 = 0.0;
  d2zjm1 = 0.0;  d2zjm2 = 0.0;

  // Evaluate stages j = 1, ..., m-1. // j = 2, ..., m. by fortran
  for ( int j = 1; j < m; j++ ) 
  {
  	zj = 2.0 * w0 * zjm1 - zjm2;
	  dzj = 2.0 * w0 * dzjm1 - dzjm2 + 2.0 * zjm1;
	  d2zj = 2.0 * w0 * d2zjm1 - d2zjm2 + 4.0 * dzjm1;
	  bj = d2zj / (dzj * dzj);
	  ajm1 = 1.0 - zjm1 * bjm1;
	  mu = 2.0 * w0 * bj / bjm1;
	  nu = -bj / bjm2;
	  mus = mu * w1 / w0;

	  // Use the y array for temporary storage here.
    dydt <<< numBlocks, numThreadsPerBlock >>> ( vec [ vtemp1 ], vec [ y ], d_, d_solve );
    //cudaDeviceSynchronize ( );
	
    /* vec [ y ] [ i ] = mu * vec [ vtemp1 ] [ i ] + nu * vec [ vtemp2 ] [ i ] 
                    + (1.0 - mu - nu) * vec [ yn_r ] [ i ] + h * mus * ( vec [ y ] [ i ] - ajm1 * vec [ fn ] [ i ] ); */
    calcY <<< numBlocks, numThreadsPerBlock >>> ( mu, vec [ vtemp1 ], nu, vec [ vtemp2 ], (1.0 - mu - nu), vec [ yn_r ], h * mus, vec [ y ], ajm1, vec [ fn ], nc );
                         
	  thj = mu * thjm1 + nu * thjm2 + mus * ( 1.0 - ajm1 );

	  // Shift the data for the next stage.		
    if ( j < m ) 
    {	
	    // vec [ vtemp2 ] [ i ] = vec [ vtemp1 ] [ i ];
      // vec [ vtemp1 ] [ i ] = vec [ y ] [ i ];
      //stat1 = cublasDcopy ( handle1, nc, vec [ vtemp1 ], 1, vec [ vtemp2 ], 1);
      //stat1 = cublasDcopy ( handle1, nc, vec [ y ], 1, vec [ vtemp1 ], 1);
      tmp = vec [ vtemp2 ];
      vec [ vtemp2 ] = vec [ vtemp1 ];
      vec [ vtemp1 ] = vec [ y ];
      vec [ y ] = tmp;
	    thjm2 = thjm1;
	    thjm1 = thj;
	    bjm2 = bjm1;
  	  bjm1 = bj;
	    zjm2 = zjm1;
	    zjm1 = zj;
  	  dzjm2 = dzjm1;
	    dzjm1 = dzj;
	    d2zjm2 = d2zjm1;
  	  d2zjm1 = d2zj;
	  }
  }
  tmp = vec [ y ];
  vec [ y ] = vec [ vtemp1 ];
  vec [ vtemp1 ] = vec [ vtemp2 ];
  vec [ vtemp2 ] = tmp;  
  return;
}

__host__
static void rkclow ( neuron_t *p_, neuron_solve_t *p_solve, 
                     neuron_t *d_, neuron_solve_t *d_solve,
                     cublasStatus_t stat1, cublasHandle_t handle1 ) 
{
  double **vec  = p_solve -> vec;  
  double *dammy = p_solve -> dammy;
  int numThreadsPerBlock = p_solve -> numThreadsPerBlock;
  int numBlocks = p_solve -> numBlocks;
  int nc = p_ -> nc;
  int m;
  double err, fac, h, temp1, temp2;
  // Initialize on the first call.
  //double uround = p_pkj_solve -> h_work [ 3 ];
  //	mmax = int(fmax_ori(mmax, 2.0)); // uroundを変更するときは注意
  double tdir = 1.0;
  //double tdir = sign_fortran(one, tend - vec [ others ] [ lt]);//sign:第一引数の絶対値に第二引数の符合をつけたものを求める
  double hmax = H_MAX;
  double hmin = p_solve -> h_work [ 6 ];

  // Debug
  //static int nreject = 0;
  //static int sum_m = 0;
  // Compute an initial step size.
  //static double absh = absh_init;//absh_initをabshとして扱う

  int loop_num = 0;
  // Start of loop for taking one step.
  while ( 1 ) 
  {
    loop_num++;
	  if ( loop_num > 1000 ) { printf ( " ループ回数多過ぎ \n " ); return; }
    if ( p_solve -> h_bool [ newspc ] ) 
    {
	    rkcrho ( p_, p_solve, d_, d_solve, stat1, handle1 );
	    p_solve -> h_bool [ jacatt ] = true;
	  }

	  //Adjust the step size and determine the number of stages m.
	  p_solve -> h_bool [ last ] = false;
    if ( 1.1 * p_solve -> h_others [ absh ] >= fabs( p_solve -> h_others [ tend ] - p_solve -> h_others [ lt ])) 
    {
	    p_solve -> h_others [ absh ] = fabs ( p_solve -> h_others [ tend ] - p_solve -> h_others [ lt ] );
	    p_solve -> h_bool [ last ] = true;
    }
    m = 1 + ( int ) ( sqrt ( 1.54 * p_solve -> h_others [ absh ] * p_solve -> h_others [ sprad ] + 1.0 ) );

    //Limit m to mmax to control the pkjowth of roundoff error.
    if ( m > p_solve -> h_others [ mmax ] ) 
    {
      m = p_solve -> h_others [ mmax ];
      p_solve -> h_others [ absh ] = ( m * m - 1) / ( 1.54 * p_solve -> h_others [ sprad ] );
      p_solve -> h_bool [ last ] = false;
    }
    h = tdir * p_solve -> h_others [ absh ];
    
    //  h を 2^(-n)に切り捨てる
    if ( h <= hmin ) { h = hmin; } 
    //else 
    //{
    //  double local_h = ceil ( log2 ( 1 / h ) );  // hの逆数が2の何乗か調べて切り上げて
    //  h = 1.0 / ( pow ( 2.0, local_h ) );		 // その数分2を累乗して逆数に戻すと切り捨てになる      
    //}

    //if ( h + p_solve -> h_others [ lt ] > p_solve -> h_others [ tend ] )
    //  h = p_solve -> h_others [ tend ] - p_solve -> h_others [ lt ];

    
    //stat1 = cublasSetVector ( n_vec_RKC_others, sizeof ( double ), h_ot, 1, vec [ others ], 1 );
    //hmin = 10.0 * p_pkj_solve -> h_work [ 3 ] * fmax_ori(fabs(p_pkj_solve -> h_others [ lt ]), fabs(p_pkj_solve -> h_others [ lt ] + h));//更新しない
    step ( p_, p_solve, d_, d_solve, stat1, handle1, h, m );
    dydt <<< numBlocks, numThreadsPerBlock >>> ( vec [ y ], vec [ vtemp1 ], d_, d_solve );
   
    err = 0.0;
    err = err_estimate ( p_solve -> h_others, h, p_solve, nc );

    // Debug
    //sum_m += m;

    double l_idid = p_solve -> h_others [ idid ];

    if ( l_idid == 3.0 )
    {
      p_solve -> h_others [ idid ] = 3;
      return; 
    }

    err = sqrt ( err / ( double ) nc );

    if ( err > 1.0 ) 
    {
      //Step is rejected.
      p_solve -> h_others [ absh ] = 0.8 * p_solve -> h_others [ absh ] / ( pow ( err, 1.0 / 3.0 ) );

      // Debug
      //nreject++;
      //printf ("reject absh = %f\n", h_ot [ absh ]);

      if ( p_solve -> h_others [ absh ] < hmin ) 
      { 
        p_solve -> h_others [ idid ] = 4;
        //stat1 = cublasSetVector ( n_vec_RKC_others, sizeof ( double ), h_ot, 1, vec [ others ], 1 );
      } 
      else 
      { 
        p_solve -> h_bool [ newspc ] = ( ! ( p_solve -> h_bool [ jacatt ] ) ); 
        //stat1 = cublasSetVector ( n_vec_RKC_others, sizeof ( double ), h_ot, 1, vec [ others ], 1 );
        continue; 
      }
    }

    // Step is accepted.
    p_solve -> h_others [ naccpt ] += 1;
    p_solve -> h_others [ lt ] += h;
    p_solve -> h_others [ gt ] = p_solve -> h_others [ lt ];
    p_solve -> h_bool [ jacatt ] = false;//The Jacobian may not be constant.
    int l_nstsig = ( ( int ) ( p_solve -> h_others [ nstsig ] ) + 1 ) % 25;//mod(nstsig+1,25)
    p_solve -> h_bool [ newspc ] = false;
    if ( l_nstsig == 0 ) { p_solve -> h_bool [ newspc ] = ( !( p_solve -> h_bool [ jacatt ] ) ); }

    // Update the data for interpolation stored in vec [ work ](*).
    p_solve -> h_work [ 0 ] = h;
    p_solve -> h_work [ 1 ] = p_solve -> h_others [ lt ];
    double *tmp;
    tmp = vec [ vtemp2 ];
    vec [ vtemp2 ] = vec [ fn ];
    vec [ fn ] = vec [ vtemp1 ];
    vec [ vtemp1 ] = vec [ yn_r ];
    vec [ yn_r ] = vec [ y ];
    vec [ y ] = tmp;

    // Debug
    //printf ("err = %f\n", err ); 

    fac = 10.0;
    if ( p_solve -> h_others [ naccpt ] == 1) 
    {
      temp2 = pow ( err, 1.0 / 3.0 );
      if ( 0.8 < fac * temp2 ) { fac = 0.8 / temp2; }

      // Debug
      //printf ("n = 1, err = %f, temp2 = %f\n",err * 1000000000, temp2 ); 
      
    }  
    else 
    {
      temp1 = 0.8 * p_solve -> h_others [ absh ] * pow ( p_solve -> h_others [ errold ], 1.0 / 3.0 );
      temp2 = fabs ( p_solve -> h_others [ hold ] ) * pow ( err, 2.0 / 3.0 );

      // Debug
      //printf ("temp1 = %f, temp2 = %f\n", temp1, temp2 ); 

      if ( temp1 < fac * temp2 ) { fac = temp1 / temp2; }
    }
    p_solve -> h_others [ absh ] = fmax_ori( 0.1, fac ) * p_solve -> h_others [ absh ];
    p_solve -> h_others [ absh ] = fmax_ori ( hmin, fmin ( hmax, p_solve -> h_others [ absh ] ) );
    p_solve -> h_others [ errold ] = err;
    p_solve -> h_others [ hold ] = h;
    h = tdir * p_solve -> h_others [ absh ];

    // Debug
    //static int l_naccpt = 0;
    //static int l_nreject = 0;
    //if ( p_solve -> h_bool [ last ] ){
    //  FILE *debug_file;
    //  debug_file = fopen ( "debug_file.csv", "a" );
    //  fprintf ( debug_file, "%lf,%lf,%d,%d,\n", p_solve -> h_others [ gt ], p_solve -> h_others [ naccpt ] - l_naccpt, nreject - l_nreject, sum_m ); 
    //  fclose ( debug_file );
    //  sum_m = 0;
    //  l_naccpt = p_solve -> h_others [ naccpt ];
    //  l_nreject = nreject;
    //}



    //stat1 = cublasSetVector ( 8, sizeof ( double ), h_work, 1, vec [ work ], 1 );    
    if ( p_solve -> h_bool [ last ] ) 
    { 
      p_solve -> h_others [ idid ] = 1;
      //stat1 = cublasSetVector ( n_vec_RKC_others, sizeof ( double ), h_ot, 1, vec [ others ], 1 );
      return;
    }
    else {
      p_solve -> h_others [ idid ] = 2;    
      //stat1 = cublasSetVector ( n_vec_RKC_others, sizeof ( double ), h_ot, 1, vec [ others ], 1 );
      return;
    }
  }
}

////////////////////******** Each neuron ************///////////////////

__host__ static
void gr_update_ion_and_matrix ( neuron_t *d_gr, neuron_solve_t *d_gr_solve, 
                             neuron_t *p_gr, neuron_solve_t *p_gr_solve, 
                             synapse_t *d_mfgr, synapse_t *d_gogr, const int nc, double sum_dt )
{    
  double **vec  = p_gr_solve -> vec;
  double **elem = p_gr -> elem;
  double **ion  = p_gr -> ion;
  static int numThreadsPerBlock = p_gr_solve -> numThreadsPerBlock;
  static int numBlocks = p_gr_solve -> numBlocks;

  gr_Na_update_2order <<< numBlocks, numThreadsPerBlock >>>
    ( nc, vec [ y ], elem [ v ], sum_dt, elem [ compart ],
      ion [ o_Na ],  ion [ c1_Na ], ion [ c2_Na ], ion [ c3_Na ], ion [ c4_Na ], ion [ c5_Na ],
      ion [ i1_Na ], ion [ i2_Na ], ion [ i3_Na ], ion [ i4_Na ], ion [ i5_Na ], ion [ i6_Na ] );
  gr_update_ion_RKC_exp_imp <<< numBlocks, numThreadsPerBlock >>> ( d_gr, d_gr_solve, vec [ y ], sum_dt );     
  gr_update_matrix <<< numBlocks, numThreadsPerBlock >>>  ( d_gr, d_gr_solve, vec [ y ] );
  add_mfgr_val <<< ( d_mfgr -> n + 127 ) / 128, 128  >>> ( d_gr_solve, vec [ y ], d_mfgr -> comp, d_mfgr -> elem, d_mfgr -> n );
  add_gogr_val <<< ( d_gogr -> n + 127 ) / 128, 128  >>> ( d_gr_solve, vec [ y ], d_gogr -> comp, d_gogr -> elem, d_gogr -> n );
}

__host__ 
void gr_solve_by_rkc ( neuron_t *d_gr, neuron_solve_t *d_gr_solve, 
                       neuron_t *p_gr, neuron_solve_t *p_gr_solve,
                       synapse_t *d_mfgr, synapse_t *d_gogr )
{  
  double **vec  = p_gr_solve -> vec;
  int nc = p_gr -> nc;
  static double sum_dt = 0.0;
  static int numThreadsPerBlock = p_gr_solve -> numThreadsPerBlock;
  static int numBlocks = p_gr_solve -> numBlocks;
  static double integ_current_time = pow ( 2.0, log10 ( p_gr_solve -> h_others [ atol_r ] ) );

  // set cublas
  cublasStatus_t stat1;
  cublasHandle_t handle1;
  stat1 = cublasCreate_v2 ( &handle1 );
  
  if ( sum_dt != 0.0 )
  {
    gr_update_ion_and_matrix ( d_gr, d_gr_solve, p_gr, p_gr_solve, d_mfgr, d_gogr, nc, sum_dt );
    sum_dt = 0.0;
  }
  while ( 1 ) 
  {
    rkclow ( p_gr, p_gr_solve, d_gr, d_gr_solve, stat1, handle1 );   
    sum_dt += p_gr_solve -> h_others [ gt ] - p_gr_solve -> h_others [ gt_old ]; 
    p_gr_solve -> h_others [ gt_old ] = p_gr_solve -> h_others [ gt ];
    if ( sum_dt > integ_current_time && p_gr_solve -> h_others [ idid ] == 2.0 ) 
    {
      gr_update_ion_and_matrix ( d_gr, d_gr_solve, p_gr, p_gr_solve, d_mfgr, d_gogr, nc, sum_dt );      
      sum_dt = 0.0;  
    } 
    if ( p_gr_solve -> h_others [ idid ] == 2.0 ) { continue; }
    else if ( p_gr_solve -> h_others [ idid ] == 1.0 ) { break; } // h_ot [ idid ] = 2.0;?
    else { printf ( " !! error !! idid = %f \n ", p_gr_solve -> h_others [ idid ] ); exit ( 1 ); }
  }
  rkcint <<< numBlocks, numThreadsPerBlock >>> ( vec [ v_new ], p_gr_solve -> h_work [ 0 ], p_gr_solve -> h_work [ 1 ],
                     p_gr_solve -> h_others [ gt_old ], vec [ vtemp1 ], vec [ vtemp2 ], vec [ yn_r ], vec [ fn ], nc );
  p_gr_solve -> h_others [ tend ] += H_MAX;
  cublasDestroy_v2 ( handle1 );
}

__host__ static
void go_update_ion_and_matrix ( neuron_t *d_go, neuron_solve_t *d_go_solve, 
                                neuron_t *p_go, neuron_solve_t *p_go_solve,
                                synapse_t *d_grgo,  double sum_dt )
{    
  double **vec  = p_go_solve -> vec;
  double **elem = p_go -> elem;
  double **ion  = p_go -> ion;
  static int numThreadsPerBlock = p_go_solve -> numThreadsPerBlock;
  static int numBlocks = p_go_solve -> numBlocks;

  go_update_ion_RKC_exp_imp <<< numBlocks, numThreadsPerBlock >>> ( d_go, d_go_solve, vec [ y ], sum_dt ); 
  go_KAHP_update_2order <<< numBlocks, numThreadsPerBlock >>> 
   ( p_go -> n, elem [ Ca ], p_go -> ca_old, ion [ o1_KAHP_go ], ion [ o2_KAHP_go ], ion [ c1_KAHP_go ],
     ion [ c2_KAHP_go ], ion [ c3_KAHP_go ], ion [ c4_KAHP_go ], sum_dt );       
  
  go_update_matrix <<< numBlocks, numThreadsPerBlock >>>  ( d_go, d_go_solve, vec [ y ] );
  //add_grgo_val <<< ( d_grgo -> n + 127 ) / 128, 128  >>> ( d_go_solve, vec [ y ], d_grgo -> comp, d_grgo -> elem, d_grgo -> n );
  go_update_matrix_new <<< numBlocks, numThreadsPerBlock >>>  ( d_go_solve, vec [ v_new ], p_go -> nc );
}

__host__ 
void go_solve_by_rkc ( neuron_t *d_go, neuron_solve_t *d_go_solve, 
                       neuron_t *p_go, neuron_solve_t *p_go_solve, synapse_t *d_grgo )
{  
  double **vec  = p_go_solve -> vec;
  static double sum_dt = 0.0;
  static int numThreadsPerBlock = p_go_solve -> numThreadsPerBlock;
  static int numBlocks = p_go_solve -> numBlocks;
  static double integ_current_time = pow ( 2.0, log10 ( p_go_solve -> h_others [ atol_r ] ) );
  //double h_ot [ n_vec_RKC_others ] = { };
  // set cublas
  cublasStatus_t stat1;
  cublasHandle_t handle1;
  stat1 = cublasCreate_v2 ( &handle1 );

  reset_b <<< numBlocks, numThreadsPerBlock >>> ( vec [ v_new ], p_go -> nc );
  add_grgo_val_new<<< ( d_grgo -> n + 127 ) / 128, 128  >>> ( vec [ v_new ], vec [ y ], d_grgo -> comp, d_grgo -> elem, d_grgo -> n );

  if ( sum_dt != 0 )
  {
    go_update_ion_and_matrix ( d_go, d_go_solve, p_go, p_go_solve, d_grgo, sum_dt );
    sum_dt = 0.0;
  }
  while ( 1 ) {
    rkclow ( p_go, p_go_solve, d_go, d_go_solve, stat1, handle1 );   
    sum_dt += p_go_solve -> h_others [ gt ] - p_go_solve -> h_others [ gt_old ];
    p_go_solve -> h_others [ gt_old ] = p_go_solve -> h_others [ gt ];

    if ( sum_dt >= integ_current_time && p_go_solve -> h_others [ idid ] == 2.0 ) 
    {  
      go_update_ion_and_matrix ( d_go, d_go_solve, p_go, p_go_solve, d_grgo, sum_dt );
      sum_dt = 0.0;
    } 

    if ( p_go_solve -> h_others [ idid ] == 2.0 ) { continue; }
    else if ( p_go_solve -> h_others [ idid ] == 1.0 ) { break; } // h_ot [ idid ] = 2.0;?
    else { printf ( " !! error !! idid = %f \n ", p_go_solve -> h_others [ idid ] ); exit ( 1 ); }
  }

  rkcint <<< numBlocks, numThreadsPerBlock >>>( vec [ v_new ], p_go_solve -> h_work [ 0 ], p_go_solve -> h_work [ 1 ],
       p_go_solve -> h_others [ gt_old ], vec [ vtemp1 ], vec [ vtemp2 ], vec [ yn_r ], vec [ fn ], p_go -> nc );
  p_go_solve -> h_others [ tend ] += H_MAX;
  cublasDestroy_v2 ( handle1 );
}

__host__ static
void pkj_update_ion_and_matrix ( neuron_t *d_pkj, neuron_solve_t *d_pkj_solve, 
                                neuron_t *p_pkj, neuron_solve_t *p_pkj_solve,
                                synapse_t *d_grpkj, synapse_t *d_mlipkj, double sum_dt )
{    
  double **vec  = p_pkj_solve -> vec;
  double **elem = p_pkj -> elem;
  double **ion  = p_pkj -> ion;
  static int numThreadsPerBlock = p_pkj_solve -> numThreadsPerBlock;
  static int numBlocks = p_pkj_solve -> numBlocks;

  pkj_update_ion_RKC <<< numBlocks, numThreadsPerBlock >>> ( d_pkj, d_pkj_solve, vec [ y ], sum_dt ); 

  pkj_update_matrix <<< numBlocks, numThreadsPerBlock >>>  ( d_pkj, d_pkj_solve, vec [ y ] );
  //add_grpkj_val <<< ( d_grpkj -> n + 127 ) / 128, 128  >>> ( d_pkj_solve, vec [ y ], d_grpkj -> comp, d_grpkj -> elem, d_grpkj -> n );
  //add_mlipkj_val <<< ( d_mlipkj -> n + 127 ) / 128, 128  >>> ( d_pkj_solve, vec [ y ], d_mlipkj -> comp, d_mlipkj -> elem, d_mlipkj -> n );
  
  if ( d_grpkj -> n > 0 ){
    double sum_decay = exp( -sum_dt / 0.6 );
    pkj_update_matrix_new <<< numBlocks, numThreadsPerBlock >>>  ( d_pkj_solve, vec [ v_new ], p_pkj -> nc, vec [ y ], sum_decay );
  }
}

__host__ 
void pkj_solve_by_rkc ( neuron_t *d_pkj, neuron_solve_t *d_pkj_solve, 
                        neuron_t *p_pkj, neuron_solve_t *p_pkj_solve,
                        synapse_t *d_grpkj, synapse_t *d_mlipkj )
{  
  double **vec  = p_pkj_solve -> vec;
  static double sum_dt = 0.0;
  static int numThreadsPerBlock = p_pkj_solve -> numThreadsPerBlock;
  static int numBlocks = p_pkj_solve -> numBlocks;
  //double h_ot [ n_vec_RKC_others ] = { };
  // set cublas
  static cublasHandle_t handle1;
  static cublasStatus_t stat1 = cublasCreate_v2 ( &handle1 );
  static double integ_current_time = pow ( 2.0, log10 ( p_pkj_solve -> h_others [ atol_r ] ) );
  
  if ( d_grpkj -> n > 0 ){
    reset_b <<< numBlocks, numThreadsPerBlock >>> ( vec [ v_new ], p_pkj -> nc );
    add_grpkj_val_new<<< ( d_grpkj -> n + 127 ) / 128, 128  >>> ( vec [ v_new ], d_grpkj -> comp, d_grpkj -> elem, vec [ y ], d_grpkj -> n );
    //add_grpkj_val <<< ( d_grpkj -> n + 127 ) / 128, 128  >>> ( d_pkj_solve, vec [ y ], d_grpkj -> comp, d_grpkj -> elem, d_grpkj -> n );
  }

  //if ( sum_dt >= integ_current_time * 0.5 ) 
  if ( sum_dt != 0 ) 
  {
    pkj_update_ion_and_matrix ( d_pkj, d_pkj_solve, p_pkj, p_pkj_solve, d_grpkj, d_mlipkj, sum_dt );
    sum_dt = 0.0;
  }

  while ( 1 ) {
    rkclow ( p_pkj, p_pkj_solve, d_pkj, d_pkj_solve, stat1, handle1 );   
    sum_dt += p_pkj_solve -> h_others [ gt ] - p_pkj_solve -> h_others [ gt_old ];
    p_pkj_solve -> h_others [ gt_old ] = p_pkj_solve -> h_others [ gt ];
    if ( sum_dt >= integ_current_time  && p_pkj_solve -> h_others [ idid ] == 2.0 )  
    {  
      pkj_update_ion_and_matrix ( d_pkj, d_pkj_solve, p_pkj, p_pkj_solve, d_grpkj, d_mlipkj, sum_dt );
      sum_dt = 0.0;
    } 

    if ( p_pkj_solve -> h_others [ idid ] == 2.0 ) { continue; }
    else if ( p_pkj_solve -> h_others [ idid ] == 1.0 ) { break; } // h_ot [ idid ] = 2.0;?
    else { printf ( " !! error !! idid = %f \n ", p_pkj_solve -> h_others [ idid ] ); exit ( 1 ); }
  }

  rkcint <<< numBlocks, numThreadsPerBlock >>>( vec [ v_new ], p_pkj_solve -> h_work [ 0 ], p_pkj_solve -> h_work [ 1 ],
       p_pkj_solve -> h_others [ gt_old ], vec [ vtemp1 ], vec [ vtemp2 ], vec [ yn_r ], vec [ fn ], p_pkj -> nc );
  p_pkj_solve -> h_others [ tend ] += H_MAX;
  //cublasDestroy_v2 ( handle1 );
}

__host__ static
void io_update_ion_and_matrix ( neuron_t *d_io, neuron_solve_t *d_io_solve, 
                                neuron_t *p_io, neuron_solve_t *p_io_solve,
                                gap_t *d_io_gap, double sum_dt )
{    
  double **vec  = p_io_solve -> vec;
  double **elem = p_io -> elem;
  double **ion  = p_io -> ion;
  static int numThreadsPerBlock = p_io_solve -> numThreadsPerBlock;
  static int numBlocks = p_io_solve -> numBlocks;
  io_update_ion_RKC <<< numBlocks, numThreadsPerBlock >>> ( d_io, d_io_solve, vec [ y ], sum_dt ); 
  io_update_matrix <<< numBlocks, numThreadsPerBlock >>>  ( d_io, d_io_solve, vec [ y ] );
  if ( p_io -> n > 1 )
  {
    io_gap_update <<< ( d_io_gap -> n + 127 ) / 128, 128 >>> ( d_io, d_io_gap -> comp, d_io_gap -> elem, d_io_gap -> n );
    add_io_gap_val <<< ( d_io_gap -> n + 127 ) / 128, 128 >>>( d_io_solve, d_io_gap -> comp, d_io_gap -> elem, d_io_gap -> n );
  }
}

__host__ 
void io_solve_by_rkc ( neuron_t *d_io, neuron_solve_t *d_io_solve, 
                       neuron_t *p_io, neuron_solve_t *p_io_solve,
                       gap_t * d_io_gap )
{  
  double **vec  = p_io_solve -> vec;
  static double sum_dt = 0.0;
  static int numThreadsPerBlock = p_io_solve -> numThreadsPerBlock;
  static int numBlocks = p_io_solve -> numBlocks;
  static double integ_current_time = pow ( 2.0, log10 ( p_io_solve -> h_others [ atol_r ] ) );
  //double h_ot [ n_vec_RKC_others ] = { };
  // set cublas
  cublasStatus_t stat1;
  cublasHandle_t handle1;
  stat1 = cublasCreate_v2 ( &handle1 );

  if ( sum_dt != 0.0 )
  {
    io_update_ion_and_matrix ( d_io, d_io_solve, p_io, p_io_solve, d_io_gap, sum_dt );
    sum_dt = 0.0;
  }
  while ( 1 ) 
  {
    rkclow ( p_io, p_io_solve, d_io, d_io_solve, stat1, handle1 );   
    sum_dt += p_io_solve -> h_others [ gt ] - p_io_solve -> h_others [ gt_old ];
    p_io_solve -> h_others [ gt_old ] = p_io_solve -> h_others [ gt ];

    if ( sum_dt > integ_current_time && p_io_solve -> h_others [ idid ] == 2.0 ) 
    {  
      io_update_ion_and_matrix ( d_io, d_io_solve, p_io, p_io_solve, d_io_gap, sum_dt );
      sum_dt = 0.0;
    } 

    if ( p_io_solve -> h_others [ idid ] == 2.0 ) { continue; }
    else if ( p_io_solve -> h_others [ idid ] == 1.0 ) { break; } // h_ot [ idid ] = 2.0;?
    else { printf ( " !! error !! idid = %f \n ", p_io_solve -> h_others [ idid ] ); exit ( 1 ); }
  }

  rkcint <<< numBlocks, numThreadsPerBlock >>>( vec [ v_new ], p_io_solve -> h_work [ 0 ], p_io_solve -> h_work [ 1 ],
       p_io_solve -> h_others [ gt_old ], vec [ vtemp1 ], vec [ vtemp2 ], vec [ yn_r ], vec [ fn ], p_io -> nc );
  p_io_solve -> h_others [ tend ] += H_MAX;
  cublasDestroy_v2 ( handle1 );
}
