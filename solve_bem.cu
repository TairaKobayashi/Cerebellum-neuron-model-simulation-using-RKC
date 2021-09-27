#include "solve_bem.cuh"

//#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <cusparse.h>

__host__
static void cg_cusparse_crs(const int ngc, const int nnz, const double *d_val, 
							const int *d_col, const int *d_row,	double *d_x, double *d_b)
{
    double *d_r, *d_p, *d_ap;
    cudaMalloc ( ( double ** ) &d_r,  ngc * sizeof ( double ) );
    cudaMalloc ( ( double ** ) &d_p,  ngc * sizeof ( double ) );
    cudaMalloc ( ( double ** ) &d_ap, ngc * sizeof ( double ) );
    double *_buffer;
    cudaMalloc ( ( double ** ) &_buffer, ngc * sizeof ( double ) );
    
	double bnorm, rnorm_k, rnorm_k1;
	double alpha, beta, pap;
	double epsilon = 1.0e-15;
	double cp1 = 1.0;
	double c0 = 0.0;
	double cm1 = -1.0;

	cublasStatus_t stat1;
	cublasHandle_t handle1;
	cusparseStatus_t stat2;
	cusparseHandle_t handle2;
	cusparseMatDescr_t descrA;
	stat1 = cublasCreate_v2(&handle1);
	stat2 = cusparseCreate(&handle2);
	cusparseCreateMatDescr(&descrA);
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);


	stat1 = cublasDscal(handle1, ngc, &c0, d_x, 1);				// x = 0
	stat1 = cublasDcopy(handle1, ngc, d_b, 1, d_r, 1);			// r = b
	stat1 = cublasDcopy(handle1, ngc, d_r, 1, d_p, 1);			// p = r
	stat1 = cublasDdot(handle1, ngc, d_b, 1, d_b, 1, &bnorm);	// ||b||

	for (int k = 0; k < 100; k++) {
		//stat2 = cusparseDcsrmv(handle2, CUSPARSE_OPERATION_NON_TRANSPOSE,
		//	ngc, ngc, nnz, &cp1, descrA, d_val, d_row, d_col, d_p, &c0, d_ap);	// Ap
		stat2 = cusparseCsrmvEx(handle2,CUSPARSE_ALG_MERGE_PATH, CUSPARSE_OPERATION_NON_TRANSPOSE,
			ngc, ngc, nnz, &cp1, CUDA_R_64F, descrA, d_val, CUDA_R_64F, d_row, d_col, d_p, CUDA_R_64F, 
			&c0, CUDA_R_64F, d_ap, CUDA_R_64F, CUDA_R_64F, _buffer );	// Ap
		
		stat1 = cublasDdot(handle1, ngc, d_r, 1, d_r, 1, &rnorm_k);				// ||r_k||^2
		stat1 = cublasDdot(handle1, ngc, d_p, 1, d_ap, 1, &pap);				// pAp		
		alpha = rnorm_k / pap;													// alpha		
		stat1 = cublasDaxpy(handle1, ngc, &alpha, d_p, 1, d_x, 1);				// x += alpha * p
		alpha = -1.0 * alpha;
		stat1 = cublasDaxpy(handle1, ngc, &alpha, d_ap, 1, d_r, 1);				// r -= alpha * ap
		stat1 = cublasDdot(handle1, ngc, d_r, 1, d_r, 1, &rnorm_k1);			// ||r_k+1||^2

		if (sqrt(rnorm_k1) <= epsilon * sqrt(bnorm)) { break; }

		// p = r + beta * p
		beta = rnorm_k1 / rnorm_k;
		stat1 = cublasDscal(handle1, ngc, &beta, d_p, 1);
		stat1 = cublasDaxpy(handle1, ngc, &cp1, d_r, 1, d_p, 1);
	}
	cusparseDestroyMatDescr(descrA);
	cublasDestroy_v2(handle1);
	cusparseDestroy(handle2);//
    
    cudaFree ( d_r );
    cudaFree ( d_p );
	cudaFree ( d_ap );
	cudaFree ( _buffer );
}

__global__ static 
void reset_b ( neuron_solve_t *d_solve, const int nc )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < nc ) 
  {
	d_solve -> b [ id ] = 0.0;
  }
}

__global__
static void create_transposed_matrix ( const int ngc, const int nnz, 
  const double *d_val, const int *d_col, const int *d_row, double *val_h, int *col_h, int *row_h )
{
  row_h [ 0 ] = 0;
  int count_row = 0;
  int n_v = 0;
  int n_r = 0;

  for ( int i = 0; i < ngc; i++ ) {
    for ( int j = 0; j < ngc; j++ ) {
      for ( int k = d_row [ j ]; k < d_row [ j + 1 ]; k++ ) {
        if ( d_col [ k ] == i ) {
          val_h [ n_v ] = d_val [ k ];
          col_h [ n_v ] = j;
          n_v++;          
        }        
      }
    }
    row_h [ n_r + 1 ] = n_v;
    n_r++;
  }
  // Debug
  /*
  for ( int i = 0; i < nnz; i++ )
    printf ( "val [ %d ] = %f\n", i, d_val [ i ] );
  for ( int i = 0; i < nnz; i++ ) 
    printf ( "val_h [ %d ] = %f\n", i, val_h [ i ] );
  for ( int i = 0; i < nnz; i++ )
    printf ( "col [ %d ] = %d\n", i, d_col [ i ] );
  for ( int i = 0; i < nnz; i++ ) 
    printf ( "col_h [ %d ] = %d\n", i, col_h [ i ] );
  for ( int i = 0; i < ngc + 1; i++ )
    printf ( "row [ %d ] = %d\n", i, d_row [ i ] );
  for ( int i = 0; i < ngc + 1; i++ )
    printf ( "row_h [ %d ] = %d\n", i, row_h [ i ] );
  */

}

__host__
static void bicg_cusparse_crs( const int ngc, const int nnz, 
  const double *d_val, const int *d_col, const int *d_row, double *d_x, double *d_b)
{
  static double *d_r, *d_p, *d_ap;
  static double *d_rs, *d_ps, *d_atps;
  static double *_buffer;
  static double *val_h;  static int *col_h, *row_h;


  static int size = 0;
  if ( size < ngc ) {
    if ( size == 0 ) {
        cudaFree ( d_r );
        cudaFree ( d_p );
        cudaFree ( d_ap );
        cudaFree ( d_rs );
        cudaFree ( d_ps );
        cudaFree ( d_atps );
        cudaFree ( _buffer );
        cudaFree ( val_h );
        cudaFree ( col_h );
        cudaFree ( row_h );
    }
    cudaMalloc ( ( double ** ) &d_r,   ngc * sizeof ( double ) );
    cudaMalloc ( ( double ** ) &d_p,   ngc * sizeof ( double ) );
    cudaMalloc ( ( double ** ) &d_ap,  ngc * sizeof ( double ) );
    cudaMalloc ( ( double ** ) &d_rs,  ngc * sizeof ( double ) );
    cudaMalloc ( ( double ** ) &d_ps,  ngc * sizeof ( double ) );
    cudaMalloc ( ( double ** ) &d_atps,  ngc * sizeof ( double ) );
    cudaMalloc ( ( double ** ) &val_h, nnz * sizeof ( double ) );
    cudaMalloc ( ( int ** )    &col_h, nnz * sizeof ( int ) );
    cudaMalloc ( ( int ** )    &row_h, ( ngc + 1 ) * sizeof ( int ) );
    cudaMalloc ( ( double ** ) &_buffer, ngc * sizeof ( double ) );
    size = ngc;
  }
  
  create_transposed_matrix <<< 1, 1 >>> ( ngc, nnz, d_val, d_col, d_row, val_h, col_h, row_h );
  cudaDeviceSynchronize();

  //cudaMalloc ( ( double ** ) &d_r,  ngc * sizeof ( double ) );
  //cudaMalloc ( ( double ** ) &d_p,  ngc * sizeof ( double ) );
  //cudaMalloc ( ( double ** ) &d_ap, ngc * sizeof ( double ) );
    
  double bnorm, rnorm_k, rnorm_k1;
  double alpha, beta, pap;
  double epsilon = 1.0e-15;
  double cp1 = 1.0;
  double c0 = 0.0;
  double cm1 = -1.0;
  
  cublasStatus_t stat1;
  cublasHandle_t handle1;
  cusparseStatus_t stat2;
  cusparseHandle_t handle2;
  cusparseMatDescr_t descrA;
  stat1 = cublasCreate_v2(&handle1);
  stat2 = cusparseCreate(&handle2);
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
 
  stat1 = cublasDscal(handle1, ngc, &c0, d_x, 1);		// x = 0
  stat1 = cublasDcopy(handle1, ngc, d_b, 1, d_r, 1);	// r = b
  stat1 = cublasDcopy(handle1, ngc, d_r, 1, d_rs, 1);   // rs = r
  stat1 = cublasDcopy(handle1, ngc, d_r, 1, d_p, 1);	// p = r
  stat1 = cublasDcopy(handle1, ngc, d_rs, 1, d_ps, 1);	// ps = rs
  stat1 = cublasDdot(handle1, ngc, d_b, 1, d_b, 1, &bnorm);	// ||b||
  /**/
  for (int k = 0; k < 100; k++) {
	stat2 = cusparseCsrmvEx(handle2,CUSPARSE_ALG_MERGE_PATH, CUSPARSE_OPERATION_NON_TRANSPOSE,
	  ngc, ngc, nnz, &cp1, CUDA_R_64F, descrA, d_val, CUDA_R_64F, d_row, d_col, d_p, CUDA_R_64F, 
      &c0, CUDA_R_64F, d_ap, CUDA_R_64F, CUDA_R_64F, _buffer );	    // Ap
    stat2 = cusparseCsrmvEx(handle2,CUSPARSE_ALG_MERGE_PATH, CUSPARSE_OPERATION_NON_TRANSPOSE,
      ngc, ngc, nnz, &cp1, CUDA_R_64F, descrA, val_h, CUDA_R_64F, row_h, col_h, d_ps, CUDA_R_64F, 
      &c0, CUDA_R_64F, d_atps, CUDA_R_64F, CUDA_R_64F, _buffer );   // Atps
    
    stat1 = cublasDdot(handle1, ngc, d_r, 1, d_rs, 1, &rnorm_k);		// ( rs, r )
    stat1 = cublasDdot(handle1, ngc, d_ps, 1, d_ap, 1, &pap);		  	// psAp		
    alpha = rnorm_k / pap;												// alpha		
    stat1 = cublasDaxpy(handle1, ngc, &alpha, d_p, 1, d_x, 1);			// x += alpha * p
    alpha = -1.0 * alpha;
    stat1 = cublasDaxpy(handle1, ngc, &alpha, d_ap, 1, d_r, 1);			// r -= alpha * ap
    stat1 = cublasDaxpy(handle1, ngc, &alpha, d_atps, 1, d_rs, 1);		// rs -= alpha * atps

	stat1 = cublasDdot(handle1, ngc, d_r, 1, d_rs, 1, &rnorm_k1);		// ( r_k+1, rs_k+1 )
	
	if (sqrt(rnorm_k1) <= epsilon * sqrt(bnorm)) { break; }
	
    // beta
    beta = rnorm_k1 / rnorm_k;
    // p = r + beta * p
    stat1 = cublasDscal(handle1, ngc, &beta, d_p, 1);
    stat1 = cublasDaxpy(handle1, ngc, &cp1, d_r, 1, d_p, 1);
    // ps = rs + beta * ps
    stat1 = cublasDscal(handle1, ngc, &beta, d_ps, 1);
    stat1 = cublasDaxpy(handle1, ngc, &cp1, d_rs, 1, d_ps, 1);
  }
  cusparseDestroyMatDescr(descrA);
  cublasDestroy_v2(handle1);
  cusparseDestroy(handle2);//
      
  //cudaFree ( d_r );
  //cudaFree ( d_p );
  //cudaFree ( d_ap );
}
  
//////////////////////////////// GR /////////////////////////////////
__global__ static
void add_mfgr_val ( neuron_t *d_gr, neuron_solve_t *d_gr_solve, int *mfgr_comp, double *mfgr_elem, const int num_mfgr )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < num_mfgr )
  {
    int post_id = mfgr_comp [ post_comp * num_mfgr + id ];
	int val_id = d_gr_solve -> dig [ post_id ];
	double l_val = mfgr_elem [ mfgr_val * num_mfgr + id ];
    atomicAdd ( & ( d_gr_solve -> val [ val_id ] ), l_val ); // no need atomicAdd ?
    atomicAdd ( & ( d_gr_solve -> b [ post_id ]  ), l_val * E_MFGR );
  }
}

__global__ static
void add_gogr_val ( neuron_t *d_gr, neuron_solve_t *d_gr_solve, int *gogr_comp, double *gogr_elem, const int num_gogr )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < num_gogr )
  {
    int post_id = gogr_comp [ post_comp * num_gogr + id ];
	int val_id = d_gr_solve -> dig [ post_id ];
	double l_val = gogr_elem [ gogr_val * num_gogr + id ];
    atomicAdd ( & ( d_gr_solve -> val [ val_id ] ), l_val ); // no need atomicAdd ?
    atomicAdd ( & ( d_gr_solve -> b [ post_id ]  ), l_val * E_GOGR );
  }
}

__global__
static void gr_update_matrix ( neuron_t *d_gr, neuron_solve_t *d_gr_solve )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  double **elem = d_gr -> elem;
  double **cond = d_gr -> cond;
  double **ion  = d_gr -> ion;
  double DT = d_gr -> DT;
  if ( id < d_gr -> nc ) 
  {
    int d = d_gr_solve -> dig [ id ];
	d_gr_solve -> val [ d ] 
	  += ( ( elem [ Cm ] [ id ] / DT )
	  + cond [ g_leak1 ] [ id ] + cond [ g_leak2 ] [ id ] + cond [ g_leak3 ] [ id ] 
      + cond [ g_Na ]  [ id ] * ion [ o_Na ] [ id ]
      + cond [ g_Ca ]  [ id ] * ion [ ch_Ca ] [ id ] * ion [ ch_Ca ] [ id ] * ion [ ci_Ca ] [ id ]
	  + cond [ g_KV ]  [ id ] * ion [ n_KV ] [ id ] * ion [ n_KV ] [ id ] * ion [ n_KV ] [ id ] * ion [ n_KV ] [ id ]
	  + cond [ g_KIR ] [ id ] *  ion [ ir_KIR ] [ id ]
	  + cond [ g_KA ]  [ id ] *  ion [ a_KA ] [ id ] * ion [ a_KA ] [ id ] * ion [ a_KA ] [ id ] * ion [ b_KA ] [ id ]
	  + cond [ g_KCa ] [ id ] *  ion [ c_KCa ] [ id ]
	  + cond [ g_KM ]  [ id ] *  ion [ s_KM ] [ id ] );
		
	d_gr_solve -> b [ id ]  
	  += ( ( elem [ Cm ] [ id ] / DT ) * elem [ v ] [ id ]
	  + cond [ g_leak1 ] [ id ] * V_LEAK1_GR + cond [ g_leak2 ] [ id ] * V_LEAK2_GR + cond [ g_leak3 ] [ id ] * V_LEAK3_GR + elem [ i_ext ] [ id ]
	  + cond [ g_Na    ] [ id ] * V_Na_GR * ion [ o_Na   ] [ id ]
	  + cond [ g_Ca    ] [ id ] * V_Ca_GR * ion [ ch_Ca  ] [ id ] * ion [ ch_Ca ] [ id ] * ion [ ci_Ca ] [ id ]
	  + cond [ g_KV    ] [ id ] * V_K_GR  * ion [ n_KV   ] [ id ] * ion [ n_KV  ] [ id ] * ion [ n_KV  ] [ id ] * ion [ n_KV ] [ id ]
	  + cond [ g_KIR   ] [ id ] * V_K_GR  * ion [ ir_KIR ] [ id ]
	  + cond [ g_KA    ] [ id ] * V_K_GR  * ion [ a_KA   ] [ id ] * ion [ a_KA  ] [ id ] * ion [ a_KA  ] [ id ] * ion [ b_KA ] [ id ]
	  + cond [ g_KCa   ] [ id ] * V_K_GR  * ion [ c_KCa  ] [ id ]
	  + cond [ g_KM    ] [ id ] * V_K_GR  * ion [ s_KM   ] [ id ] );
  }
}

__host__
void gr_solve_by_bem ( neuron_t *d_gr, neuron_solve_t *d_gr_solve, 
					   neuron_t *p_gr, neuron_solve_t *p_gr_solve, 
					   synapse_t *d_mfgr, synapse_t *d_gogr )
{  
	// global
  double **ion  = p_gr -> ion;
  double **elem = p_gr -> elem;
  int nc = p_gr -> nc;
  static int numThreadsPerBlock = p_gr_solve -> numThreadsPerBlock;
  static int numBlocks = p_gr_solve -> numBlocks; 
  
  // update ion
  gr_Na_update <<< numBlocks, numThreadsPerBlock >>>
   ( nc, elem [ v ], BE_DT, elem [ compart ],
     ion [ o_Na ],  ion [ c1_Na ], ion [ c2_Na ], ion [ c3_Na ], ion [ c4_Na ], ion [ c5_Na ],
     ion [ i1_Na ], ion [ i2_Na ], ion [ i3_Na ], ion [ i4_Na ], ion [ i5_Na ], ion [ i6_Na ] );
  gr_update_ion <<< numBlocks, numThreadsPerBlock >>> ( d_gr, d_gr_solve, BE_DT );

  // reset val and b
  cudaMemcpy ( p_gr_solve -> val,  p_gr_solve -> val_ori, p_gr_solve -> nnz * sizeof ( double ), cudaMemcpyDeviceToDevice );
  reset_b <<< numBlocks, numThreadsPerBlock >>> ( d_gr_solve, p_gr -> nc );

  // update val, b and v
  add_mfgr_val <<< ( d_mfgr -> n + 127 ) / 128, 128 >>> 
    ( d_gr, d_gr_solve, d_mfgr -> comp, d_mfgr -> elem, d_mfgr -> n );          //cudaDeviceSynchronize();
  add_gogr_val <<< ( d_gogr -> n + 127 ) / 128, 128 >>> 
	( d_gr, d_gr_solve, d_gogr -> comp, d_gogr -> elem, d_gogr -> n );          //cudaDeviceSynchronize();
  gr_update_matrix <<< numBlocks, numThreadsPerBlock >>> ( d_gr, d_gr_solve );  //cudaDeviceSynchronize();
  cg_cusparse_crs( nc, p_gr_solve -> nnz, p_gr_solve -> val, p_gr_solve -> col, p_gr_solve -> row, elem [ v ], p_gr_solve -> b );
}

//////////////////////////////// GO /////////////////////////////////

__global__ static
void add_grgo_val ( neuron_t *d_go, neuron_solve_t *d_go_solve, int *grgo_comp, double *grgo_elem, const int num_grgo )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < num_grgo )
  {
    int post_id = grgo_comp [ post_comp * num_grgo + id ];
	int val_id = d_go_solve -> dig [ post_id ];
	double l_val = grgo_elem [ grgo_val * num_grgo + id ];
    atomicAdd ( & ( d_go_solve -> val [ val_id ] ), l_val ); // no need atomicAdd ?
    atomicAdd ( & ( d_go_solve -> b [ post_id ]  ), l_val * E_GRGO );
  }
}

__global__
static void go_update_matrix ( neuron_t *d_go, neuron_solve_t *d_go_solve )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  double **elem = d_go -> elem;
  double **cond = d_go -> cond;
  double **ion  = d_go -> ion;
  double DT = d_go -> DT;
  if ( id < d_go -> nc) 
  {
    int d = d_go_solve -> dig [ id ];
    d_go_solve -> val [ d ] += ( ( elem [ Cm ] [ id ] / DT )
			       + cond [ g_leak_go ] [ id ]
			       + cond [ g_NaT_go ] [ id ] * ion [ m_NaT_go ] [ id ] * ion [ m_NaT_go ] [ id ] * ion [ m_NaT_go ] [ id ] * ion [ h_NaT_go ] [ id ]
			       + cond [ g_NaR_go ] [ id ] * ion [ r_NaR_go ] [ id ] * ion [ s_NaR_go ] [ id ]
			       + cond [ g_NaP_go ] [ id ] * ion [ p_NaP_go ] [ id ]
			       + cond [ g_CaHVA_go ] [ id ] * ion [ ch_CaHVA_go ] [ id ] * ion [ ch_CaHVA_go ] [ id ] * ion [ ci_CaHVA_go ] [ id ]
			       + cond [ g_CaLVA_go ] [ id ] * ion [ cl_CaLVA_go ] [ id ] * ion [ cl_CaLVA_go ] [ id ] * ion [ cm_CaLVA_go ] [ id ]
			       + cond [ g_KAHP_go ] [ id ] * ( ion [ o1_KAHP_go ] [ id ] + ion [ o2_KAHP_go ] [ id ] )
			       + cond [ g_KV_go ] [ id ] * ion [ n_KV_go ] [ id ] * ion [ n_KV_go ] [ id ] * ion [ n_KV_go ] [ id ] * ion [ n_KV_go ] [ id ]
			       + cond [ g_KA_go ] [ id ] * ion [ a_KA_go ] [ id ] * ion [ a_KA_go ] [ id ] * ion [ a_KA_go ] [ id ] * ion [ b_KA_go ] [ id ]
			       + cond [ g_KC_go ] [ id ] * ion [ c_KC_go ] [ id ]
			       + cond [ g_Kslow_go ] [ id ] * ion [ sl_Kslow_go ] [ id ]
			       + cond [ g_HCN1_go ] [ id ] * ( ion [ hf_HCN1_go ] [ id ] + ion [ hs_HCN1_go ] [ id ] )
			       + cond [ g_HCN2_go ] [ id ] * ( ion [ hf_HCN2_go ] [ id ] + ion [ hs_HCN2_go ] [ id ] ) );
 
	d_go_solve -> b [ id ]    += ( ( elem [ Cm ] [ id ] / DT ) * elem [ v ] [ id ]
			       + cond [ g_leak_go ] [ id ] * V_LEAK_GO + elem [ i_ext ] [ id ]
			       + cond [ g_NaT_go ] [ id ] * V_Na_GO * ion [ m_NaT_go ] [ id ] * ion [ m_NaT_go ] [ id ] * ion [ m_NaT_go ] [ id ] * ion [ h_NaT_go ] [ id ]
			       + cond [ g_NaR_go ] [ id ] * V_Na_GO * ion [ r_NaR_go ] [ id ] * ion [ s_NaR_go ] [ id ]
			       + cond [ g_NaP_go ] [ id ] * V_Na_GO * ion [ p_NaP_go ] [ id ]
			       + cond [ g_CaHVA_go ] [ id ] * V_Ca_GO * ion [ ch_CaHVA_go ] [ id ] * ion [ ch_CaHVA_go ] [ id ] * ion [ ci_CaHVA_go ] [ id ]
			       + cond [ g_CaLVA_go ] [ id ] * ( d_go -> rev_ca2 [ id ] ) * ion [ cl_CaLVA_go ] [ id ] * ion [ cl_CaLVA_go ] [ id ] * ion [ cm_CaLVA_go ] [ id ]
			       + cond [ g_KAHP_go ] [ id ] * V_K_GO * ( ion [ o1_KAHP_go ] [ id ] + ion [ o2_KAHP_go ] [ id ] )
			       + cond [ g_KV_go ] [ id ] * V_K_GO  * ion [ n_KV_go ] [ id ] * ion [ n_KV_go ] [ id ] * ion [ n_KV_go ] [ id ] * ion [ n_KV_go ] [ id ]
			       + cond [ g_KA_go ] [ id ] * V_K_GO  * ion [ a_KA_go ] [ id ] * ion [ a_KA_go ] [ id ] * ion [ a_KA_go ] [ id ] * ion [ b_KA_go ] [ id ]
			       + cond [ g_KC_go ] [ id ] * V_K_GO * ion [ c_KC_go ] [ id ]
			       + cond [ g_Kslow_go ] [ id ] * V_K_GO * ion [ sl_Kslow_go ] [ id ]
			       + cond [ g_HCN1_go ] [ id ] * V_H_GO * ( ion [ hf_HCN1_go ] [ id ] + ion [ hs_HCN1_go ] [ id ] )
			       + cond [ g_HCN2_go ] [ id ] * V_H_GO * ( ion [ hf_HCN2_go ] [ id ] + ion [ hs_HCN2_go ] [ id ] ) );
  }
}
__host__
void go_solve_by_bem ( neuron_t *d_go, neuron_solve_t *d_go_solve, 
					   neuron_t *p_go, neuron_solve_t *p_go_solve, synapse_t *d_grgo )
{  
	// global
  double **ion  = p_go -> ion;
  int nc = p_go -> nc;
  int numThreadsPerBlock = p_go_solve -> numThreadsPerBlock;
  int numBlocks = p_go_solve -> numBlocks;
  
  // update ion
  go_KAHP_update <<< numBlocks, numThreadsPerBlock >>>
   ( p_go -> n, p_go -> elem [ Ca ], ion [ o1_KAHP_go ], ion [ o2_KAHP_go ], ion [ c1_KAHP_go ],
           ion [ c2_KAHP_go ], ion [ c3_KAHP_go ], ion [ c4_KAHP_go ], BE_DT );
  go_update_ion <<< numBlocks, numThreadsPerBlock >>> ( d_go, d_go_solve, BE_DT );
  
  // reset val and b
  cudaMemcpy ( p_go_solve -> val,  p_go_solve -> val_ori, p_go_solve -> nnz * sizeof ( double ), cudaMemcpyDeviceToDevice );
  reset_b <<< numBlocks, numThreadsPerBlock >>> ( d_go_solve, nc );
  
  // update val, b and v
  add_grgo_val <<< ( d_grgo -> n + 127 ) / 128, 128 >>> 
    ( d_go, d_go_solve, d_grgo -> comp, d_grgo -> elem, d_grgo -> n );          //cudaDeviceSynchronize();
  go_update_matrix <<< numBlocks, numThreadsPerBlock >>> ( d_go, d_go_solve);
  cg_cusparse_crs( nc, p_go_solve -> nnz, p_go_solve -> val,
     p_go_solve -> col, p_go_solve -> row, p_go -> elem [ v ], p_go_solve -> b );
}

//////////////////////////////// PKJ /////////////////////////////////

static __global__ 
void pkj_update_matrix ( neuron_t *d_pkj, neuron_solve_t *d_pkj_solve )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  if ( id < d_pkj -> nc) 
  {
    double **elem = d_pkj -> elem;
    double **cond = d_pkj -> cond;
    double **ion  = d_pkj -> ion;  
    double DT = d_pkj -> DT;
    int d = d_pkj_solve -> dig [ id ];
    double l_v_Ca = d_pkj -> rev_ca2 [ id ] ;

    d_pkj_solve -> val [ d ] += ( ( elem [ Cm ] [ id ] / DT )
			       + cond [ g_leak_pkj ] [ id ]
			       + cond [ g_NaF_pkj ] [ id ] * ion [ m_NaF_pkj ] [ id ] * ion [ m_NaF_pkj ] [ id ] * ion [ m_NaF_pkj ] [ id ] * ion [ h_NaF_pkj ] [ id ]
			       + cond [ g_NaP_pkj ] [ id ] * ion [ m_NaP_pkj ] [ id ] * ion [ m_NaP_pkj ] [ id ] * ion [ m_NaP_pkj ] [ id ]
			       + cond [ g_CaP_pkj ] [ id ] * ion [ m_CaP_pkj ] [ id ] * ion [ h_CaP_pkj ] [ id ]
			       + cond [ g_CaT_pkj ] [ id ] * ion [ m_CaT_pkj ] [ id ] * ion [ h_CaT_pkj ] [ id ]
			       + cond [ g_Kh_pkj ] [ id ] * ion [ m_Kh1_pkj ] [ id ]
			       + cond [ g_Kh_pkj ] [ id ] * ion [ m_Kh2_pkj ] [ id ]
			       + cond [ g_Kdr_pkj ] [ id ] * ion [ m_Kdr_pkj ] [ id ] * ion [ m_Kdr_pkj ] [ id ] * ion [ h_Kdr_pkj ] [ id ]
			       + cond [ g_KM_pkj ] [ id ] * ion [ m_KM_pkj ] [ id ]
			       + cond [ g_KA_pkj ] [ id ] * ion [ m_KA_pkj ] [ id ] * ion [ m_KA_pkj ] [ id ] * ion [ m_KA_pkj ] [ id ] * ion [ m_KA_pkj ] [ id ] * ion [ h_KA_pkj ] [ id ]
			       + cond [ g_KC_pkj ] [ id ] * ion [ m_KC_pkj ] [ id ] * ion [ z_KC_pkj ] [ id ] * ion [ z_KC_pkj ] [ id ]
				   + cond [ g_K2_pkj ] [ id ] * ion [ m_K2_pkj ] [ id ] * ion [ z_K2_pkj ] [ id ] * ion [ z_K2_pkj ] [ id ] 
				);

	d_pkj_solve -> b [ id ]    += ( ( elem [ Cm ] [ id ] / DT ) * elem [ v ] [ id ] + elem [ i_ext ] [ id ]
			       + cond [ g_leak_pkj ] [ id ] * ( V_LEAK_PKJ )
			       + cond [ g_NaF_pkj ] [ id ] * ion [ m_NaF_pkj ] [ id ] * ion [ m_NaF_pkj ] [ id ] * ion [ m_NaF_pkj ] [ id ] * ion [ h_NaF_pkj ] [ id ] * ( V_Na_PKJ )
			       + cond [ g_NaP_pkj ] [ id ] * ion [ m_NaP_pkj ] [ id ] * ion [ m_NaP_pkj ] [ id ] * ion [ m_NaP_pkj ] [ id ] * ( V_Na_PKJ )
			       + cond [ g_CaP_pkj ] [ id ] * ion [ m_CaP_pkj ] [ id ] * ion [ h_CaP_pkj ] [ id ] * ( l_v_Ca )
			       + cond [ g_CaT_pkj ] [ id ] * ion [ m_CaT_pkj ] [ id ] * ion [ h_CaT_pkj ] [ id ] * ( l_v_Ca )
			       + cond [ g_Kh_pkj ] [ id ] * ion [ m_Kh1_pkj ] [ id ] * ( V_KH_PKJ ) //KH????
			       + cond [ g_Kh_pkj ] [ id ] * ion [ m_Kh2_pkj ] [ id ] * ( V_KH_PKJ ) //KH???
			       + cond [ g_Kdr_pkj ] [ id ] * ion [ m_Kdr_pkj ] [ id ] * ion [ m_Kdr_pkj ] [ id ] * ion [ h_Kdr_pkj ] [ id ] * ( V_K_PKJ )
			       + cond [ g_KM_pkj ] [ id ] * ion [ m_KM_pkj ] [ id ] * ( V_K_PKJ )
			       + cond [ g_KA_pkj ] [ id ] * ion [ m_KA_pkj ] [ id ] * ion [ m_KA_pkj ] [ id ] * ion [ m_KA_pkj ] [ id ] * ion [ m_KA_pkj ] [ id ] * ion [ h_KA_pkj ] [ id ] * ( V_K_PKJ )
			       + cond [ g_KC_pkj ] [ id ] * ion [ m_KC_pkj ] [ id ] * ion [ z_KC_pkj ] [ id ] * ion [ z_KC_pkj ] [ id ] * ( V_K_PKJ )
				   + cond [ g_K2_pkj ] [ id ] * ion [ m_K2_pkj ] [ id ] * ion [ z_K2_pkj ] [ id ] * ion [ z_K2_pkj ] [ id ]  * ( V_K_PKJ )
				);
  }
}
__host__
void pkj_solve_by_bem ( neuron_t *d_pkj, neuron_solve_t *d_pkj_solve, 
				 	    neuron_t *p_pkj, neuron_solve_t *p_pkj_solve )
{ 
  double **ion  = p_pkj -> ion;
  int nc = p_pkj -> nc;
  int numThreadsPerBlock = p_pkj_solve -> numThreadsPerBlock;
  int numBlocks = p_pkj_solve -> numBlocks;
  
  // update ion
  pkj_update_ion <<< numBlocks, numThreadsPerBlock >>> ( d_pkj, d_pkj_solve, BE_DT );
  
  // reset val and b
  cudaMemcpy ( p_pkj_solve -> val,  p_pkj_solve -> val_ori, p_pkj_solve -> nnz * sizeof ( double ), cudaMemcpyDeviceToDevice );
  reset_b <<< numBlocks, numThreadsPerBlock >>> ( d_pkj_solve, nc );
  
  // update val, b and v
  pkj_update_matrix <<< numBlocks, numThreadsPerBlock >>> ( d_pkj, d_pkj_solve);
  cg_cusparse_crs( nc, p_pkj_solve -> nnz, p_pkj_solve -> val,
     p_pkj_solve -> col, p_pkj_solve -> row, p_pkj -> elem [ v ], p_pkj_solve -> b );
}


///////////////////////////////////////// IO ///////////////////////////////////////////////

__global__ static
void add_io_gap_val ( neuron_solve_t *d_io_solve, int *io_gap_comp, double *io_gap_elem, const int num_io_gap )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < num_io_gap ) 
  { 
    double l_val = -1 * io_gap_elem [ gap_current * num_io_gap + id ];
    int post_num = io_gap_comp [ post_comp_gap  * num_io_gap + id ];    
    atomicAdd ( & ( d_io_solve -> b [ post_num ] ), ( l_val ) );
  }
}

__global__
static void io_update_matrix ( neuron_t *d_io, neuron_solve_t *d_io_solve )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  double **elem = d_io -> elem;
  double **cond = d_io -> cond;
  double **ion  = d_io -> ion;
  double DT = d_io -> DT;
  if ( id < d_io -> nc) 
  {
    int d = d_io_solve -> dig [ id ];
	d_io_solve -> val [ d ] 
	  += ( ( elem [ Cm ] [ id ] / DT )
	  + cond [ g_leak_io ] [ id ]
	  + cond [ g_CaL_io  ] [ id ] * ion [ k_CaL_io ] [ id ] * ion [ k_CaL_io ] [ id ] * ion [ k_CaL_io ] [ id ] * ion [ l_CaL_io ] [ id ]
	  + cond [ g_Na_io   ] [ id ] * ion [ m_Na_io  ] [ id ] * ion [ m_Na_io  ] [ id ] * ion [ m_Na_io  ] [ id ] * ion [ h_Na_io  ] [ id ]
  	  + cond [ g_Kdr_io  ] [ id ] * ion [ n_Kdr_io ] [ id ] * ion [ p_Kdr_io ] [ id ] 
  	  + cond [ g_K_io    ] [ id ] * ion [ x_K_io   ] [ id ] * ion [ x_K_io   ] [ id ] * ion [ x_K_io   ] [ id ] * ion [ x_K_io   ] [ id ] 
      + cond [ g_CaH_io  ] [ id ] * ion [ r_CaH_io ] [ id ] * ion [ r_CaH_io ] [ id ]
      + cond [ g_KCa_io  ] [ id ] * ion [ s_KCa_io ] [ id ]
      + cond [ g_H_io    ] [ id ] * ion [ q_H_io   ] [ id ] );
 
	d_io_solve -> b [ id ]
	  += ( ( elem [ Cm ] [ id ] / DT ) * elem [ v ] [ id ]
	  + cond [ g_leak_io ] [ id ] * V_LEAK_IO + elem [ i_ext ] [ id ]
	  + cond [ g_CaL_io  ] [ id ] * V_Ca_IO * ion [ k_CaL_io ] [ id ] * ion [ k_CaL_io ] [ id ] * ion [ k_CaL_io ] [ id ] * ion [ l_CaL_io ] [ id ]
	  + cond [ g_Na_io   ] [ id ] * V_Na_IO * ion [ m_Na_io  ] [ id ] * ion [ m_Na_io  ] [ id ] * ion [ m_Na_io  ] [ id ] * ion [ h_Na_io  ] [ id ]
	  + cond [ g_Kdr_io  ] [ id ] * V_K_IO  * ion [ n_Kdr_io ] [ id ] * ion [ p_Kdr_io ] [ id ] 
	  + cond [ g_K_io    ] [ id ] * V_K_IO  * ion [ x_K_io   ] [ id ] * ion [ x_K_io   ] [ id ] * ion [ x_K_io   ] [ id ] * ion [ x_K_io   ] [ id ] 
	  + cond [ g_CaH_io  ] [ id ] * V_Ca_IO * ion [ r_CaH_io ] [ id ] * ion [ r_CaH_io ] [ id ]
	  + cond [ g_KCa_io  ] [ id ] * V_K_IO  * ion [ s_KCa_io ] [ id ]
	  + cond [ g_H_io    ] [ id ] * V_H_IO  * ion [ q_H_io   ] [ id ] );
  }
}
__host__
void io_solve_by_bem ( neuron_t *d_io, neuron_solve_t *d_io_solve, 
					   neuron_t *p_io, neuron_solve_t *p_io_solve, gap_t* d_io_gap )
{  
	// global
  double **ion  = p_io -> ion;
  int nc = p_io -> nc;
  int numThreadsPerBlock = p_io_solve -> numThreadsPerBlock;
  int numBlocks = p_io_solve -> numBlocks;
  
  // update ion
  io_update_ion <<< numBlocks, numThreadsPerBlock >>> ( d_io, d_io_solve, BE_DT );
 
  // reset val and b
  cudaMemcpy ( p_io_solve -> val,  p_io_solve -> val_ori, p_io_solve -> nnz * sizeof ( double ), cudaMemcpyDeviceToDevice );
  reset_b <<< numBlocks, numThreadsPerBlock >>> ( d_io_solve, nc );
  
  // update val, b and v 
  io_update_matrix <<< numBlocks, numThreadsPerBlock >>> ( d_io, d_io_solve );
  if ( p_io -> n > 1 )
  {
    io_gap_update <<< ( d_io_gap -> n + 127 ) / 128, 128 >>> ( d_io, d_io_gap -> comp, d_io_gap -> elem, d_io_gap -> n );
    add_io_gap_val <<< ( d_io_gap -> n + 127 ) / 128, 128 >>>( d_io_solve, d_io_gap -> comp, d_io_gap -> elem, d_io_gap -> n );
  }
  bicg_cusparse_crs ( nc, p_io_solve -> nnz, p_io_solve -> val,
	 p_io_solve -> col, p_io_solve -> row, p_io -> elem [ v ], p_io_solve -> b );
}
