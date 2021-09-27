#include "go_ion.cuh"

//Na channel
__host__ __device__ static double alpha_m_NaT(const double v) { return 0.417*(v + 25.0) / (1 - exp(-(v + 25.0) / 10.0)); }
__host__ __device__ static double beta_m_NaT(const double v) { return 16.68*exp(-0.055*(v + 50.0)); }
__host__ __device__ static double inf_m_NaT(const double v) { return alpha_m_NaT(v) / (alpha_m_NaT(v) + beta_m_NaT(v)); }
__host__ __device__ static double tau_m_NaT(const double v) { return 1.0 / (alpha_m_NaT(v) + beta_m_NaT(v)); }

__host__ __device__ static double alpha_h_NaT(const double v) { return 0.292*exp(-0.3*(v + 50.0)); }
__host__ __device__ static double beta_h_NaT(const double v) { return 4.17 / (1 + exp(-(v + 17.0) / 5.0)); }
__host__ __device__ static double inf_h_NaT(const double v) { return alpha_h_NaT(v) / (alpha_h_NaT(v) + beta_h_NaT(v)); }
__host__ __device__ static double tau_h_NaT(const double v) { return 1.0 / (alpha_h_NaT(v) + beta_h_NaT(v)); }

__host__ __device__ static double alpha_r_NaR(const double v) { return ((1.11 - 68.5*(v - 4.48) / (exp(-(v - 4.48) / 6.8) - 1.0))*1.0e-4); }
__host__ __device__ static double beta_r_NaR(const double v) { double x = (v + 44.0) / 0.11; if (x > 200.0)x = 200.0; return ((66.0 + 21.7*(v + 44) / (exp(x) - 1.0))*1.0e-3); }
__host__ __device__ static double inf_r_NaR(const double v) { return alpha_r_NaR(v) / (alpha_r_NaR(v) + beta_r_NaR(v)); }
__host__ __device__ static double tau_r_NaR(const double v) { return 1.0 / (alpha_r_NaR(v) + beta_r_NaR(v)); }

__host__ __device__ static double alpha_s_NaR(const double v) { return (0.443*exp(-(v + 80.0) / 62.5)); }
__host__ __device__ static double beta_s_NaR(const double v) { return (0.014*exp((v + 83.3) / 16.1)); }
__host__ __device__ static double inf_s_NaR(const double v) { return alpha_s_NaR(v) / (alpha_s_NaR(v) + beta_s_NaR(v)); }
__host__ __device__ static double tau_s_NaR(const double v) { return 1.0 / (alpha_s_NaR(v) + beta_s_NaR(v)); }

__host__ __device__ static double alpha_p_NaP(const double v) { return (0.421*(v + 40.0) / (1.0 - exp(-(v + 40.0) / 5.0))); }
__host__ __device__ static double beta_p_NaP(const double v) { return (-0.287*(v + 40.0) / (1.0 - exp((v + 40.0) / 5.0))); }
__host__ __device__ static double inf_p_NaP(const double v) { return (1 / (1.0 + exp(-(v + 43.0) / 5.0))); }
__host__ __device__ static double tau_p_NaP(const double v) { return (5.0 / (alpha_p_NaP(v) + beta_p_NaP(v))); }

//Ca channel
__host__ __device__ static double alpha_ch_CaHVA(const double v) { return (0.0687 * exp(0.063*(v + 29.0))); }
__host__ __device__ static double beta_ch_CaHVA(const double v) { return  (0.115 * exp(-0.039*(v + 18.66))); }
__host__ __device__ static double inf_ch_CaHVA(const double v) { return (alpha_ch_CaHVA(v) / (alpha_ch_CaHVA(v) + beta_ch_CaHVA(v))); }
__host__ __device__ static double tau_ch_CaHVA(const double v) { return (1.0 / (alpha_ch_CaHVA(v) + beta_ch_CaHVA(v))); }

__host__ __device__ static double alpha_ci_CaHVA(const double v) { return (1.8e-3*exp(-(v + 48.0) / 18.0)); }
__host__ __device__ static double beta_ci_CaHVA(const double v) { return (1.8e-3*exp((v + 48.0) / 83.0)); }
__host__ __device__ static double inf_ci_CaHVA(const double v) { return (alpha_ci_CaHVA(v) / (alpha_ci_CaHVA(v) + beta_ci_CaHVA(v))); }
__host__ __device__ static double tau_ci_CaHVA(const double v) { return (1.0 / (alpha_ci_CaHVA(v) + beta_ci_CaHVA(v))); }

__host__ __device__ static double inf_cl_CaLVA(const double v) { return (1.0 / (1.0 + exp(-(v + 52.0) / 7.4))); }
__host__ __device__ static double tau_cl_CaLVA(const double v) { return ((3.0 + 1.0 / (exp((v + 27.0) / 10.0) + exp(-(v + 102.0) / 15.0))) / 0.85); }
__host__ __device__ static double inf_cm_CaLVA(const double v) { return (1.0 / (1.0 + exp((v + 80.0) / 5.0))); }
__host__ __device__ static double tau_cm_CaLVA(const double v) { return ((85.0 + 1.0 / (exp((v + 48.0) / 4.0) + exp(-(v + 407.0) / 50.0))) / 0.9); }

//K channel
__host__ __device__ static double alpha_n_KV(const double v) { return 0.062*(v + 26.0) / (1.0 - exp(-(v + 26.0) / 10.0)); }
__host__ __device__ static double beta_n_KV(const double v) { return 0.78*exp(-(v + 36.0) / 80.0); }
__host__ __device__ static double inf_n_KV(const double v) { return (alpha_n_KV(v) / (alpha_n_KV(v) + beta_n_KV(v))); }
__host__ __device__ static double tau_n_KV(const double v) { return (1.0 / (alpha_n_KV(v) + beta_n_KV(v))); }

__host__ __device__ static double alpha_a_KA(const double v) { return (0.62 / (1.0 + exp(-(v + 9.17) / 23.32))); }
__host__ __device__ static double beta_a_KA(const double v) { return (0.126 / (exp((v + 18.28) / 19.47))); }
__host__ __device__ static double inf_a_KA(const double v) { return (1.0 / (1.0 + exp(-(v + 38.0) / 17.0))); }
__host__ __device__ static double tau_a_KA(const double v) { return (1.0 / (alpha_a_KA(v) + beta_a_KA(v))); }

__host__ __device__ static double alpha_b_KA(const double v) { return (0.028 / (1.0 + exp((v + 111.0) / 12.84))); }
__host__ __device__ static double beta_b_KA(const double v) { return(0.026 / (1.0 + exp(-(v + 49.95) / 8.9))); }
__host__ __device__ static double inf_b_KA(const double v) { return (1.0 / (1.0 + exp((v + 78.8) / 8.4))); }
__host__ __device__ static double tau_b_KA(const double v) { return (1.0 / (alpha_b_KA(v) + beta_b_KA(v))); }

__host__ __device__ static double alpha_c_KC(const double v, const double ca) { return (3.2 / (1.0 + 0.0015*exp(-(v) / 11.7) / ca)); }
__host__ __device__ static double beta_c_KC(const double v, const double ca) { return (0.46 / (1.0 + ca / (1.5e-4*exp(-(v) / 11.7)))); }
__host__ __device__ static double inf_c_KC(const double v, const double ca) { return (alpha_c_KC(v, ca) / (alpha_c_KC(v, ca) + beta_c_KC(v, ca))); }
__host__ __device__ static double tau_c_KC(const double v, const double ca) { return (1.0 / (alpha_c_KC(v, ca) + beta_c_KC(v, ca))); }

__host__ __device__ static double alpha_sl_Kslow(const double v) { return (0.0037*exp((v + 30.0) / 40.0)); }
__host__ __device__ static double beta_sl_Kslow(const double v) { return (0.0037*exp(-(v + 30.0) / 20.0)); }
__host__ __device__ static double inf_sl_Kslow(const double v) { return (1.0 / (1.0 + exp(-(v + 35.0) / 6.0))); }
__host__ __device__ static double tau_sl_Kslow(const double v) { return (1.0 / (alpha_sl_Kslow(v) + beta_sl_Kslow(v))); }

__host__ __device__ static double r_HCN1(const double v) { return (0.0021 * (v) + 0.97); }
__host__ __device__ static double inf_hf_HCN1(const double v) { return (r_HCN1(v) * (1.0 / (1.0 + exp((v + 72.5)*0.11)))); }
__host__ __device__ static double inf_hs_HCN1(const double v) { return ((1.0 - r_HCN1(v)) * (1.0 / (1.0 + exp((v + 72.5)*0.11)))); }
__host__ __device__ static double tau_hf_HCN1(const double v) { return (exp((0.0137*v + 3.37)*2.3)); }
__host__ __device__ static double tau_hs_HCN1(const double v) { return (exp((0.0145*v + 4.06)*2.3)); }

__host__ __device__ static double r_HCN2(const double v) {
	//return (-0.0227 * (v + 10.0) - 1.47);
	if (v >= -64.70)return 0.0;
	else if (v <= -108.70) return 1.0;
	else return (-0.0227 * (v) - 1.47);
}
__host__ __device__ static double inf_hf_HCN2(const double v) { return (r_HCN2(v) * (1.0 / (1.0 + exp((v + 81.9)*0.16)))); }
__host__ __device__ static double inf_hs_HCN2(const double v) { return ((1.0 - r_HCN2(v)) * (1.0 / (1.0 + exp((v + 81.9)*0.16)))); }
__host__ __device__ static double tau_hf_HCN2(const double v) { return (exp((0.027*v + 5.6)*2.3)); }
__host__ __device__ static double tau_hs_HCN2(const double v) { return (exp((0.015*v + 5.3)*2.3)); }

__global__
void go_KAHP_update_2order ( const int n, const double *ca, double *ca_old, 
  double *o1, double *o2, double *c1, double *c2, double *c3, double *c4, const double DT )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  
  double i1, i2, i3, a1, a2, b1, b2;
  i1 = 80e-3;
  i2 = 80e-3;
  i3 = 200e-3;
  a1 = 1.0;
  a2 = 100e-3;
  b1 = 160e-3;
  b2 = 1.2;
  //double k1, k2, k3, k4;
  //for (int i = 0; i < n * GO_COMP; i++) {
  if ( id < n * GO_COMP )
  {			
    double l_ca = ca_old [ id ];
    ca_old [ id ] = ca [ id ];
    double l_dt = 2.0 / DT;

    double d1 = 200 * l_ca / 3.0;
    double d2 = 160 * l_ca / 3.0;
    double d3 = 80  * l_ca / 3.0;  

    double vca_n [ 6 ] [ 6 ] = {
      { -d1 + l_dt, i1,              0.0,                    0.0,                0.0,          0.0 },
      {  d1,        -d2 - i1 + l_dt,  i2,                    0.0,                0.0,          0.0 },
      { 0.0,         d2,              -(d3 + i2 + b1) + l_dt, i3,                 a1,          0.0 },
      { 0.0,        0.0,               d3,                    -(i3 + b2) + l_dt, 0.0,          a2 },
      { 0.0,        0.0,               b1,                    0.0,                -a1 + l_dt, 0.0 },
      { 0.0,        0.0,              0.0,                     b2,                0.0,         -a2 + l_dt }
    };
    double vc [ 6 ] = { c1 [ id ], c2 [ id ], c3 [ id ], c4 [ id ], o1 [ id ], o2 [ id ] };
    double vcb [ 6 ] = { };

    for ( int i = 0; i < 6; i++ )
    {
      for ( int j = 0; j < 6; j++ )
      {
        vcb [ i ] -= vc [ j ] * vca_n [ i ] [ j ];
      }
    }

    l_ca = ca [ id ];
    d1 = 200 * l_ca / 3.0;
    d2 = 160 * l_ca / 3.0;
    d3 = 80  * l_ca / 3.0;

    double vca [ 6 ] [ 6 ] = {
      { -d1 - l_dt, i1,              0.0,                    0.0,                0.0,          0.0 },
      {  d1,        -d2 - i1 - l_dt,  i2,                    0.0,                0.0,          0.0 },
      { 0.0,         d2,              -(d3 + i2 + b1) - l_dt, i3,                 a1,          0.0 },
      { 0.0,        0.0,               d3,                    -(i3 + b2) - l_dt, 0.0,          a2 },
      { 0.0,        0.0,               b1,                    0.0,                -a1 - l_dt, 0.0 },
      { 0.0,        0.0,              0.0,                     b2,                0.0,         -a2 - l_dt }
    };
      
    //////////////////// Pivot selection //////////////////
    double temp;
    for ( int i = 0; i < 5; i++ ) 
    {
      int pivot = i;
      double p_max = fabs ( vca [ i ] [ i ] );
      for ( int j = i + 1; j < 6; j++ )
      {
        if ( fabs ( vca [ j ] [ i ] ) > p_max )
        {
          pivot = j;
          p_max = fabs ( vca [ j ] [ i ] );
        }
      }
      // minimum pivot error
      if ( fabs ( p_max ) < 1.0e-12 ) { printf ( "pivot error\n" ); return; }      
  
      if ( pivot != i ) // pivot exchange
      {
        for ( int j = i; j < 6; j++ )
        {
          temp = vca [ i ] [ j ];
          vca [ i ] [ j ] =  vca [ pivot ] [ j ];
          vca [ pivot ] [ j ] = temp;
        }
        temp = vcb [ i ];
        vcb [ i ] = vcb [ pivot ];
        vcb [ pivot ] = temp;
      }
      //////////////////// Forward elimination //////////////////
      for ( int j = i + 1; j < 6; j++ ) 
      {
        double w = vca [ j ] [ i ] / vca [ i ] [ i ];
        vca [ j ] [ i ] = 0.0;
        // Multiply the ith line by -a[j][i]/a[i][i] and add it to the jth line
        for ( int k = i + 1; k < 6; k++ ) 
        {
           vca [ j ] [ k ] = vca [ j ] [ k ] - vca [ i ] [ k ] * w;
        }
        vcb [ j ] = vcb [ j ] - vcb [ i ] * w;
      }
    }
    //////////////////// Backward elimination //////////////////   
    for ( int i = 6 - 1; i >= 0; i-- )
    {
      for( int j = i + 1; j < 6; j++)
      {
        vcb [ i ] = vcb [ i ] - vca [ i ] [ j ] * vcb [ j ];
        vca [ i ] [ j ] = 0.0;
      }
      vcb [ i ] = vcb [ i ] / vca [ i ] [ i ];
      vca [ i ] [ i ] = 1.0;
    }       
      
    c1[id] = vcb[0];
    c2[id] = vcb[1];
    c3[id] = vcb[2];
    c4[id] = vcb[3];
    o1[id] = vcb[4];
    o2[id] = vcb[5];
  
    if ((o1[id] + o2[id] + c1[id] + c2[id] + c3[id] + c4[id] < 0.9999)
     || (o1[id] + o2[id] + c1[id] + c2[id] + c3[id] + c4[id] > 1.0001)) {
      printf("KAHP error %.15f\n", o1[id] + o2[id] + c1[id] + c2[id] + c3[id] + c4[id]); //break;
    }
  }
}
__global__
void go_KAHP_update ( const int n, const double *ca, double *o1, double *o2, double *c1, double *c2, double *c3, double *c4, const double DT )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  
  double i1, i2, i3, a1, a2, b1, b2;
  i1 = 80e-3;
  i2 = 80e-3;
  i3 = 200e-3;
  a1 = 1.0;
  a2 = 100e-3;
  b1 = 160e-3;
  b2 = 1.2;
  //double k1, k2, k3, k4;
  //for (int i = 0; i < n * GO_COMP; i++) {
  if ( id < n * GO_COMP )
  {			
    double d1 = 200 * ca [ id ] / 3.0;
    double d2 = 160 * ca [ id ] / 3.0;
    double d3 = 80  * ca [ id ] / 3.0;
  
    double vca [ 6 ] [ 6 ] = {
      { -d1,       i1,             0.0,        0.0, 0.0, 0.0 },
      {  d1, -d2 - i1,              i2,        0.0, 0.0, 0.0 },
      { 0.0,       d2, -(d3 + i2 + b1),         i3,  a1, 0.0 },
      { 0.0,      0.0,              d3, -(i3 + b2), 0.0,  a2 },
      { 0.0,      0.0,              b1,        0.0, -a1, 0.0 },
      { 0.0,      0.0,              0.0,        b2, 0.0, -a2 }
    };
    double vcb [ 6 ] = { c1 [ id ], c2 [ id ], c3 [ id ], c4 [ id ], o1 [ id ], o2 [ id ] };
   
      
    for ( int i = 0; i < 6; i++ )
    { 
      for ( int j = 0; j < 6; j++ ) { vca [ i ] [ j ] *= - DT; }
    }
    for ( int i = 0; i < 6; i++ ) { vca [ i ] [ i ] += 1.0; }
    //////////////////// Pivot selection //////////////////
    double temp;
    for ( int i = 0; i < 5; i++ ) 
    {
      int pivot = i;
      double p_max = fabs ( vca [ i ] [ i ] );
      for ( int j = i + 1; j < 6; j++ )
      {
        if ( fabs ( vca [ j ] [ i ] ) > p_max )
        {
          pivot = j;
          p_max = fabs ( vca [ j ] [ i ] );
        }
      }
      // minimum pivot error
      if ( fabs ( p_max ) < 1.0e-12 ) { printf ( "pivot error\n" ); return; }      
  
      if ( pivot != i ) // pivot exchange
      {
        for ( int j = i; j < 6; j++ )
        {
          temp = vca [ i ] [ j ];
          vca [ i ] [ j ] =  vca [ pivot ] [ j ];
          vca [ pivot ] [ j ] = temp;
        }
        temp = vcb [ i ];
        vcb [ i ] = vcb [ pivot ];
        vcb [ pivot ] = temp;
      }
      //////////////////// Forward elimination //////////////////
      for ( int j = i + 1; j < 6; j++ ) 
      {
        double w = vca [ j ] [ i ] / vca [ i ] [ i ];
        vca [ j ] [ i ] = 0.0;
        // Multiply the ith line by -a[j][i]/a[i][i] and add it to the jth line
        for ( int k = i + 1; k < 6; k++ ) 
        {
           vca [ j ] [ k ] = vca [ j ] [ k ] - vca [ i ] [ k ] * w;
        }
        vcb [ j ] = vcb [ j ] - vcb [ i ] * w;
      }
    }
    //////////////////// Backward elimination //////////////////   
    for ( int i = 6 - 1; i >= 0; i-- )
    {
      for( int j = i + 1; j < 6; j++)
      {
        vcb [ i ] = vcb [ i ] - vca [ i ] [ j ] * vcb [ j ];
        vca [ i ] [ j ] = 0.0;
      }
      vcb [ i ] = vcb [ i ] / vca [ i ] [ i ];
      vca [ i ] [ i ] = 1.0;
    }       
      
    c1[id] = vcb[0];
    c2[id] = vcb[1];
    c3[id] = vcb[2];
    c4[id] = vcb[3];
    o1[id] = vcb[4];
    o2[id] = vcb[5];
  
    if ((o1[id] + o2[id] + c1[id] + c2[id] + c3[id] + c4[id] < 0.9999)
     || (o1[id] + o2[id] + c1[id] + c2[id] + c3[id] + c4[id] > 1.0001)) {
      printf("KAHP error %.15f\n", o1[id] + o2[id] + c1[id] + c2[id] + c3[id] + c4[id]); //break;
    }
  }
}

__global__ void go_update_ion_exp_imp ( neuron_t *d_go, neuron_solve_t *d_go_solve, const double DT )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  double **elem = d_go -> elem;
  double **ion  = d_go -> ion;
  double **cond = d_go -> cond;
  
  if ( id < d_go -> nc)
  {      
    double v_val  = ( elem [ v ] [ id ] + d_go_solve -> vec [ cn_v_old ] [ id ] ) / 2.0;
    double Ca_val = elem [ Ca ] [ id ];
    double Ca2_val = d_go -> ca2 [ id ];

    double I_Ca1 = 1e-3 * cond [ g_CaHVA_go ] [ id ] / elem [ area ] [ id ] * ion [ ch_CaHVA_go ] [ id ] *
      ion [ ch_CaHVA_go ] [ id ] * ion [ ci_CaHVA_go ] [ id ] * ( d_go_solve -> vec [ cn_v_old ] [ id ] - V_Ca_GO ); // I_Ca [mA/cm^2]
    double I_Ca2 = 1e-3 * cond [ g_CaLVA_go ] [ id ] / elem [ area ] [ id ] * ion [ cl_CaLVA_go ] [ id ] *
      ion [ cl_CaLVA_go ] [ id ] * ion [ cm_CaLVA_go ] [ id ] * ( d_go_solve -> vec [ cn_v_old ] [ id ] - ( d_go -> rev_ca2 [ id ] ) ); // I_Ca [mA/cm^2]   

    double k1 = DT * ( - I_Ca1 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca_val            - Ca1_0_GO ) ); //[mA*mol/cm^3*sec*A]=[M/sec] **[1mM = 1mol/m^3]**
    double k2 = DT * ( - I_Ca1 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca_val + k1 / 2.0 - Ca1_0_GO ) );
    double k3 = DT * ( - I_Ca1 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca_val + k2 / 2.0 - Ca1_0_GO ) );
    double k4 = DT * ( - I_Ca1 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca_val + k3       - Ca1_0_GO ) );
    elem [ Ca ] [ id ] += ( k1 + k2 * 2.0 + k3 * 2.0 + k4 ) / 6.0;
  
    k1 = DT * ( - I_Ca2 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca2_val            - Ca1_0_GO ) ); //[mA*mol/cm^3*sec*A]=[M/sec] **[1mM = 1mol/m^3]**
    k2 = DT * ( - I_Ca2 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca2_val + k1 / 2.0 - Ca1_0_GO ) );
    k3 = DT * ( - I_Ca2 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca2_val + k2 / 2.0 - Ca1_0_GO ) );
    k4 = DT * ( - I_Ca2 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca2_val + k3       - Ca1_0_GO ) );
    d_go -> ca2 [ id ] += ( k1 + k2 * 2.0 + k3 * 2.0 + k4 ) / 6.0;
    // Ca2 Vrev update
    d_go -> rev_ca2 [ id ] =  ( 1e3 ) * ( 8.313424 * ( 23.0 + 273.15 ) ) / (2 * F_GO ) * log ( Ca1OUT_GO / d_go -> ca2 [ id ] );//[mV]
  
    Ca_val = ( Ca_val + elem [ Ca ] [ id ] ) / 2.0;

    ion [ m_NaT_go   ] [ id ] = ( 2.0 * DT * inf_m_NaT   ( v_val ) + ( 2.0 * tau_m_NaT   ( v_val ) - DT ) * ion [ m_NaT_go   ] [ id ] ) / ( 2.0 * tau_m_NaT   ( v_val ) + DT );
    ion [ h_NaT_go   ] [ id ] = ( 2.0 * DT * inf_h_NaT   ( v_val ) + ( 2.0 * tau_h_NaT   ( v_val ) - DT ) * ion [ h_NaT_go   ] [ id ] ) / ( 2.0 * tau_h_NaT   ( v_val ) + DT );
    ion [ r_NaR_go   ] [ id ] = ( 2.0 * DT * inf_r_NaR   ( v_val ) + ( 2.0 * tau_r_NaR   ( v_val ) - DT ) * ion [ r_NaR_go   ] [ id ] ) / ( 2.0 * tau_r_NaR   ( v_val ) + DT );
    ion [ s_NaR_go   ] [ id ] = ( 2.0 * DT * inf_s_NaR   ( v_val ) + ( 2.0 * tau_s_NaR   ( v_val ) - DT ) * ion [ s_NaR_go   ] [ id ] ) / ( 2.0 * tau_s_NaR   ( v_val ) + DT );
    ion [ p_NaP_go   ] [ id ] = ( 2.0 * DT * inf_p_NaP   ( v_val ) + ( 2.0 * tau_p_NaP   ( v_val ) - DT ) * ion [ p_NaP_go   ] [ id ] ) / ( 2.0 * tau_p_NaP   ( v_val ) + DT );

    ion [ n_KV_go   ] [ id ] = ( 2.0 * DT * inf_n_KV   ( v_val ) + ( 2.0 * tau_n_KV   ( v_val ) - DT ) * ion [ n_KV_go   ] [ id ] ) / ( 2.0 * tau_n_KV   ( v_val ) + DT );
    ion [ a_KA_go   ] [ id ] = ( 2.0 * DT * inf_a_KA   ( v_val ) + ( 2.0 * tau_a_KA   ( v_val ) - DT ) * ion [ a_KA_go   ] [ id ] ) / ( 2.0 * tau_a_KA   ( v_val ) + DT );
    ion [ b_KA_go   ] [ id ] = ( 2.0 * DT * inf_b_KA   ( v_val ) + ( 2.0 * tau_b_KA   ( v_val ) - DT ) * ion [ b_KA_go   ] [ id ] ) / ( 2.0 * tau_b_KA   ( v_val ) + DT );
    ion [ sl_Kslow_go   ] [ id ] = ( 2.0 * DT * inf_sl_Kslow   ( v_val ) + ( 2.0 * tau_sl_Kslow   ( v_val ) - DT ) * ion [ sl_Kslow_go   ] [ id ] ) / ( 2.0 * tau_sl_Kslow   ( v_val ) + DT );
    ion [ c_KC_go  ] [ id ] = ( 2.0 * DT * inf_c_KC  ( v_val, Ca_val ) + ( 2.0 * tau_c_KC ( v_val, Ca_val ) - DT ) * ion [ c_KC_go   ] [ id ] ) / ( 2.0 * tau_c_KC ( v_val, Ca_val ) + DT ); 
   
    ion [ ch_CaHVA_go   ] [ id ] = ( 2.0 * DT * inf_ch_CaHVA   ( v_val ) + ( 2.0 * tau_ch_CaHVA   ( v_val ) - DT ) * ion [ ch_CaHVA_go   ] [ id ] ) / ( 2.0 * tau_ch_CaHVA   ( v_val ) + DT );
    ion [ ci_CaHVA_go   ] [ id ] = ( 2.0 * DT * inf_ci_CaHVA   ( v_val ) + ( 2.0 * tau_ci_CaHVA   ( v_val ) - DT ) * ion [ ci_CaHVA_go   ] [ id ] ) / ( 2.0 * tau_ci_CaHVA   ( v_val ) + DT );
    ion [ cl_CaLVA_go   ] [ id ] = ( 2.0 * DT * inf_cl_CaLVA   ( v_val ) + ( 2.0 * tau_cl_CaLVA   ( v_val ) - DT ) * ion [ cl_CaLVA_go   ] [ id ] ) / ( 2.0 * tau_cl_CaLVA   ( v_val ) + DT );
    ion [ cm_CaLVA_go   ] [ id ] = ( 2.0 * DT * inf_cm_CaLVA   ( v_val ) + ( 2.0 * tau_cm_CaLVA   ( v_val ) - DT ) * ion [ cm_CaLVA_go   ] [ id ] ) / ( 2.0 * tau_cm_CaLVA   ( v_val ) + DT );
    
    ion [ hf_HCN1_go   ] [ id ] = ( 2.0 * DT * inf_hf_HCN1   ( v_val ) + ( 2.0 * tau_hf_HCN1   ( v_val ) - DT ) * ion [ hf_HCN1_go   ] [ id ] ) / ( 2.0 * tau_hf_HCN1   ( v_val ) + DT );
    ion [ hf_HCN2_go   ] [ id ] = ( 2.0 * DT * inf_hf_HCN2   ( v_val ) + ( 2.0 * tau_hf_HCN2   ( v_val ) - DT ) * ion [ hf_HCN2_go   ] [ id ] ) / ( 2.0 * tau_hf_HCN2   ( v_val ) + DT );
    ion [ hs_HCN1_go   ] [ id ] = ( 2.0 * DT * inf_hs_HCN1   ( v_val ) + ( 2.0 * tau_hs_HCN1   ( v_val ) - DT ) * ion [ hs_HCN1_go   ] [ id ] ) / ( 2.0 * tau_hs_HCN1   ( v_val ) + DT );
    ion [ hs_HCN2_go   ] [ id ] = ( 2.0 * DT * inf_hs_HCN2   ( v_val ) + ( 2.0 * tau_hs_HCN2   ( v_val ) - DT ) * ion [ hs_HCN2_go   ] [ id ] ) / ( 2.0 * tau_hs_HCN2   ( v_val ) + DT );

    d_go_solve -> vec [ cn_v_old ] [ id ]  = elem [ v ] [ id ] ;
  }
}

__global__ void go_update_ion_RKC_exp_imp ( neuron_t *d_go, neuron_solve_t *d_go_solve, double *vnew, const double DT )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  double **elem = d_go -> elem;
  double **ion  = d_go -> ion;
  double **cond = d_go -> cond;
  
  if ( id < d_go -> nc)
  {      
    double v_val  = ( elem [ v ] [ id ] + vnew [ id ] ) / 2.0;
    double Ca_val = elem [ Ca ] [ id ];
    double Ca2_val = d_go -> ca2 [ id ];

    double I_Ca1 = 1e-3 * cond [ g_CaHVA_go ] [ id ] / elem [ area ] [ id ] * ion [ ch_CaHVA_go ] [ id ] *
      ion [ ch_CaHVA_go ] [ id ] * ion [ ci_CaHVA_go ] [ id ] * ( elem [ v ] [ id ] - V_Ca_GO ); // I_Ca [mA/cm^2]
    double I_Ca2 = 1e-3 * cond [ g_CaLVA_go ] [ id ] / elem [ area ] [ id ] * ion [ cl_CaLVA_go ] [ id ] *
      ion [ cl_CaLVA_go ] [ id ] * ion [ cm_CaLVA_go ] [ id ] * ( elem [ v ] [ id ]  - ( d_go -> rev_ca2 [ id ] ) ); // I_Ca [mA/cm^2]   

    double k1 = DT * ( - I_Ca1 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca_val            - Ca1_0_GO ) ); //[mA*mol/cm^3*sec*A]=[M/sec] **[1mM = 1mol/m^3]**
    double k2 = DT * ( - I_Ca1 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca_val + k1 / 2.0 - Ca1_0_GO ) );
    double k3 = DT * ( - I_Ca1 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca_val + k2 / 2.0 - Ca1_0_GO ) );
    double k4 = DT * ( - I_Ca1 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca_val + k3       - Ca1_0_GO ) );
    elem [ Ca ] [ id ] += ( k1 + k2 * 2.0 + k3 * 2.0 + k4 ) / 6.0;
  
    k1 = DT * ( - I_Ca2 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca2_val            - Ca1_0_GO ) ); //[mA*mol/cm^3*sec*A]=[M/sec] **[1mM = 1mol/m^3]**
    k2 = DT * ( - I_Ca2 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca2_val + k1 / 2.0 - Ca1_0_GO ) );
    k3 = DT * ( - I_Ca2 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca2_val + k2 / 2.0 - Ca1_0_GO ) );
    k4 = DT * ( - I_Ca2 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca2_val + k3       - Ca1_0_GO ) );
    d_go -> ca2 [ id ] += ( k1 + k2 * 2.0 + k3 * 2.0 + k4 ) / 6.0;
    // Ca2 Vrev update
    d_go -> rev_ca2 [ id ] =  ( 1e3 ) * ( 8.313424 * ( 23.0 + 273.15 ) ) / (2 * F_GO ) * log ( Ca1OUT_GO / d_go -> ca2 [ id ] );//[mV]
  
    Ca_val = ( Ca_val + elem [ Ca ] [ id ] ) / 2.0;

    ion [ m_NaT_go   ] [ id ] = ( 2.0 * DT * inf_m_NaT   ( v_val ) + ( 2.0 * tau_m_NaT   ( v_val ) - DT ) * ion [ m_NaT_go   ] [ id ] ) / ( 2.0 * tau_m_NaT   ( v_val ) + DT );
    ion [ h_NaT_go   ] [ id ] = ( 2.0 * DT * inf_h_NaT   ( v_val ) + ( 2.0 * tau_h_NaT   ( v_val ) - DT ) * ion [ h_NaT_go   ] [ id ] ) / ( 2.0 * tau_h_NaT   ( v_val ) + DT );
    ion [ r_NaR_go   ] [ id ] = ( 2.0 * DT * inf_r_NaR   ( v_val ) + ( 2.0 * tau_r_NaR   ( v_val ) - DT ) * ion [ r_NaR_go   ] [ id ] ) / ( 2.0 * tau_r_NaR   ( v_val ) + DT );
    ion [ s_NaR_go   ] [ id ] = ( 2.0 * DT * inf_s_NaR   ( v_val ) + ( 2.0 * tau_s_NaR   ( v_val ) - DT ) * ion [ s_NaR_go   ] [ id ] ) / ( 2.0 * tau_s_NaR   ( v_val ) + DT );
    ion [ p_NaP_go   ] [ id ] = ( 2.0 * DT * inf_p_NaP   ( v_val ) + ( 2.0 * tau_p_NaP   ( v_val ) - DT ) * ion [ p_NaP_go   ] [ id ] ) / ( 2.0 * tau_p_NaP   ( v_val ) + DT );

    ion [ n_KV_go   ] [ id ] = ( 2.0 * DT * inf_n_KV   ( v_val ) + ( 2.0 * tau_n_KV   ( v_val ) - DT ) * ion [ n_KV_go   ] [ id ] ) / ( 2.0 * tau_n_KV   ( v_val ) + DT );
    ion [ a_KA_go   ] [ id ] = ( 2.0 * DT * inf_a_KA   ( v_val ) + ( 2.0 * tau_a_KA   ( v_val ) - DT ) * ion [ a_KA_go   ] [ id ] ) / ( 2.0 * tau_a_KA   ( v_val ) + DT );
    ion [ b_KA_go   ] [ id ] = ( 2.0 * DT * inf_b_KA   ( v_val ) + ( 2.0 * tau_b_KA   ( v_val ) - DT ) * ion [ b_KA_go   ] [ id ] ) / ( 2.0 * tau_b_KA   ( v_val ) + DT );
    ion [ sl_Kslow_go   ] [ id ] = ( 2.0 * DT * inf_sl_Kslow   ( v_val ) + ( 2.0 * tau_sl_Kslow   ( v_val ) - DT ) * ion [ sl_Kslow_go   ] [ id ] ) / ( 2.0 * tau_sl_Kslow   ( v_val ) + DT );
    ion [ c_KC_go  ] [ id ] = ( 2.0 * DT * inf_c_KC  ( v_val, Ca_val ) + ( 2.0 * tau_c_KC ( v_val, Ca_val ) - DT ) * ion [ c_KC_go   ] [ id ] ) / ( 2.0 * tau_c_KC ( v_val, Ca_val ) + DT ); 
   
    ion [ ch_CaHVA_go   ] [ id ] = ( 2.0 * DT * inf_ch_CaHVA   ( v_val ) + ( 2.0 * tau_ch_CaHVA   ( v_val ) - DT ) * ion [ ch_CaHVA_go   ] [ id ] ) / ( 2.0 * tau_ch_CaHVA   ( v_val ) + DT );
    ion [ ci_CaHVA_go   ] [ id ] = ( 2.0 * DT * inf_ci_CaHVA   ( v_val ) + ( 2.0 * tau_ci_CaHVA   ( v_val ) - DT ) * ion [ ci_CaHVA_go   ] [ id ] ) / ( 2.0 * tau_ci_CaHVA   ( v_val ) + DT );
    ion [ cl_CaLVA_go   ] [ id ] = ( 2.0 * DT * inf_cl_CaLVA   ( v_val ) + ( 2.0 * tau_cl_CaLVA   ( v_val ) - DT ) * ion [ cl_CaLVA_go   ] [ id ] ) / ( 2.0 * tau_cl_CaLVA   ( v_val ) + DT );
    ion [ cm_CaLVA_go   ] [ id ] = ( 2.0 * DT * inf_cm_CaLVA   ( v_val ) + ( 2.0 * tau_cm_CaLVA   ( v_val ) - DT ) * ion [ cm_CaLVA_go   ] [ id ] ) / ( 2.0 * tau_cm_CaLVA   ( v_val ) + DT );
    
    ion [ hf_HCN1_go   ] [ id ] = ( 2.0 * DT * inf_hf_HCN1   ( v_val ) + ( 2.0 * tau_hf_HCN1   ( v_val ) - DT ) * ion [ hf_HCN1_go   ] [ id ] ) / ( 2.0 * tau_hf_HCN1   ( v_val ) + DT );
    ion [ hf_HCN2_go   ] [ id ] = ( 2.0 * DT * inf_hf_HCN2   ( v_val ) + ( 2.0 * tau_hf_HCN2   ( v_val ) - DT ) * ion [ hf_HCN2_go   ] [ id ] ) / ( 2.0 * tau_hf_HCN2   ( v_val ) + DT );
    ion [ hs_HCN1_go   ] [ id ] = ( 2.0 * DT * inf_hs_HCN1   ( v_val ) + ( 2.0 * tau_hs_HCN1   ( v_val ) - DT ) * ion [ hs_HCN1_go   ] [ id ] ) / ( 2.0 * tau_hs_HCN1   ( v_val ) + DT );
    ion [ hs_HCN2_go   ] [ id ] = ( 2.0 * DT * inf_hs_HCN2   ( v_val ) + ( 2.0 * tau_hs_HCN2   ( v_val ) - DT ) * ion [ hs_HCN2_go   ] [ id ] ) / ( 2.0 * tau_hs_HCN2   ( v_val ) + DT );

    elem [ v ] [ id ] = vnew [ id ];
  }
}
__global__ void go_update_ion ( neuron_t *d_go, neuron_solve_t *d_go_solve, const double DT )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  double **elem = d_go -> elem;
  double **ion  = d_go -> ion;
  double **cond = d_go -> cond;
  
  if ( id < d_go -> nc)
  {      
    double v_val = elem [ v ] [ id ];
    double Ca_val = elem [ Ca ] [ id ];
    double Ca2_val = d_go -> ca2 [ id ];

    double I_Ca1 = 1e-3 * cond [ g_CaHVA_go ] [ id ] / elem [ area ] [ id ] * ion [ ch_CaHVA_go ] [ id ] *
      ion [ ch_CaHVA_go ] [ id ] * ion [ ci_CaHVA_go ] [ id ] * ( elem [ v ] [ id ] - V_Ca_GO ); // I_Ca [mA/cm^2]
    double I_Ca2 = 1e-3 * cond [ g_CaLVA_go ] [ id ] / elem [ area ] [ id ] * ion [ cl_CaLVA_go ] [ id ] *
      ion [ cl_CaLVA_go ] [ id ] * ion [ cm_CaLVA_go ] [ id ] * ( elem [ v ] [ id ] - ( d_go -> rev_ca2 [ id ] ) ); // I_Ca [mA/cm^2]   

    ion [ m_NaT_go ] [ id ] = inf_m_NaT ( v_val ) + ( ion [ m_NaT_go ] [ id ] - inf_m_NaT ( v_val ) ) * exp ( -DT / tau_m_NaT ( v_val ) );
    ion [ h_NaT_go ] [ id ] = inf_h_NaT ( v_val ) + ( ion [ h_NaT_go ] [ id ] - inf_h_NaT ( v_val ) ) * exp ( -DT / tau_h_NaT ( v_val ) );
    ion [ r_NaR_go ] [ id ] = inf_r_NaR ( v_val ) + ( ion [ r_NaR_go ] [ id ] - inf_r_NaR ( v_val ) ) * exp ( -DT / tau_r_NaR ( v_val ) );
    ion [ s_NaR_go ] [ id ] = inf_s_NaR ( v_val ) + ( ion [ s_NaR_go ] [ id ] - inf_s_NaR ( v_val ) ) * exp ( -DT / tau_s_NaR ( v_val ) );
    ion [ p_NaP_go ] [ id ] = inf_p_NaP ( v_val ) + ( ion [ p_NaP_go ] [ id ] - inf_p_NaP ( v_val ) ) * exp ( -DT / tau_p_NaP ( v_val ) );

    ion [ n_KV_go ] [ id ] = inf_n_KV ( v_val ) + ( ion [ n_KV_go ] [ id ] - inf_n_KV ( v_val ) ) * exp ( -DT / tau_n_KV ( v_val ) );
    ion [ a_KA_go ] [ id ] = inf_a_KA ( v_val ) + ( ion [ a_KA_go ] [ id ] - inf_a_KA ( v_val ) ) * exp ( -DT / tau_a_KA ( v_val ) );
    ion [ b_KA_go ] [ id ] = inf_b_KA ( v_val ) + ( ion [ b_KA_go ] [ id ] - inf_b_KA ( v_val ) ) * exp ( -DT / tau_b_KA ( v_val ) );
    ion [ c_KC_go ] [ id ] = inf_c_KC ( v_val , Ca_val )
                           + (ion [ c_KC_go ] [ id ] - inf_c_KC ( v_val , Ca_val ) ) * exp ( -DT / tau_c_KC ( v_val , Ca_val ) );
    ion [ sl_Kslow_go ] [ id ] = inf_sl_Kslow ( v_val ) + ( ion [ sl_Kslow_go ] [ id ] - inf_sl_Kslow ( v_val ) ) * exp ( -DT / tau_sl_Kslow ( v_val ) );
		
    ion [ ch_CaHVA_go ] [ id ] = inf_ch_CaHVA( v_val ) + ( ion [ ch_CaHVA_go ] [ id ] - inf_ch_CaHVA ( v_val ) ) * exp ( -DT / tau_ch_CaHVA( v_val ) );
    ion [ ci_CaHVA_go ] [ id ] = inf_ci_CaHVA( v_val ) + ( ion [ ci_CaHVA_go ] [ id ] - inf_ci_CaHVA ( v_val ) ) * exp ( -DT / tau_ci_CaHVA( v_val ) );
    ion [ cl_CaLVA_go ] [ id ] = inf_cl_CaLVA( v_val ) + ( ion [ cl_CaLVA_go ] [ id ] - inf_cl_CaLVA ( v_val ) ) * exp ( -DT / tau_cl_CaLVA( v_val ) );
    ion [ cm_CaLVA_go ] [ id ] = inf_cm_CaLVA( v_val ) + ( ion [ cm_CaLVA_go ] [ id ] - inf_cm_CaLVA ( v_val ) ) * exp ( -DT / tau_cm_CaLVA( v_val ) );

    ion [ hf_HCN1_go ] [ id ] = inf_hf_HCN1( v_val ) + ( ion [ hf_HCN1_go ] [ id ] - inf_hf_HCN1 ( v_val ) ) * exp ( -DT / tau_hf_HCN1( v_val ) );
    ion [ hf_HCN2_go ] [ id ] = inf_hf_HCN2( v_val ) + ( ion [ hf_HCN2_go ] [ id ] - inf_hf_HCN2 ( v_val ) ) * exp ( -DT / tau_hf_HCN2( v_val ) );
    ion [ hs_HCN1_go ] [ id ] = inf_hs_HCN1( v_val ) + ( ion [ hs_HCN1_go ] [ id ] - inf_hs_HCN1 ( v_val ) ) * exp ( -DT / tau_hs_HCN1( v_val ) );
    ion [ hs_HCN2_go ] [ id ] = inf_hs_HCN2( v_val ) + ( ion [ hs_HCN2_go ] [ id ] - inf_hs_HCN2 ( v_val ) ) * exp ( -DT / tau_hs_HCN2( v_val ) );

    // integral
    //elem [ Ca ] [ id ] = ( DT * ( - I_Ca1 / (  2.0 * F_GO * SHELL1_D_GO ) + B_Ca1_GO *  Ca1_0_GO) + Ca_val ) / (1.0 + DT * B_Ca1_GO);

    // Euler
    //double dCa = - I_Ca1 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca_val  - Ca1_0_GO ); //[mA*mol/cm^3*sec*A]=[M/sec] **[1mM = 1mol/m^3]**
    //elem [ Ca ] [ id ] += dCa * DT;
    // RK4    
    double k1 = DT * ( - I_Ca1 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca_val            - Ca1_0_GO ) ); //[mA*mol/cm^3*sec*A]=[M/sec] **[1mM = 1mol/m^3]**
    double k2 = DT * ( - I_Ca1 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca_val + k1 / 2.0 - Ca1_0_GO ) );
    double k3 = DT * ( - I_Ca1 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca_val + k2 / 2.0 - Ca1_0_GO ) );
    double k4 = DT * ( - I_Ca1 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca_val + k3       - Ca1_0_GO ) );
    elem [ Ca ] [ id ] += ( k1 + k2 * 2.0 + k3 * 2.0 + k4 ) / 6.0;

    //double Cinf = 5e-5 - I_Ca2 / ( 2.0 * 9.6485e4 * 0.2e-4 * 1.3 );
    //go -> ca2 [ id ]  = Cinf - (Cinf - Ca2_val) * exp ( - DT * 1.3);
    
    // Ca2 Euler
    //double dCa2 = - I_Ca2 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca2_val  - Ca1_0_GO ); //[mA*mol/cm^3*sec*A]=[M/sec] **[1mM = 1mol/m^3]**
    //d_go -> ca2 [ id ] += dCa2 * DT;
    // Ca2 RK4
    k1 = DT * ( - I_Ca2 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca2_val            - Ca1_0_GO ) ); //[mA*mol/cm^3*sec*A]=[M/sec] **[1mM = 1mol/m^3]**
    k2 = DT * ( - I_Ca2 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca2_val + k1 / 2.0 - Ca1_0_GO ) );
    k3 = DT * ( - I_Ca2 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca2_val + k2 / 2.0 - Ca1_0_GO ) );
    k4 = DT * ( - I_Ca2 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca2_val + k3       - Ca1_0_GO ) );
    d_go -> ca2 [ id ] += ( k1 + k2 * 2.0 + k3 * 2.0 + k4 ) / 6.0;
    // Ca2 Vrev update
    d_go -> rev_ca2 [ id ] =  ( 1e3 ) * ( 8.313424 * ( 23.0 + 273.15 ) ) / (2 * F_GO ) * log ( Ca1OUT_GO / d_go -> ca2 [ id ] );//[mV]

    //double Cinf = 5e-5 - I_Ca1 / ( 2.0 * 9.6485e4 * 0.2e-4 * 1.3 );
	  //elem [ Ca ] [ id ]  = Cinf - (Cinf - Ca_val) * exp ( - DT * 1.3);

  }
}

__global__ void go_update_ion_RKC ( neuron_t *d_go, neuron_solve_t *d_go_solve, double *elem_v, const double DT )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  double **elem = d_go -> elem;
  double **ion  = d_go -> ion;
  double **cond = d_go -> cond;
  
  if ( id < d_go -> nc)
  {
    //double v_val = elem [ v ] [ id ];
    double v_val = elem_v [ id ];

    double Ca_val = elem [ Ca ] [ id ];
    double Ca2_val = d_go -> ca2 [ id ];

    double I_Ca1 = 1e-3 * cond [ g_CaHVA_go ] [ id ] / elem [ area ] [ id ] * ion [ ch_CaHVA_go ] [ id ] *
      ion [ ch_CaHVA_go ] [ id ] * ion [ ci_CaHVA_go ] [ id ] * ( elem [ v ] [ id ] - V_Ca_GO ); // I_Ca [mA/cm^2]
    double I_Ca2 = 1e-3 * cond [ g_CaLVA_go ] [ id ] / elem [ area ] [ id ] * ion [ cl_CaLVA_go ] [ id ] *
      ion [ cl_CaLVA_go ] [ id ] * ion [ cm_CaLVA_go ] [ id ] * ( elem [ v ] [ id ] - ( d_go -> rev_ca2 [ id ] ) ); // I_Ca [mA/cm^2]   

    ion [ m_NaT_go ] [ id ] = inf_m_NaT ( v_val ) + ( ion [ m_NaT_go ] [ id ] - inf_m_NaT ( v_val ) ) * exp ( -DT / tau_m_NaT ( v_val ) );
    ion [ h_NaT_go ] [ id ] = inf_h_NaT ( v_val ) + ( ion [ h_NaT_go ] [ id ] - inf_h_NaT ( v_val ) ) * exp ( -DT / tau_h_NaT ( v_val ) );
    ion [ r_NaR_go ] [ id ] = inf_r_NaR ( v_val ) + ( ion [ r_NaR_go ] [ id ] - inf_r_NaR ( v_val ) ) * exp ( -DT / tau_r_NaR ( v_val ) );
    ion [ s_NaR_go ] [ id ] = inf_s_NaR ( v_val ) + ( ion [ s_NaR_go ] [ id ] - inf_s_NaR ( v_val ) ) * exp ( -DT / tau_s_NaR ( v_val ) );
    ion [ p_NaP_go ] [ id ] = inf_p_NaP ( v_val ) + ( ion [ p_NaP_go ] [ id ] - inf_p_NaP ( v_val ) ) * exp ( -DT / tau_p_NaP ( v_val ) );

    ion [ n_KV_go ] [ id ] = inf_n_KV ( v_val ) + ( ion [ n_KV_go ] [ id ] - inf_n_KV ( v_val ) ) * exp ( -DT / tau_n_KV ( v_val ) );
    ion [ a_KA_go ] [ id ] = inf_a_KA ( v_val ) + ( ion [ a_KA_go ] [ id ] - inf_a_KA ( v_val ) ) * exp ( -DT / tau_a_KA ( v_val ) );
    ion [ b_KA_go ] [ id ] = inf_b_KA ( v_val ) + ( ion [ b_KA_go ] [ id ] - inf_b_KA ( v_val ) ) * exp ( -DT / tau_b_KA ( v_val ) );
    ion [ c_KC_go ] [ id ] = inf_c_KC ( v_val , Ca_val )
                           + (ion [ c_KC_go ] [ id ] - inf_c_KC ( v_val , Ca_val ) ) * exp ( -DT / tau_c_KC ( v_val , Ca_val ) );
    ion [ sl_Kslow_go ] [ id ] = inf_sl_Kslow ( v_val ) + ( ion [ sl_Kslow_go ] [ id ] - inf_sl_Kslow ( v_val ) ) * exp ( -DT / tau_sl_Kslow ( v_val ) );
		
    ion [ ch_CaHVA_go ] [ id ] = inf_ch_CaHVA( v_val ) + ( ion [ ch_CaHVA_go ] [ id ] - inf_ch_CaHVA ( v_val ) ) * exp ( -DT / tau_ch_CaHVA( v_val ) );
    ion [ ci_CaHVA_go ] [ id ] = inf_ci_CaHVA( v_val ) + ( ion [ ci_CaHVA_go ] [ id ] - inf_ci_CaHVA ( v_val ) ) * exp ( -DT / tau_ci_CaHVA( v_val ) );
    ion [ cl_CaLVA_go ] [ id ] = inf_cl_CaLVA( v_val ) + ( ion [ cl_CaLVA_go ] [ id ] - inf_cl_CaLVA ( v_val ) ) * exp ( -DT / tau_cl_CaLVA( v_val ) );
    ion [ cm_CaLVA_go ] [ id ] = inf_cm_CaLVA( v_val ) + ( ion [ cm_CaLVA_go ] [ id ] - inf_cm_CaLVA ( v_val ) ) * exp ( -DT / tau_cm_CaLVA( v_val ) );

    ion [ hf_HCN1_go ] [ id ] = inf_hf_HCN1( v_val ) + ( ion [ hf_HCN1_go ] [ id ] - inf_hf_HCN1 ( v_val ) ) * exp ( -DT / tau_hf_HCN1( v_val ) );
    ion [ hf_HCN2_go ] [ id ] = inf_hf_HCN2( v_val ) + ( ion [ hf_HCN2_go ] [ id ] - inf_hf_HCN2 ( v_val ) ) * exp ( -DT / tau_hf_HCN2( v_val ) );
    ion [ hs_HCN1_go ] [ id ] = inf_hs_HCN1( v_val ) + ( ion [ hs_HCN1_go ] [ id ] - inf_hs_HCN1 ( v_val ) ) * exp ( -DT / tau_hs_HCN1( v_val ) );
    ion [ hs_HCN2_go ] [ id ] = inf_hs_HCN2( v_val ) + ( ion [ hs_HCN2_go ] [ id ] - inf_hs_HCN2 ( v_val ) ) * exp ( -DT / tau_hs_HCN2( v_val ) );

    // integral
    //elem [ Ca ] [ id ] = ( DT * ( - I_Ca1 / (  2.0 * F_GO * SHELL1_D_GO ) + B_Ca1_GO *  Ca1_0_GO) + Ca_val ) / (1.0 + DT * B_Ca1_GO);

    // Euler
    double dCa = - I_Ca1 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca_val  - Ca1_0_GO ); //[mA*mol/cm^3*sec*A]=[M/sec] **[1mM = 1mol/m^3]**
    elem [ Ca ] [ id ] += dCa * DT;
    // RK4    
    //double k1 = DT * ( - I_Ca1 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca_val            - Ca1_0_GO ) ); //[mA*mol/cm^3*sec*A]=[M/sec] **[1mM = 1mol/m^3]**
    //double k2 = DT * ( - I_Ca1 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca_val + k1 / 2.0 - Ca1_0_GO ) );
    //double k3 = DT * ( - I_Ca1 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca_val + k2 / 2.0 - Ca1_0_GO ) );
    //double k4 = DT * ( - I_Ca1 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca_val + k3       - Ca1_0_GO ) );
    //elem [ Ca ] [ id ] += ( k1 + k2 * 2.0 + k3 * 2.0 + k4 ) / 6.0;

    //double Cinf = 5e-5 - I_Ca2 / ( 2.0 * 9.6485e4 * 0.2e-4 * 1.3 );
    //go -> ca2 [ id ]  = Cinf - (Cinf - Ca2_val) * exp ( - DT * 1.3);
    
    // Ca2 Euler
    double dCa2 = - I_Ca2 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca2_val  - Ca1_0_GO ); //[mA*mol/cm^3*sec*A]=[M/sec] **[1mM = 1mol/m^3]**
    d_go -> ca2 [ id ] += dCa2 * DT;
    // Ca2 RK4
    //k1 = DT * ( - I_Ca2 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca2_val            - Ca1_0_GO ) ); //[mA*mol/cm^3*sec*A]=[M/sec] **[1mM = 1mol/m^3]**
    //k2 = DT * ( - I_Ca2 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca2_val + k1 / 2.0 - Ca1_0_GO ) );
    //k3 = DT * ( - I_Ca2 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca2_val + k2 / 2.0 - Ca1_0_GO ) );
    //k4 = DT * ( - I_Ca2 / ( 2.0 * F_GO * SHELL1_D_GO ) - B_Ca1_GO * ( Ca2_val + k3       - Ca1_0_GO ) );
    //d_go -> ca2 [ id ] += ( k1 + k2 * 2.0 + k3 * 2.0 + k4 ) / 6.0;
    // Ca2 Vrev update
    d_go -> rev_ca2 [ id ] =  ( 1e3 ) * ( 8.313424 * ( 23.0 + 273.15 ) ) / (2 * F_GO ) * log ( Ca1OUT_GO / d_go -> ca2 [ id ] );//[mV]

    //double Cinf = 5e-5 - I_Ca1 / ( 2.0 * 9.6485e4 * 0.2e-4 * 1.3 );
	  //elem [ Ca ] [ id ]  = Cinf - (Cinf - Ca_val) * exp ( - DT * 1.3);

  }
}

__host__ void go_initialize_ion ( neuron_t *go )
{
  double **elem = go -> elem;
  double **ion  = go -> ion;
  double init_v_rand = 0.0;
  for ( int i = 0; i < go -> nc; i++) {
    
    if ( i % GO_COMP == 0 )
      init_v_rand = ( ( double ) rand ( ) / RAND_MAX ) - 0.5;

    elem [ v     ] [ i ] = V_INIT_GO + 2.0 * init_v_rand;
    elem [ Ca    ] [ i ] = Ca1_0_GO;
    go -> ca2 [ i ] = Ca1_0_GO;
    go -> rev_ca2 [ i ] = V_Ca_GO;
    go -> ca_old  [ i ] = Ca1_0_GO;
    elem [ i_ext ] [ i ] = 0.0;
    double v_val = elem [ v  ] [ i ];
    double ca_val = elem [ Ca ] [ i ];

    ion [ m_NaT_go ] [ i ] = inf_m_NaT ( v_val );
    ion [ h_NaT_go ] [ i ] = inf_h_NaT ( v_val );
    ion [ r_NaR_go ] [ i ] = inf_r_NaR ( v_val );
    ion [ s_NaR_go ] [ i ] = inf_s_NaR ( v_val );
    ion [ p_NaP_go ] [ i ] = inf_p_NaP ( v_val );
    ion [ n_KV_go  ] [ i ] = inf_n_KV ( v_val );
    ion [ a_KA_go  ] [ i ] = inf_a_KA ( v_val );
    ion [ b_KA_go  ] [ i ] = inf_b_KA ( v_val );
    ion [ c_KC_go  ] [ i ] = inf_c_KC ( v_val, ca_val );
    ion [ sl_Kslow_go ] [ i ] = inf_sl_Kslow ( v_val );
    ion [ ch_CaHVA_go ] [ i ] = inf_ch_CaHVA ( v_val );
    ion [ ci_CaHVA_go ] [ i ] = inf_ci_CaHVA ( v_val );
    ion [ cl_CaLVA_go ] [ i ] = inf_cl_CaLVA ( v_val );
    ion [ cm_CaLVA_go ] [ i ] = inf_cm_CaLVA ( v_val );
    ion [ hf_HCN1_go ] [ i ] = inf_hf_HCN1 ( v_val );
    ion [ hf_HCN2_go ] [ i ] = inf_hf_HCN2 ( v_val );
    ion [ hs_HCN1_go ] [ i ] = inf_hs_HCN1 ( v_val );
    ion [ hs_HCN2_go ] [ i ] = inf_hs_HCN2 ( v_val );
    ion [ o1_KAHP_go ] [ i ] = 0.0;
    ion [ o2_KAHP_go ] [ i ] = 0.0;
    ion [ c2_KAHP_go ] [ i ] = 0.0;
    ion [ c3_KAHP_go ] [ i ] = 0.0;
    ion [ c4_KAHP_go ] [ i ] = 0.0;
    ion [ c1_KAHP_go ] [ i ] = 1.0;
  }
}
