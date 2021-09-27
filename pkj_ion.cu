#include "pkj_ion.cuh"

// 
static __host__ __device__ double l_func_pkj ( const double v, const double A, const double B, const double C, const double D){
    return A / ( B + exp ( ( v + C ) / D ) );
  }
  
  // alpha
  static __host__ __device__ double alpha_m_NaF(const double v){return l_func_pkj(v,35.0,  0.0,5.0,  -10.0);}
  static __host__ __device__ double alpha_h_NaF(const double v){return l_func_pkj(v,0.225, 1.0,80.0, 10.0);}
  static __host__ __device__ double alpha_m_NaP(const double v){return l_func_pkj(v,200.0, 1.0,-18.0,-16.0);}
  static __host__ __device__ double alpha_m_CaP(const double v){return l_func_pkj(v,8.5,   1.0,-8.0, -12.5);}
  static __host__ __device__ double alpha_h_CaP(const double v){return l_func_pkj(v,0.0015,1.0,29.0, 8.0);}
  static __host__ __device__ double alpha_m_CaT(const double v){return l_func_pkj(v,2.60,  1.0,21.0, -8.0);}
  static __host__ __device__ double alpha_h_CaT(const double v){return l_func_pkj(v,0.0025,1.0,40.0, 8.0);}
  //static __host__ __device__ double alpha_m_KDR(const double v){return -0.0235*(v+12)/(exp(-(v+12)/12)-1);}
  static __host__ __device__ double alpha_m_KA(const double v){return l_func_pkj(v,1.40,  1.0,27.0, -12.0);}
  static __host__ __device__ double alpha_h_KA(const double v){return l_func_pkj(v,0.0175,1.0,50.0, 8.0);}
  static __host__ __device__ double alpha_m_KC(const double v){ return 7.5; }
  static __host__ __device__ double alpha_m_K2(const double v){ return 25.0; }
  
  // beta
  static __host__ __device__ double beta_m_NaF(const double v){return l_func_pkj(v,7.0,   0.0, 65.0, 20.0);}
  static __host__ __device__ double beta_h_NaF(const double v){return l_func_pkj(v,7.5,   0.0, -3.0, -18.0);}
  static __host__ __device__ double beta_m_NaP(const double v){return l_func_pkj(v,25.0,  1.0, 58.0, 8.0);}
  static __host__ __device__ double beta_m_CaP(const double v){return l_func_pkj(v,35.0,    1.0, 74.0, 14.5);}
  static __host__ __device__ double beta_h_CaP(const double v){return l_func_pkj(v,0.0055,1.0, 23.0, -8.0);}
  static __host__ __device__ double beta_m_CaT(const double v){return l_func_pkj(v,0.180, 1.0, 40.0, 4.0);}
  static __host__ __device__ double beta_h_CaT(const double v){return l_func_pkj(v,0.190, 1.0, 50.0, -10.0);}
  //static __host__ __device__ double beta_m_KDR(const double v){return 5.0*exp(-(v+147)/30);}
  static __host__ __device__ double beta_m_KA(const double v){return l_func_pkj(v,0.490, 1.0, 30.0, 4.0);}
  static __host__ __device__ double beta_h_KA(const double v){return l_func_pkj(v,1.30,  1.0, 13.0, -10.0);}
  static __host__ __device__ double beta_m_KC(const double v){return l_func_pkj(v,0.110, 0.0, -35.0,14.9);}
  //static __host__ __device__ double beta_m_K2(const double v){return l_func_pkj(v,0.075, 0, 5,  10.0);}
  static __host__ __device__ double beta_m_K2(const double v){return l_func_pkj(v,0.075, 0.0, 25.0,  6.0);}
  
  // tau
  static __host__ __device__ double tau_m_NaF(const double v){ return 1.0   / (alpha_m_NaF(v) + beta_m_NaF(v)); }
  static __host__ __device__ double tau_h_NaF(const double v){ return 1.0   / (alpha_h_NaF(v) + beta_h_NaF(v)); }
  static __host__ __device__ double tau_m_NaP(const double v){ return 1.0   / (alpha_m_NaP(v) + beta_m_NaP(v)); }
  static __host__ __device__ double tau_m_CaP(const double v){ return 1.0   / (alpha_m_CaP(v) + beta_m_CaP(v)); }
  static __host__ __device__ double tau_h_CaP(const double v){ return 1.0   / (alpha_h_CaP(v) + beta_h_CaP(v)); }
  static __host__ __device__ double tau_m_CaT(const double v){ return 1.0   / (alpha_m_CaT(v) + beta_m_CaT(v)); }
  static __host__ __device__ double tau_h_CaT(const double v){ return 1.0   / (alpha_h_CaT(v) + beta_h_CaT(v)); }
  static __host__ __device__ double tau_m_KA(const double v){ return 1.0    / (alpha_m_KA(v) + beta_m_KA(v)); }
  static __host__ __device__ double tau_h_KA(const double v){ return 1.0    / (alpha_h_KA(v) + beta_h_KA(v)); }
  static __host__ __device__ double tau_m_KC(const double v){ return 1.0    / (alpha_m_KC(v) + beta_m_KC(v)); }
  static __host__ __device__ double tau_m_K2(const double v){ return 1.0    / (alpha_m_K2(v) + beta_m_K2(v)); }
  static __host__ __device__ double tau_m_Kh1(const double v){ return 7.6; }
  static __host__ __device__ double tau_m_Kh2(const double v){ return 36.8; }
  static __host__ __device__ double tau_m_Kdr(const double v){
    double alpha;
    (fabs(v+12.0)>1e-6)?
      alpha=-0.0235*(v+12.0)/(exp(-(v+12.0)/12.0)-1.0) : alpha=0.0235*12.0;
    double beta=5.0*exp(-(v+147.0)/30.0);
    return 1.0/(alpha+beta);
  }
  static __host__ __device__ double tau_h_Kdr(const double v){ return (v<-25.0) ? 1200.0 : 10.0; }
  static __host__ __device__ double tau_m_KM(const double v){ return 200.0/(3.3*(exp ((v + 35.0)/20.0)) + exp (-(v + 35.0)/20.0) );}
  static __host__ __device__ double tau_z_KC(const double v){ return 10.0; }
  static __host__ __device__ double tau_z_K2(const double v){ return 10.0; }
  
  // inf
  static __host__ __device__ double inf_m_NaF(const double v){ return alpha_m_NaF(v) / (alpha_m_NaF(v) + beta_m_NaF(v)); }
  static __host__ __device__ double inf_h_NaF(const double v){ return alpha_h_NaF(v) / (alpha_h_NaF(v) + beta_h_NaF(v)); }
  static __host__ __device__ double inf_m_NaP(const double v){ return alpha_m_NaP(v) / (alpha_m_NaP(v) + beta_m_NaP(v)); }
  static __host__ __device__ double inf_m_CaP(const double v){ return alpha_m_CaP(v) / (alpha_m_CaP(v) + beta_m_CaP(v)); }
  static __host__ __device__ double inf_h_CaP(const double v){ return alpha_h_CaP(v) / (alpha_h_CaP(v) + beta_h_CaP(v)); }
  static __host__ __device__ double inf_m_CaT(const double v){ return alpha_m_CaT(v) / (alpha_m_CaT(v) + beta_m_CaT(v)); }
  static __host__ __device__ double inf_h_CaT(const double v){ return alpha_h_CaT(v) / (alpha_h_CaT(v) + beta_h_CaT(v)); }
  static __host__ __device__ double inf_m_KA(const double v){ return alpha_m_KA(v) / (alpha_m_KA(v) + beta_m_KA(v)); }
  static __host__ __device__ double inf_h_KA(const double v){ return alpha_h_KA(v) / (alpha_h_KA(v) + beta_h_KA(v)); }
  static __host__ __device__ double inf_m_KC(const double v){ return alpha_m_KC(v) / (alpha_m_KC(v) + beta_m_KC(v)); }
  static __host__ __device__ double inf_m_K2(const double v){ return alpha_m_K2(v) / (alpha_m_K2(v) + beta_m_K2(v)); }
  static __host__ __device__ double inf_m_Kh1(const double v){return 0.8/(1.0+exp((v+82.0)/7.0) );}
  static __host__ __device__ double inf_m_Kh2(const double v){return 0.2/(1.0+exp((v+82.0)/7.0) );}
  static __host__ __device__ double inf_m_Kdr(const double v){
    double alpha;
    (fabs(v-8.0)>1e-6)?
      alpha=-0.0235*(v-8.0)/(exp(-(v-8.0)/12.0)-1.0) : alpha=0.0235*12.0;
    double beta=5.0*exp(-(v+127.0)/30.0);
    return alpha/(alpha+beta);
  }
  static __host__ __device__ double inf_h_Kdr(const double v){return 1.0/(1.0+exp((v+25.0)/4.0) );}
  static __host__ __device__ double inf_m_KM(const double v){return 1.0/(1.0+exp(-(v+35.0)/10.0) );}
  static __host__ __device__ double inf_z_KC(const double v,const double Ca){
    return 1.0/(1.0+4.00/Ca);
  }
  static __host__ __device__ double inf_z_K2(const double v,const double Ca){
    return 1.0/(1.0+0.20/Ca);
  }
  

__global__ void pkj_update_ion ( neuron_t *d_pkj, neuron_solve_t *d_pkj_solve, const double DT )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if ( id < d_pkj -> nc )
  {
    double **elem = d_pkj -> elem;
    double **ion  = d_pkj -> ion;
    double **cond = d_pkj -> cond;
    
    double v_val = elem [ v ] [ id ];
    double Ca_val = elem [ Ca ] [ id ];
    double l_v_Ca = d_pkj -> rev_ca2 [ id ];
    double cafloor = floor ( ( Ca_val * 1000 - 0.04 ) / 0.099986667 ) * 0.099986667 + 0.04;
    //double cafloor = Ca_val * 1000.0;
    double I_Ca1 = 1e-3 * ( v_val - l_v_Ca )  // I_Ca [mA/cm^2]
                        * ( cond [ g_CaP_pkj ] [ id ] * ion [ m_CaP_pkj ] [ id ] * ion [ h_CaP_pkj ] [ id ]  
                          + cond [ g_CaT_pkj ] [ id ] * ion [ m_CaT_pkj ] [ id ] * ion [ h_CaT_pkj ] [ id ] );
    ( I_Ca1 > 0.0 )? I_Ca1 = 0.0 : I_Ca1 * 1.0;

    ion [ m_NaF_pkj ] [ id ] = inf_m_NaF ( v_val ) + ( ion [ m_NaF_pkj ] [ id ] - inf_m_NaF ( v_val ) ) * exp ( - DT / tau_m_NaF ( v_val ) );
    ion [ h_NaF_pkj ] [ id ] = inf_h_NaF ( v_val ) + ( ion [ h_NaF_pkj ] [ id ] - inf_h_NaF ( v_val ) ) * exp ( - DT / tau_h_NaF ( v_val ) );
    ion [ m_NaP_pkj ] [ id ] = inf_m_NaP ( v_val ) + ( ion [ m_NaP_pkj ] [ id ] - inf_m_NaP ( v_val ) ) * exp ( - DT / tau_m_NaP ( v_val ) );
    ion [ m_CaP_pkj ] [ id ] = inf_m_CaP ( v_val ) + ( ion [ m_CaP_pkj ] [ id ] - inf_m_CaP ( v_val ) ) * exp ( - DT / tau_m_CaP ( v_val ) );
    ion [ h_CaP_pkj ] [ id ] = inf_h_CaP ( v_val ) + ( ion [ h_CaP_pkj ] [ id ] - inf_h_CaP ( v_val ) ) * exp ( - DT / tau_h_CaP ( v_val ) );
    ion [ m_CaT_pkj ] [ id ] = inf_m_CaT ( v_val ) + ( ion [ m_CaT_pkj ] [ id ] - inf_m_CaT ( v_val ) ) * exp ( - DT / tau_m_CaT ( v_val ) );
    ion [ h_CaT_pkj ] [ id ] = inf_h_CaT ( v_val ) + ( ion [ h_CaT_pkj ] [ id ] - inf_h_CaT ( v_val ) ) * exp ( - DT / tau_h_CaT ( v_val ) );
    ion [ m_Kh1_pkj ] [ id ] = inf_m_Kh1 ( v_val ) + ( ion [ m_Kh1_pkj ] [ id ] - inf_m_Kh1 ( v_val ) ) * exp ( - DT / tau_m_Kh1 ( v_val ) );
    ion [ m_Kh2_pkj ] [ id ] = inf_m_Kh2 ( v_val ) + ( ion [ m_Kh2_pkj ] [ id ] - inf_m_Kh2 ( v_val ) ) * exp ( - DT / tau_m_Kh2 ( v_val ) );
    ion [ m_Kdr_pkj ] [ id ] = inf_m_Kdr ( v_val ) + ( ion [ m_Kdr_pkj ] [ id ] - inf_m_Kdr ( v_val ) ) * exp ( - DT / tau_m_Kdr ( v_val ) );
    ion [ h_Kdr_pkj ] [ id ] = inf_h_Kdr ( v_val ) + ( ion [ h_Kdr_pkj ] [ id ] - inf_h_Kdr ( v_val ) ) * exp ( - DT / tau_h_Kdr ( v_val ) );
    ion [ m_KM_pkj  ] [ id ] = inf_m_KM  ( v_val ) + ( ion [ m_KM_pkj  ] [ id ] - inf_m_KM  ( v_val ) ) * exp ( - DT / tau_m_KM  ( v_val ) ); 
    ion [ m_KA_pkj  ] [ id ] = inf_m_KA  ( v_val ) + ( ion [ m_KA_pkj  ] [ id ] - inf_m_KA  ( v_val ) ) * exp ( - DT / tau_m_KA  ( v_val ) ); 
    ion [ h_KA_pkj  ] [ id ] = inf_h_KA  ( v_val ) + ( ion [ h_KA_pkj  ] [ id ] - inf_h_KA  ( v_val ) ) * exp ( - DT / tau_h_KA  ( v_val ) );  
    ion [ m_KC_pkj  ] [ id ] = inf_m_KC  ( v_val ) + ( ion [ m_KC_pkj  ] [ id ] - inf_m_KC  ( v_val ) ) * exp ( - DT / tau_m_KC  ( v_val ) );  
    ion [ z_KC_pkj  ] [ id ] = inf_z_KC  ( v_val, cafloor ) + ( ion [ z_KC_pkj  ] [ id ] - inf_z_KC  ( v_val, cafloor ) ) * exp ( - DT / tau_z_KC  ( v_val ) );  
    ion [ m_K2_pkj  ] [ id ] = inf_m_K2  ( v_val ) + ( ion [ m_K2_pkj  ] [ id ] - inf_m_K2 ( v_val ) ) * exp ( - DT / tau_m_K2   ( v_val ) );
    ion [ z_K2_pkj  ] [ id ] = inf_z_K2  ( v_val, cafloor ) + ( ion [ z_K2_pkj  ] [ id ] - inf_z_K2 ( v_val, cafloor ) ) * exp ( - DT / tau_z_K2   ( v_val ) );
 
    double l_shell = d_pkj -> shell [ id ] * elem [ area ] [ id ];
    double k1 = DT * ( - I_Ca1 / ( 2.0 * F_PKJ * l_shell ) - B_Ca1_PKJ * ( Ca_val            - Ca1_0_PKJ ) ); //[mA*mol/cm^3*sec*A]=[M/sec] **[1mM = 1mol/m^3]**
    double k2 = DT * ( - I_Ca1 / ( 2.0 * F_PKJ * l_shell ) - B_Ca1_PKJ * ( Ca_val + k1 / 2.0 - Ca1_0_PKJ ) );
    double k3 = DT * ( - I_Ca1 / ( 2.0 * F_PKJ * l_shell ) - B_Ca1_PKJ * ( Ca_val + k2 / 2.0 - Ca1_0_PKJ ) );
    double k4 = DT * ( - I_Ca1 / ( 2.0 * F_PKJ * l_shell ) - B_Ca1_PKJ * ( Ca_val + k3       - Ca1_0_PKJ ) );
    elem [ Ca ] [ id ] += ( k1 + k2 * 2.0 + k3 * 2.0 + k4 ) / 6.0;
    //double inf_Ca = - I_Ca1 / ( 2.0 * F_PKJ * l_shell ) + Ca1_0_PKJ;
    //elem [ Ca ] [ id ] = inf_Ca + ( Ca_val - inf_Ca ) * exp ( - DT / 0.1 );

    ( elem [ compart ] [ id ] == PKJ_soma )?
      d_pkj -> rev_ca2 [ id ] = 12.5 * log ( Ca1OUT_PKJ / Ca1_0_PKJ ) :    
      d_pkj -> rev_ca2 [ id ] = 13.361624877 * log ( Ca1OUT_PKJ / Ca_val ); //310.15*8.3134/2/96485.309*1000
  }
}

static __device__ double dmdt ( const double m, const double inf_m, const double tau_m  )
{
  return ( 1.0 / tau_m ) * ( - m + inf_m );
}

__global__ void pkj_update_ion_RK2 ( neuron_t *d_pkj, neuron_solve_t *d_pkj_solve, const double DT )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
    
  if ( id < d_pkj -> nc)
  {
    double **elem = d_pkj -> elem;
    double **ion  = d_pkj -> ion;
    double **cond = d_pkj -> cond;
    
    double v_val = elem [ v ] [ id ];
    double Ca_val = elem [ Ca ] [ id ];
    //double cafloor = floor ( ( Ca_val * 1000 - 0.04 ) / 0.099986667 ) * 0.099986667 + 0.04;
    double l_shell = d_pkj -> shell [ id ] * elem [ area ] [ id ];
    double Ca_rev  = d_pkj -> rev_ca2 [ id ];

    double cafloor = Ca_val * 1000.0;
    double mNaF_1 = dmdt ( ion [ m_NaF_pkj ] [ id ], inf_m_NaF ( v_val ), tau_m_NaF ( v_val ) );
    double hNaF_1 = dmdt ( ion [ h_NaF_pkj ] [ id ], inf_h_NaF ( v_val ), tau_h_NaF ( v_val ) );
    double mNaP_1 = dmdt ( ion [ m_NaP_pkj ] [ id ], inf_m_NaP ( v_val ), tau_m_NaP ( v_val ) );
    double mCaP_1 = dmdt ( ion [ m_CaP_pkj ] [ id ], inf_m_CaP ( v_val ), tau_m_CaP ( v_val ) );
    double hCaP_1 = dmdt ( ion [ h_CaP_pkj ] [ id ], inf_h_CaP ( v_val ), tau_h_CaP ( v_val ) );
    double mCaT_1 = dmdt ( ion [ m_CaT_pkj ] [ id ], inf_m_CaT ( v_val ), tau_m_CaT ( v_val ) );
    double hCaT_1 = dmdt ( ion [ h_CaT_pkj ] [ id ], inf_h_CaT ( v_val ), tau_h_CaT ( v_val ) );
    double mKh1_1 = dmdt ( ion [ m_Kh1_pkj ] [ id ], inf_m_Kh1 ( v_val ), tau_m_Kh1 ( v_val ) );
    double mKh2_1 = dmdt ( ion [ m_Kh2_pkj ] [ id ], inf_m_Kh2 ( v_val ), tau_m_Kh2 ( v_val ) );
    double mKdr_1 = dmdt ( ion [ m_Kdr_pkj ] [ id ], inf_m_Kdr ( v_val ), tau_m_Kdr ( v_val ) );
    double hKdr_1 = dmdt ( ion [ h_Kdr_pkj ] [ id ], inf_h_Kdr ( v_val ), tau_h_Kdr ( v_val ) );
    double mKM_1  = dmdt ( ion [ m_KM_pkj  ] [ id ], inf_m_KM  ( v_val ), tau_m_KM  ( v_val ) );
    double mKA_1  = dmdt ( ion [ m_KA_pkj  ] [ id ], inf_m_KA  ( v_val ), tau_m_KA  ( v_val ) );
    double hKA_1  = dmdt ( ion [ h_KA_pkj  ] [ id ], inf_h_KA  ( v_val ), tau_h_KA  ( v_val ) );
    double mKC_1  = dmdt ( ion [ m_KC_pkj  ] [ id ], inf_m_KC  ( v_val ), tau_m_KC  ( v_val ) );
    double zKC_1  = dmdt ( ion [ z_KC_pkj  ] [ id ], inf_z_KC  ( v_val, cafloor ), tau_z_KC  ( v_val ) );
    double mK2_1  = dmdt ( ion [ m_K2_pkj  ] [ id ], inf_m_K2  ( v_val ), tau_m_K2  ( v_val ) );
    double zK2_1  = dmdt ( ion [ z_K2_pkj  ] [ id ], inf_z_K2  ( v_val, cafloor ), tau_z_K2  ( v_val ) );
    double I_Ca1 = 1e-3 * ( v_val - Ca_rev )  // I_Ca [mA/cm^2]
                        * ( cond [ g_CaP_pkj ] [ id ] * ion [ m_CaP_pkj ] [ id ] * ion [ h_CaP_pkj ] [ id ]  
                          + cond [ g_CaT_pkj ] [ id ] * ion [ m_CaT_pkj ] [ id ] * ion [ h_CaT_pkj ] [ id ] );
    ( I_Ca1 > 0.0 )? I_Ca1 = 0.0 : I_Ca1 * 1.0;
    double k1 = ( - I_Ca1 / ( 2.0 * F_PKJ * l_shell ) - B_Ca1_PKJ * ( Ca_val - Ca1_0_PKJ ) ); //[mA*mol/cm^3*sec*A]=[M/sec] **[1mM = 1mol/m^3]**
    
    ( elem [ compart ] [ id ] == PKJ_soma )?
      Ca_rev = 12.5 * log ( Ca1OUT_PKJ / Ca1_0_PKJ ) :    
      Ca_rev = 13.361624877 * log ( Ca1OUT_PKJ / ( Ca_val + DT * k1 ) ); //310.15*8.3134/2/96485.309*1000
    cafloor = ( Ca_val + DT * k1 ) * 1000.0;
    double mNaF_2 = dmdt ( ion [ m_NaF_pkj ] [ id ] + DT * mNaF_1, inf_m_NaF ( v_val ), tau_m_NaF ( v_val ) );
    double hNaF_2 = dmdt ( ion [ h_NaF_pkj ] [ id ] + DT * hNaF_1, inf_h_NaF ( v_val ), tau_h_NaF ( v_val ) );
    double mNaP_2 = dmdt ( ion [ m_NaP_pkj ] [ id ] + DT * mNaP_1, inf_m_NaP ( v_val ), tau_m_NaP ( v_val ) );
    double mCaP_2 = dmdt ( ion [ m_CaP_pkj ] [ id ] + DT * mCaP_1, inf_m_CaP ( v_val ), tau_m_CaP ( v_val ) );
    double hCaP_2 = dmdt ( ion [ h_CaP_pkj ] [ id ] + DT * hCaP_1, inf_h_CaP ( v_val ), tau_h_CaP ( v_val ) );
    double mCaT_2 = dmdt ( ion [ m_CaT_pkj ] [ id ] + DT * mCaT_1, inf_m_CaT ( v_val ), tau_m_CaT ( v_val ) );
    double hCaT_2 = dmdt ( ion [ h_CaT_pkj ] [ id ] + DT * hCaT_1, inf_h_CaT ( v_val ), tau_h_CaT ( v_val ) );
    double mKh1_2 = dmdt ( ion [ m_Kh1_pkj ] [ id ] + DT * mKh1_1, inf_m_Kh1 ( v_val ), tau_m_Kh1 ( v_val ) );
    double mKh2_2 = dmdt ( ion [ m_Kh2_pkj ] [ id ] + DT * mKh2_1, inf_m_Kh2 ( v_val ), tau_m_Kh2 ( v_val ) );
    double mKdr_2 = dmdt ( ion [ m_Kdr_pkj ] [ id ] + DT * mKdr_1, inf_m_Kdr ( v_val ), tau_m_Kdr ( v_val ) );
    double hKdr_2 = dmdt ( ion [ h_Kdr_pkj ] [ id ] + DT * hKdr_1, inf_h_Kdr ( v_val ), tau_h_Kdr ( v_val ) );
    double mKM_2  = dmdt ( ion [ m_KM_pkj  ] [ id ] + DT * mKM_1, inf_m_KM  ( v_val ), tau_m_KM  ( v_val ) );
    double mKA_2  = dmdt ( ion [ m_KA_pkj  ] [ id ] + DT * mKA_1, inf_m_KA  ( v_val ), tau_m_KA  ( v_val ) );
    double hKA_2  = dmdt ( ion [ h_KA_pkj  ] [ id ] + DT * hKA_1, inf_h_KA  ( v_val ), tau_h_KA  ( v_val ) );
    double mKC_2  = dmdt ( ion [ m_KC_pkj  ] [ id ] + DT * mKC_1, inf_m_KC  ( v_val ), tau_m_KC  ( v_val ) );
    double zKC_2  = dmdt ( ion [ z_KC_pkj  ] [ id ] + DT * zKC_1, inf_z_KC  ( v_val, cafloor ), tau_z_KC  ( v_val ) );
    double mK2_2  = dmdt ( ion [ m_K2_pkj  ] [ id ] + DT * mK2_1, inf_m_K2  ( v_val ), tau_m_K2  ( v_val ) );
    double zK2_2  = dmdt ( ion [ z_K2_pkj  ] [ id ] + DT * zK2_1, inf_z_K2  ( v_val, cafloor ), tau_z_K2  ( v_val ) );
    double I_Ca2 = 1e-3 * ( v_val - Ca_rev )  // I_Ca [mA/cm^2]
                        * ( cond [ g_CaP_pkj ] [ id ] * ( ion [ m_CaP_pkj ] [ id ] + DT * mCaP_1 ) * ( ion [ h_CaP_pkj ] [ id ] + DT * hCaP_1 )
                          + cond [ g_CaT_pkj ] [ id ] * ( ion [ m_CaT_pkj ] [ id ] + DT * mCaT_1 ) * ( ion [ h_CaT_pkj ] [ id ] + DT * hCaT_1 ) );
    ( I_Ca2 > 0.0 )? I_Ca2 = 0.0 : I_Ca2 * 1.0;
    double k2 = ( - I_Ca2 / ( 2.0 * F_PKJ * l_shell ) - B_Ca1_PKJ * ( Ca_val + DT * k1 - Ca1_0_PKJ ) );    

    ion [ m_NaF_pkj ] [ id ] += DT * ( mNaF_1 + mNaF_2 ) / 2.0;
    ion [ h_NaF_pkj ] [ id ] += DT * ( hNaF_1 + hNaF_2 ) / 2.0;
    ion [ m_NaP_pkj ] [ id ] += DT * ( mNaP_1 + mNaP_2 ) / 2.0;
    ion [ m_CaP_pkj ] [ id ] += DT * ( mCaP_1 + mCaP_2 ) / 2.0;
    ion [ h_CaP_pkj ] [ id ] += DT * ( hCaP_1 + hCaP_2 ) / 2.0;
    ion [ m_CaT_pkj ] [ id ] += DT * ( mCaT_1 + mCaT_2 ) / 2.0;
    ion [ h_CaT_pkj ] [ id ] += DT * ( hCaT_1 + hCaT_2 ) / 2.0;
    ion [ m_Kh1_pkj ] [ id ] += DT * ( mKh1_1 + mKh1_2 ) / 2.0;
    ion [ m_Kh2_pkj ] [ id ] += DT * ( mKh2_1 + mKh2_2 ) / 2.0;
    ion [ m_Kdr_pkj ] [ id ] += DT * ( mKdr_1 + mKdr_2 ) / 2.0;
    ion [ h_Kdr_pkj ] [ id ] += DT * ( hKdr_1 + hKdr_2 ) / 2.0;
    ion [ m_KM_pkj  ] [ id ] += DT * ( mKM_1  + mKM_2  ) / 2.0;
    ion [ m_KA_pkj  ] [ id ] += DT * ( mKA_1  + mKA_2  ) / 2.0;
    ion [ h_KA_pkj  ] [ id ] += DT * ( hKA_1  + hKA_2  ) / 2.0;
    ion [ m_KC_pkj  ] [ id ] += DT * ( mKC_1  + mKC_2  ) / 2.0;
    ion [ z_KC_pkj  ] [ id ] += DT * ( zKC_1  + zKC_2  ) / 2.0;
    ion [ m_K2_pkj  ] [ id ] += DT * ( mK2_1  + mK2_2  ) / 2.0;
    ion [ z_K2_pkj  ] [ id ] += DT * ( zK2_1  + zK2_2  ) / 2.0;    
    elem [ Ca ] [ id ]       += DT * ( k1 + k2 ) / 2.0;
    ( elem [ compart ] [ id ] == PKJ_soma )?
      d_pkj -> rev_ca2 [ id ] = 12.5 * log ( Ca1OUT_PKJ / Ca1_0_PKJ ) :    
      d_pkj -> rev_ca2 [ id ] = 13.361624877 * log ( Ca1OUT_PKJ / elem [ Ca ] [ id ] ); //310.15*8.3134/2/96485.309*1000
  }
}

__global__ void pkj_update_ion_2nd ( neuron_t *d_pkj, neuron_solve_t *d_pkj_solve, const double DT )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
    
  if ( id < d_pkj -> nc)
  {
    double **elem = d_pkj -> elem;
    double **ion  = d_pkj -> ion;

    double v_val = ( 1.5 * elem [ v ] [ id ] - 0.5 * d_pkj_solve -> vec [ cn_v_old ] [ id ] );
    //double l_shell = d_pkj -> shell [ id ];// * elem [ area ] [ id ];
    double Ca_val = elem [ Ca ] [ id ];
    double I_Ca1 = 1e-3 * ( elem [ v ] [ id ] - d_pkj -> rev_ca2 [ id ] )  // I_Ca [mA/cm^2]
                        * ( d_pkj -> cond [ g_CaP_pkj ] [ id ] * ion [ m_CaP_pkj ] [ id ] * ion [ h_CaP_pkj ] [ id ]  
                          + d_pkj -> cond [ g_CaT_pkj ] [ id ] * ion [ m_CaT_pkj ] [ id ] * ion [ h_CaT_pkj ] [ id ] );
    ( I_Ca1 > 0.0 )? I_Ca1 = 0.0 : I_Ca1  = - I_Ca1 / ( 2.0 * F_PKJ * d_pkj -> shell [ id ] );
    double k1 = DT * ( I_Ca1 - B_Ca1_PKJ * ( Ca_val            - Ca1_0_PKJ ) ); //[mA*mol/cm^3*sec*A]=[M/sec] **[1mM = 1mol/m^3]**
    double k2 = DT * ( I_Ca1 - B_Ca1_PKJ * ( Ca_val + k1 / 2.0 - Ca1_0_PKJ ) );
    double k3 = DT * ( I_Ca1 - B_Ca1_PKJ * ( Ca_val + k2 / 2.0 - Ca1_0_PKJ ) );
    double k4 = DT * ( I_Ca1 - B_Ca1_PKJ * ( Ca_val + k3       - Ca1_0_PKJ ) );

    k1 = Ca_val + ( k1 + k2 * 2.0 + k3 * 2.0 + k4 ) / 6.0;
    elem [ Ca ] [ id ] = k1;
    Ca_val = 0.5 * ( Ca_val + k1 );
    
    //( elem [ compart ] [ id ] == PKJ_soma )?
    ( id % 1599 == 0 )?
      d_pkj -> rev_ca2 [ id ] = 12.5 * log ( Ca1OUT_PKJ / Ca1_0_PKJ ) :    
      d_pkj -> rev_ca2 [ id ] = 13.361624877 * log ( Ca1OUT_PKJ / Ca_val ); //310.15*8.3134/2/96485.309*1000
    //l_v_Ca = 0.5 * ( l_v_Ca + d_pkj -> rev_ca2 [ id ] );    

    //double cafloor = floor ( ( Ca_val * 1000 - 0.04 ) * 10.0 ) * 0.1 + 0.04;
    Ca_val = floor ( ( Ca_val * 1000 - 0.04 ) / 0.099986667 ) * 0.099986667 + 0.04;
                
    ion [ m_NaF_pkj ] [ id ] = ( 2.0 * DT * inf_m_NaF ( v_val ) + ( 2.0 * tau_m_NaF ( v_val ) - DT ) * ion [ m_NaF_pkj ] [ id ] ) / ( 2.0 * tau_m_NaF ( v_val ) + DT );
    ion [ h_NaF_pkj ] [ id ] = ( 2.0 * DT * inf_h_NaF ( v_val ) + ( 2.0 * tau_h_NaF ( v_val ) - DT ) * ion [ h_NaF_pkj ] [ id ] ) / ( 2.0 * tau_h_NaF ( v_val ) + DT );
    ion [ m_NaP_pkj ] [ id ] = ( 2.0 * DT * inf_m_NaP ( v_val ) + ( 2.0 * tau_m_NaP ( v_val ) - DT ) * ion [ m_NaP_pkj ] [ id ] ) / ( 2.0 * tau_m_NaP ( v_val ) + DT );
    ion [ m_CaP_pkj ] [ id ] = ( 2.0 * DT * inf_m_CaP ( v_val ) + ( 2.0 * tau_m_CaP ( v_val ) - DT ) * ion [ m_CaP_pkj ] [ id ] ) / ( 2.0 * tau_m_CaP ( v_val ) + DT );
    ion [ h_CaP_pkj ] [ id ] = ( 2.0 * DT * inf_h_CaP ( v_val ) + ( 2.0 * tau_h_CaP ( v_val ) - DT ) * ion [ h_CaP_pkj ] [ id ] ) / ( 2.0 * tau_h_CaP ( v_val ) + DT );
    ion [ m_CaT_pkj ] [ id ] = ( 2.0 * DT * inf_m_CaT ( v_val ) + ( 2.0 * tau_m_CaT ( v_val ) - DT ) * ion [ m_CaT_pkj ] [ id ] ) / ( 2.0 * tau_m_CaT ( v_val ) + DT );
    ion [ h_CaT_pkj ] [ id ] = ( 2.0 * DT * inf_h_CaT ( v_val ) + ( 2.0 * tau_h_CaT ( v_val ) - DT ) * ion [ h_CaT_pkj ] [ id ] ) / ( 2.0 * tau_h_CaT ( v_val ) + DT );
    ion [ m_Kh1_pkj ] [ id ] = ( 2.0 * DT * inf_m_Kh1 ( v_val ) + ( 2.0 * tau_m_Kh1 ( v_val ) - DT ) * ion [ m_Kh1_pkj ] [ id ] ) / ( 2.0 * tau_m_Kh1 ( v_val ) + DT );
    ion [ m_Kh2_pkj ] [ id ] = ( 2.0 * DT * inf_m_Kh2 ( v_val ) + ( 2.0 * tau_m_Kh2 ( v_val ) - DT ) * ion [ m_Kh2_pkj ] [ id ] ) / ( 2.0 * tau_m_Kh2 ( v_val ) + DT );    
    ion [ m_Kdr_pkj ] [ id ] = ( 2.0 * DT * inf_m_Kdr ( v_val ) + ( 2.0 * tau_m_Kdr ( v_val ) - DT ) * ion [ m_Kdr_pkj ] [ id ] ) / ( 2.0 * tau_m_Kdr ( v_val ) + DT );  
    ion [ h_Kdr_pkj ] [ id ] = ( 2.0 * DT * inf_h_Kdr ( v_val ) + ( 2.0 * tau_h_Kdr ( v_val ) - DT ) * ion [ h_Kdr_pkj ] [ id ] ) / ( 2.0 * tau_h_Kdr ( v_val ) + DT );  
    ion [ m_KM_pkj  ] [ id ] = ( 2.0 * DT * inf_m_KM  ( v_val ) + ( 2.0 * tau_m_KM  ( v_val ) - DT ) * ion [ m_KM_pkj  ] [ id ] ) / ( 2.0 * tau_m_KM  ( v_val ) + DT );  
    ion [ m_KA_pkj  ] [ id ] = ( 2.0 * DT * inf_m_KA  ( v_val ) + ( 2.0 * tau_m_KA  ( v_val ) - DT ) * ion [ m_KA_pkj  ] [ id ] ) / ( 2.0 * tau_m_KA  ( v_val ) + DT );    
    ion [ h_KA_pkj  ] [ id ] = ( 2.0 * DT * inf_h_KA  ( v_val ) + ( 2.0 * tau_h_KA  ( v_val ) - DT ) * ion [ h_KA_pkj  ] [ id ] ) / ( 2.0 * tau_h_KA  ( v_val ) + DT );
    ion [ m_KC_pkj  ] [ id ] = ( 2.0 * DT * inf_m_KC  ( v_val ) + ( 2.0 * tau_m_KC  ( v_val ) - DT ) * ion [ m_KC_pkj  ] [ id ] ) / ( 2.0 * tau_m_KC  ( v_val ) + DT );    
    ion [ z_KC_pkj  ] [ id ] = ( 2.0 * DT * inf_z_KC  ( v_val, Ca_val ) + ( 2.0 * tau_z_KC  ( v_val ) - DT ) * ion [ z_KC_pkj  ] [ id ] ) / ( 2.0 * tau_z_KC  ( v_val ) + DT );
    ion [ m_K2_pkj  ] [ id ] = ( 2.0 * DT * inf_m_K2  ( v_val ) + ( 2.0 * tau_m_K2  ( v_val ) - DT ) * ion [ m_K2_pkj  ] [ id ] ) / ( 2.0 * tau_m_K2  ( v_val ) + DT );
    ion [ z_K2_pkj  ] [ id ] = ( 2.0 * DT * inf_z_K2  ( v_val, Ca_val ) + ( 2.0 * tau_z_K2  ( v_val ) - DT ) * ion [ z_K2_pkj  ] [ id ] ) / ( 2.0 * tau_z_K2  ( v_val ) + DT );
   
    d_pkj_solve -> vec [ cn_v_old ] [ id ] = elem [ v ] [ id ];
    
  }
}

__global__ void pkj_update_ion_RKC ( neuron_t *d_pkj, neuron_solve_t *d_pkj_solve, double *elem_vnew, const double DT )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
    
  if ( id < d_pkj -> nc)
  {
    double **elem = d_pkj -> elem;
    double **ion  = d_pkj -> ion;
    double v_val = 0.5 * ( elem_vnew [ id ] + elem [ v ] [ id ] );
    //double l_shell = d_pkj -> shell [ id ];// * elem [ area ] [ id ];
    double Ca_val = elem [ Ca ] [ id ];
    double I_Ca1 = 1e-3 * ( elem [ v ] [ id ] - d_pkj -> rev_ca2 [ id ] )  // I_Ca [mA/cm^2]
                        * ( d_pkj -> cond [ g_CaP_pkj ] [ id ] * ion [ m_CaP_pkj ] [ id ] * ion [ h_CaP_pkj ] [ id ]  
                          + d_pkj -> cond [ g_CaT_pkj ] [ id ] * ion [ m_CaT_pkj ] [ id ] * ion [ h_CaT_pkj ] [ id ] );
    ( I_Ca1 > 0.0 )? I_Ca1 = 0.0 : I_Ca1  = - I_Ca1 / ( 2.0 * F_PKJ * d_pkj -> shell [ id ] );
    double k1 = DT * ( I_Ca1 - B_Ca1_PKJ * ( Ca_val            - Ca1_0_PKJ ) ); //[mA*mol/cm^3*sec*A]=[M/sec] **[1mM = 1mol/m^3]**
    double k2 = DT * ( I_Ca1 - B_Ca1_PKJ * ( Ca_val + k1 / 2.0 - Ca1_0_PKJ ) );
    double k3 = DT * ( I_Ca1 - B_Ca1_PKJ * ( Ca_val + k2 / 2.0 - Ca1_0_PKJ ) );
    double k4 = DT * ( I_Ca1 - B_Ca1_PKJ * ( Ca_val + k3       - Ca1_0_PKJ ) );

    k1 = Ca_val + ( k1 + k2 * 2.0 + k3 * 2.0 + k4 ) / 6.0;
    elem [ Ca ] [ id ] = k1;
    Ca_val = 0.5 * ( Ca_val + k1 );
    
    //( elem [ compart ] [ id ] == PKJ_soma )?
    ( id % 1599 == 0 )?
      d_pkj -> rev_ca2 [ id ] = 12.5 * log ( Ca1OUT_PKJ / Ca1_0_PKJ ) :    
      d_pkj -> rev_ca2 [ id ] = 13.361624877 * log ( Ca1OUT_PKJ / Ca_val ); //310.15*8.3134/2/96485.309*1000
    //l_v_Ca = 0.5 * ( l_v_Ca + d_pkj -> rev_ca2 [ id ] );    

    //double cafloor = floor ( ( Ca_val * 1000 - 0.04 ) * 10.0 ) * 0.1 + 0.04;
    Ca_val = floor ( ( Ca_val * 1000 - 0.04 ) / 0.099986667 ) * 0.099986667 + 0.04;
        
    ion [ m_NaF_pkj ] [ id ] = ( 2.0 * DT * inf_m_NaF ( v_val ) + ( 2.0 * tau_m_NaF ( v_val ) - DT ) * ion [ m_NaF_pkj ] [ id ] ) / ( 2.0 * tau_m_NaF ( v_val ) + DT );
    ion [ h_NaF_pkj ] [ id ] = ( 2.0 * DT * inf_h_NaF ( v_val ) + ( 2.0 * tau_h_NaF ( v_val ) - DT ) * ion [ h_NaF_pkj ] [ id ] ) / ( 2.0 * tau_h_NaF ( v_val ) + DT );
    ion [ m_NaP_pkj ] [ id ] = ( 2.0 * DT * inf_m_NaP ( v_val ) + ( 2.0 * tau_m_NaP ( v_val ) - DT ) * ion [ m_NaP_pkj ] [ id ] ) / ( 2.0 * tau_m_NaP ( v_val ) + DT );
    ion [ m_CaP_pkj ] [ id ] = ( 2.0 * DT * inf_m_CaP ( v_val ) + ( 2.0 * tau_m_CaP ( v_val ) - DT ) * ion [ m_CaP_pkj ] [ id ] ) / ( 2.0 * tau_m_CaP ( v_val ) + DT );
    ion [ h_CaP_pkj ] [ id ] = ( 2.0 * DT * inf_h_CaP ( v_val ) + ( 2.0 * tau_h_CaP ( v_val ) - DT ) * ion [ h_CaP_pkj ] [ id ] ) / ( 2.0 * tau_h_CaP ( v_val ) + DT );
    ion [ m_CaT_pkj ] [ id ] = ( 2.0 * DT * inf_m_CaT ( v_val ) + ( 2.0 * tau_m_CaT ( v_val ) - DT ) * ion [ m_CaT_pkj ] [ id ] ) / ( 2.0 * tau_m_CaT ( v_val ) + DT );
    ion [ h_CaT_pkj ] [ id ] = ( 2.0 * DT * inf_h_CaT ( v_val ) + ( 2.0 * tau_h_CaT ( v_val ) - DT ) * ion [ h_CaT_pkj ] [ id ] ) / ( 2.0 * tau_h_CaT ( v_val ) + DT );
    ion [ m_Kh1_pkj ] [ id ] = ( 2.0 * DT * inf_m_Kh1 ( v_val ) + ( 2.0 * tau_m_Kh1 ( v_val ) - DT ) * ion [ m_Kh1_pkj ] [ id ] ) / ( 2.0 * tau_m_Kh1 ( v_val ) + DT );
    ion [ m_Kh2_pkj ] [ id ] = ( 2.0 * DT * inf_m_Kh2 ( v_val ) + ( 2.0 * tau_m_Kh2 ( v_val ) - DT ) * ion [ m_Kh2_pkj ] [ id ] ) / ( 2.0 * tau_m_Kh2 ( v_val ) + DT );    
    ion [ m_Kdr_pkj ] [ id ] = ( 2.0 * DT * inf_m_Kdr ( v_val ) + ( 2.0 * tau_m_Kdr ( v_val ) - DT ) * ion [ m_Kdr_pkj ] [ id ] ) / ( 2.0 * tau_m_Kdr ( v_val ) + DT );  
    ion [ h_Kdr_pkj ] [ id ] = ( 2.0 * DT * inf_h_Kdr ( v_val ) + ( 2.0 * tau_h_Kdr ( v_val ) - DT ) * ion [ h_Kdr_pkj ] [ id ] ) / ( 2.0 * tau_h_Kdr ( v_val ) + DT );  
    ion [ m_KM_pkj  ] [ id ] = ( 2.0 * DT * inf_m_KM  ( v_val ) + ( 2.0 * tau_m_KM  ( v_val ) - DT ) * ion [ m_KM_pkj  ] [ id ] ) / ( 2.0 * tau_m_KM  ( v_val ) + DT );  
    ion [ m_KA_pkj  ] [ id ] = ( 2.0 * DT * inf_m_KA  ( v_val ) + ( 2.0 * tau_m_KA  ( v_val ) - DT ) * ion [ m_KA_pkj  ] [ id ] ) / ( 2.0 * tau_m_KA  ( v_val ) + DT );    
    ion [ h_KA_pkj  ] [ id ] = ( 2.0 * DT * inf_h_KA  ( v_val ) + ( 2.0 * tau_h_KA  ( v_val ) - DT ) * ion [ h_KA_pkj  ] [ id ] ) / ( 2.0 * tau_h_KA  ( v_val ) + DT );
    ion [ m_KC_pkj  ] [ id ] = ( 2.0 * DT * inf_m_KC  ( v_val ) + ( 2.0 * tau_m_KC  ( v_val ) - DT ) * ion [ m_KC_pkj  ] [ id ] ) / ( 2.0 * tau_m_KC  ( v_val ) + DT );    
    ion [ z_KC_pkj  ] [ id ] = ( 2.0 * DT * inf_z_KC  ( v_val, Ca_val ) + ( 2.0 * tau_z_KC  ( v_val ) - DT ) * ion [ z_KC_pkj  ] [ id ] ) / ( 2.0 * tau_z_KC  ( v_val ) + DT );
    ion [ m_K2_pkj  ] [ id ] = ( 2.0 * DT * inf_m_K2  ( v_val ) + ( 2.0 * tau_m_K2  ( v_val ) - DT ) * ion [ m_K2_pkj  ] [ id ] ) / ( 2.0 * tau_m_K2  ( v_val ) + DT );
    ion [ z_K2_pkj  ] [ id ] = ( 2.0 * DT * inf_z_K2  ( v_val, Ca_val ) + ( 2.0 * tau_z_K2  ( v_val ) - DT ) * ion [ z_K2_pkj  ] [ id ] ) / ( 2.0 * tau_z_K2  ( v_val ) + DT );
 
    elem [ v ] [ id ] = elem_vnew [ id ];
  }
}

__host__ void pkj_initialize_ion ( neuron_t *pkj )
{
  double **elem = pkj -> elem;
  double **ion  = pkj -> ion;
  double init_v_rand = 0.0;
  for ( int i = 0; i < pkj -> nc; i++) 
  {
    if ( i % PKJ_COMP == 0 ) { init_v_rand = ( ( ( double ) rand ( ) + 1.0 ) / ( ( double ) RAND_MAX + 2.0 ) ) - 0.5; }

    elem [ v     ] [ i ] = V_INIT_PKJ + 2.0 * init_v_rand;
    elem [ Ca    ] [ i ] = Ca1_0_PKJ;
    
    pkj -> rev_ca2 [ i ] = V_Ca_PKJ;
    elem [ i_ext ] [ i ] = 0.0;
    double v_val = elem [ v  ] [ i ];
    double ca_val = elem [ Ca ] [ i ];
    ion [ m_NaF_pkj ] [ i ] = inf_m_NaF( v_val );
    ion [ h_NaF_pkj ] [ i ] = inf_h_NaF( v_val );
    ion [ m_NaP_pkj ] [ i ] = inf_m_NaP( v_val );
    ion [ m_CaP_pkj ] [ i ] = inf_m_CaP( v_val );
    ion [ h_CaP_pkj ] [ i ] = inf_h_CaP( v_val );
    ion [ m_CaT_pkj ] [ i ] = inf_m_CaT( v_val );
    ion [ h_CaT_pkj ] [ i ] = inf_h_CaT( v_val );
    ion [ m_Kh1_pkj ] [ i ] = inf_m_Kh1( v_val );
    ion [ m_Kh2_pkj ] [ i ] = inf_m_Kh2( v_val );
    ion [ m_Kdr_pkj ] [ i ] = inf_m_Kdr( v_val );
    ion [ h_Kdr_pkj ] [ i ] = inf_h_Kdr( v_val );
    ion [ m_KM_pkj  ] [ i ] = inf_m_KM( v_val );
    ion [ m_KA_pkj  ] [ i ] = inf_m_KA( v_val );
    ion [ h_KA_pkj  ] [ i ] = inf_h_KA( v_val );
    ion [ m_KC_pkj  ] [ i ] = inf_m_KC( v_val );
    ion [ z_KC_pkj  ] [ i ] = inf_z_KC(v_val, ca_val );
    ion [ m_K2_pkj  ] [ i ] = inf_m_K2( v_val );
    ion [ z_K2_pkj  ] [ i ] = inf_z_K2(v_val, ca_val );
  }
}
