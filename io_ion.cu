#include "io_ion.cuh"

// Action variable rate functions
//  Forward(α_{param})
__host__ __device__ static double alpha_r(double v)  { return (1.7)/(1.0+exp((-(v-5.0))/13.9)); }
__host__ __device__ static double alpha_s(double x)  { return fmin(0.00002*x, 0.01); }                 // check
__host__ __device__ static double alpha_x(double v)  { return (0.13*(v+25.0))/(1.0-exp(-(v+25.0)/10.0)); } // check

// Action variable rate functions
//  Backward(β_{param})
__host__ __device__ static double beta_r(double v)   { return (0.02*(v+8.5))/(exp((v+8.5)/5.0)-1.0); }
__host__ __device__ static double beta_s()           { return 0.015; }                                 // check
__host__ __device__ static double beta_x(double v)   { return 1.69*exp(-0.0125*(v+35.0)); }

// Steady-state value
//  {param}
__host__ __device__ static double h_infty(double v ,int eval)  {
  double h_val = 1.0;
  if(eval==1) h_val = 1.0/(1.0+exp((-v-70.0)/(-5.8)));
  if(eval==2) h_val = 1.0/(1.0+exp((-v-60.0)/(-5.8)));
  return h_val;
}
__host__ __device__ static double k_infty(double v)  { return 1.0/(1.0+exp((-v-61.0)/4.2)); }      // check
__host__ __device__ static double l_infty(double v)  { return 1.0/(1.0+exp(( v+85.5)/( 8.5))); }   // check
__host__ __device__ static double m_infty(double v)  { return 1.0/(1.0+exp((-v-30.0)/( 5.5))); }   // check
__host__ __device__ static double n_infty(double v)  { return 1.0/(1.0+exp((-v -3.0)/(10.0))); }   // check
__host__ __device__ static double p_infty(double v)  { return 1.0/(1.0+exp((-v-51.0)/(-12.0))); }  // check
__host__ __device__ static double q_infty(double v)  { return 1.0/(1.0+exp((v+80.0)/4.0)); }       // fix?
__host__ __device__ static double r_infty(double v)  { return alpha_r(v)/(alpha_r(v)+beta_r(v)); } // check
__host__ __device__ static double s_infty(double x)  { return alpha_s(x)/(alpha_s(x)+beta_s()); }  // check
__host__ __device__ static double x_infty(double v)  { return alpha_x(v)/(alpha_x(v)+beta_x(v)); } // check

// Steady-state value τ_x
//  τ_{param}
__host__ __device__ static double tau_h(double v ,int eval)    {
  double h_val = 1.0;
  if(eval==1) h_val = 3.0*exp((-v-40.0)/33.0);
  if(eval==2) h_val = 1.5*exp((-v-40.0)/33.0);
  return h_val;
}                                     // check
__host__ __device__ static double tau_k(double v)    { return 1.0; }                                                         // check
__host__ __device__ static double tau_l(double v)    { return ((20.0*exp((v+160.0)/30.0))/(1.0+exp((v+84.0)/7.3)))+35.0; }   // check
__host__ __device__ static double tau_n(double v)    { return 47.0*exp((-(-v-50.0))/900.0)+5.0; }                            // check
/* matlab code: 1.3 somatic Potassium rectifier current
double tau_n(double v)    { return 5 + (  47 * exp( -(-50 - v) /  900)); }
*/
__host__ __device__ static double tau_p(double v)    { return tau_n(v); }                                                    // chcek
__host__ __device__ static double tau_q(double v)    { return 1.0/(exp(-0.086*v-14.6)+exp(0.07*v-1.87)); }                   // OK? , ION model, Nicolas Schweighofer, Kenji Doya and Mitsuo Kawato (1999)
__host__ __device__ static double tau_r(double v)    { return 5.0/(alpha_r(v)+beta_r(v)); }                                  // check (fix)
__host__ __device__ static double tau_s(double x)    { return 1.0/(alpha_s(x)+beta_s()); }                                   // check
__host__ __device__ static double tau_x(double v)    { return 1.0/(alpha_x(v)+beta_x(v)); }                                  // check

__global__ void io_update_ion_2nd ( neuron_t *d_io, neuron_solve_t *d_io_solve, const double DT )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  
  if ( id < d_io -> nc)
  {
    double **elem = d_io -> elem;
    double **ion  = d_io -> ion;
    double **cond = d_io -> cond;
    int    l_comp = elem [ compart ] [ id ];
    double v_val  = 1.5 * elem [ v ] [ id ] - 0.5 * d_io_solve -> vec [ cn_v_old ] [ id ];
    //double v_val  = 0.5 * elem [ v ] [ id ] + 0.5 * d_io_solve -> vec [ cn_v_old ] [ id ];
    double Ca_val = elem [ Ca ] [ id ];    
    
    double I_Ca1 = cond [ g_CaH_io ] [ id ] * ion [ r_CaH_io ] [ id ]
                  * ion [ r_CaH_io ] [ id ] * ( elem [ v ] [ id ] - V_Ca_IO ); // I_Ca [muA/cm^2]
    double k1 = DT * ( - I_Ca1 * CA_PH_IO - BETA_CA_IO * ( Ca_val            ) ); //[muA*mol/cm^3*sec*A]=[mM/sec] **[1mM = 1mol/m^3]**
    double k2 = DT * ( - I_Ca1 * CA_PH_IO - BETA_CA_IO * ( Ca_val + k1 / 2.0 ) );
    double k3 = DT * ( - I_Ca1 * CA_PH_IO - BETA_CA_IO * ( Ca_val + k2 / 2.0 ) );
    double k4 = DT * ( - I_Ca1 * CA_PH_IO - BETA_CA_IO * ( Ca_val + k3       ) );
    elem [ Ca ] [ id ] += ( k1 + k2 * 2.0 + k3 * 2.0 + k4 ) / 6.0;
    Ca_val = ( Ca_val + elem [ Ca ] [ id ] ) / 2.0;

    ion [ k_CaL_io ] [ id ] = ( 2.0 * DT * k_infty ( v_val  ) + ( 2.0 * tau_k ( v_val  ) - DT ) * ion [ k_CaL_io ] [ id ] ) / ( 2.0 * tau_k ( v_val  ) + DT );
    ion [ l_CaL_io ] [ id ] = ( 2.0 * DT * l_infty ( v_val  ) + ( 2.0 * tau_l ( v_val  ) - DT ) * ion [ l_CaL_io ] [ id ] ) / ( 2.0 * tau_l ( v_val  ) + DT );
    ion [ n_Kdr_io ] [ id ] = ( 2.0 * DT * n_infty ( v_val  ) + ( 2.0 * tau_n ( v_val  ) - DT ) * ion [ n_Kdr_io ] [ id ] ) / ( 2.0 * tau_n ( v_val  ) + DT );
    ion [ p_Kdr_io ] [ id ] = ( 2.0 * DT * p_infty ( v_val  ) + ( 2.0 * tau_p ( v_val  ) - DT ) * ion [ p_Kdr_io ] [ id ] ) / ( 2.0 * tau_p ( v_val  ) + DT );
    ion [ x_K_io   ] [ id ] = ( 2.0 * DT * x_infty ( v_val  ) + ( 2.0 * tau_x ( v_val  ) - DT ) * ion [ x_K_io   ] [ id ] ) / ( 2.0 * tau_x ( v_val  ) + DT );
    ion [ r_CaH_io ] [ id ] = ( 2.0 * DT * r_infty ( v_val  ) + ( 2.0 * tau_r ( v_val  ) - DT ) * ion [ r_CaH_io ] [ id ] ) / ( 2.0 * tau_r ( v_val  ) + DT );
    ion [ q_H_io   ] [ id ] = ( 2.0 * DT * q_infty ( v_val  ) + ( 2.0 * tau_q ( v_val  ) - DT ) * ion [ q_H_io   ] [ id ] ) / ( 2.0 * tau_q ( v_val  ) + DT );
    ion [ s_KCa_io ] [ id ] = ( 2.0 * DT * s_infty ( Ca_val ) + ( 2.0 * tau_s ( Ca_val ) - DT ) * ion [ s_KCa_io ] [ id ] ) / ( 2.0 * tau_s ( Ca_val ) + DT );
    ion [ m_Na_io  ] [ id ] = m_infty ( v_val ); //( 2.0 * DT * m_infty ( v_val  ) + ( 2.0 * tau_m ( v_val  ) - DT ) * ion [ m_Na_io  ] [ id ] ) / ( 2.0 * tau_m ( v_val  ) + DT );
    ion [ h_Na_io  ] [ id ] = ( 2.0 * DT * h_infty ( v_val, l_comp ) + ( 2.0 * tau_h ( v_val, l_comp) - DT ) * ion [ h_Na_io  ] [ id ] ) / ( 2.0 * tau_h ( v_val, l_comp ) + DT );

    d_io_solve -> vec [ cn_v_old ] [ id ] = elem [ v ] [ id ] ;
  }
}

__global__ void io_update_ion_RKC ( neuron_t *d_io, neuron_solve_t *d_io_solve, double *vnew, const double DT )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  
  if ( id < d_io -> nc)
  {
    double **elem = d_io -> elem;
    double **ion  = d_io -> ion;
    double **cond = d_io -> cond;
    int    l_comp = elem [ compart ] [ id ];
    double v_val = 0.5 * ( vnew [ id ] + elem [ v ] [ id ] );
    double Ca_val = elem [ Ca ] [ id ];
 
    double I_Ca1 = cond [ g_CaH_io ] [ id ] * ion [ r_CaH_io ] [ id ]
                  * ion [ r_CaH_io ] [ id ] * ( elem [ v ] [ id ] - V_Ca_IO ); // I_Ca [muA/cm^2]
    double k1 = DT * ( - I_Ca1 * CA_PH_IO - BETA_CA_IO * ( Ca_val            ) ); //[muA*mol/cm^3*sec*A]=[mM/sec] **[1mM = 1mol/m^3]**
    double k2 = DT * ( - I_Ca1 * CA_PH_IO - BETA_CA_IO * ( Ca_val + k1 / 2.0 ) );
    double k3 = DT * ( - I_Ca1 * CA_PH_IO - BETA_CA_IO * ( Ca_val + k2 / 2.0 ) );
    double k4 = DT * ( - I_Ca1 * CA_PH_IO - BETA_CA_IO * ( Ca_val + k3       ) );
    elem [ Ca ] [ id ] += ( k1 + k2 * 2.0 + k3 * 2.0 + k4 ) / 6.0;
    Ca_val = ( Ca_val + elem [ Ca ] [ id ] ) / 2.0;

    ion [ k_CaL_io ] [ id ] = ( 2.0 * DT * k_infty ( v_val  ) + ( 2.0 * tau_k ( v_val  ) - DT ) * ion [ k_CaL_io ] [ id ] ) / ( 2.0 * tau_k ( v_val  ) + DT );
    ion [ l_CaL_io ] [ id ] = ( 2.0 * DT * l_infty ( v_val  ) + ( 2.0 * tau_l ( v_val  ) - DT ) * ion [ l_CaL_io ] [ id ] ) / ( 2.0 * tau_l ( v_val  ) + DT );
    ion [ n_Kdr_io ] [ id ] = ( 2.0 * DT * n_infty ( v_val  ) + ( 2.0 * tau_n ( v_val  ) - DT ) * ion [ n_Kdr_io ] [ id ] ) / ( 2.0 * tau_n ( v_val  ) + DT );
    ion [ p_Kdr_io ] [ id ] = ( 2.0 * DT * p_infty ( v_val  ) + ( 2.0 * tau_p ( v_val  ) - DT ) * ion [ p_Kdr_io ] [ id ] ) / ( 2.0 * tau_p ( v_val  ) + DT );
    ion [ x_K_io   ] [ id ] = ( 2.0 * DT * x_infty ( v_val  ) + ( 2.0 * tau_x ( v_val  ) - DT ) * ion [ x_K_io   ] [ id ] ) / ( 2.0 * tau_x ( v_val  ) + DT );
    ion [ r_CaH_io ] [ id ] = ( 2.0 * DT * r_infty ( v_val  ) + ( 2.0 * tau_r ( v_val  ) - DT ) * ion [ r_CaH_io ] [ id ] ) / ( 2.0 * tau_r ( v_val  ) + DT );
    ion [ q_H_io   ] [ id ] = ( 2.0 * DT * q_infty ( v_val  ) + ( 2.0 * tau_q ( v_val  ) - DT ) * ion [ q_H_io   ] [ id ] ) / ( 2.0 * tau_q ( v_val  ) + DT );
    ion [ s_KCa_io ] [ id ] = ( 2.0 * DT * s_infty ( Ca_val ) + ( 2.0 * tau_s ( Ca_val ) - DT ) * ion [ s_KCa_io ] [ id ] ) / ( 2.0 * tau_s ( Ca_val ) + DT );
    ion [ m_Na_io  ] [ id ] = m_infty ( v_val ); //( 2.0 * DT * m_infty ( v_val  ) + ( 2.0 * tau_m ( v_val  ) - DT ) * ion [ m_Na_io  ] [ id ] ) / ( 2.0 * tau_m ( v_val  ) + DT );
    ion [ h_Na_io  ] [ id ] = ( 2.0 * DT * h_infty ( v_val, l_comp ) + ( 2.0 * tau_h ( v_val, l_comp) - DT ) * ion [ h_Na_io  ] [ id ] ) / ( 2.0 * tau_h ( v_val, l_comp ) + DT );

    elem [ v ] [ id ] = vnew [ id ];
  }
}

__global__ void io_update_ion ( neuron_t *d_io, neuron_solve_t *d_io_solve, const double DT )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  
  if ( id < d_io -> nc)
  {
    double **elem = d_io -> elem;
    double **ion  = d_io -> ion;
    double **cond = d_io -> cond;
    int    l_comp = elem [ compart ] [ id ];
    double v_val  = elem [ v ] [ id ];
    double Ca_val = elem [ Ca ] [ id ];

    ion [ k_CaL_io ] [ id ] = k_infty ( v_val  ) + ( ion [ k_CaL_io ] [ id ] - k_infty ( v_val  ) ) * exp ( - DT / tau_k ( v_val  ) );
    ion [ l_CaL_io ] [ id ] = l_infty ( v_val  ) + ( ion [ l_CaL_io ] [ id ] - l_infty ( v_val  ) ) * exp ( - DT / tau_l ( v_val  ) );
    ion [ n_Kdr_io ] [ id ] = n_infty ( v_val  ) + ( ion [ n_Kdr_io ] [ id ] - n_infty ( v_val  ) ) * exp ( - DT / tau_n ( v_val  ) );
    ion [ p_Kdr_io ] [ id ] = p_infty ( v_val  ) + ( ion [ p_Kdr_io ] [ id ] - p_infty ( v_val  ) ) * exp ( - DT / tau_p ( v_val  ) );
    ion [ x_K_io   ] [ id ] = x_infty ( v_val  ) + ( ion [ x_K_io   ] [ id ] - x_infty ( v_val  ) ) * exp ( - DT / tau_x ( v_val  ) );
    ion [ r_CaH_io ] [ id ] = r_infty ( v_val  ) + ( ion [ r_CaH_io ] [ id ] - r_infty ( v_val  ) ) * exp ( - DT / tau_r ( v_val  ) );
    ion [ q_H_io   ] [ id ] = q_infty ( v_val  ) + ( ion [ q_H_io   ] [ id ] - q_infty ( v_val  ) ) * exp ( - DT / tau_q ( v_val  ) );
    ion [ s_KCa_io ] [ id ] = s_infty ( Ca_val ) + ( ion [ s_KCa_io ] [ id ] - s_infty ( Ca_val ) ) * exp ( - DT / tau_s ( Ca_val ) );
    ion [ m_Na_io  ] [ id ] = m_infty ( v_val ); // = m_infty ( v_val  ) + ( ion [ m_Na_io  ] [ id ] - m_infty ( v_val  ) ) * exp ( - DT / tau_m ( v_val  ) );
    ion [ h_Na_io  ] [ id ] = h_infty ( v_val, l_comp ) + ( ion [ h_Na_io  ] [ id ] - h_infty ( v_val, l_comp ) ) * exp ( - DT / tau_h ( v_val, l_comp ) );
                 
    double I_Ca1 = cond [ g_CaH_io ] [ id ] * ion [ r_CaH_io ] [ id ]
                  * ion [ r_CaH_io ] [ id ] * ( elem [ v ] [ id ] - V_Ca_IO ); // I_Ca [muA/cm^2]
    double k1 = DT * ( - I_Ca1 * CA_PH_IO - BETA_CA_IO * ( Ca_val            ) ); //[muA*mol/cm^3*sec*A]=[mM/sec] **[1mM = 1mol/m^3]**
    double k2 = DT * ( - I_Ca1 * CA_PH_IO - BETA_CA_IO * ( Ca_val + k1 / 2.0 ) );
    double k3 = DT * ( - I_Ca1 * CA_PH_IO - BETA_CA_IO * ( Ca_val + k2 / 2.0 ) );
    double k4 = DT * ( - I_Ca1 * CA_PH_IO - BETA_CA_IO * ( Ca_val + k3       ) );
    elem [ Ca ] [ id ] += ( k1 + k2 * 2.0 + k3 * 2.0 + k4 ) / 6.0;
  }
}


__host__ void io_initialize_ion ( neuron_t *io )
{
  double **elem = io -> elem;
  double **ion  = io -> ion;
  double **cond = io -> cond;
  double init_v_rand = 0.0;
  for ( int i = 0; i < io -> nc; i++) {
    
    if ( i % IO_COMP == 0 )
      init_v_rand = ( ( double ) rand ( ) / RAND_MAX ) - 0.5;

    elem [ v     ] [ i ] = V_INIT_IO + 10.0 * init_v_rand;
    elem [ i_ext ] [ i ] = 0.0;    
    double v_val  =  elem [ v  ] [ i ];
    int    l_comp = elem [ compart ] [ i ];

    ion [ k_CaL_io ] [ i ] = k_infty ( v_val  );
    ion [ l_CaL_io ] [ i ] = l_infty ( v_val  );
    ion [ n_Kdr_io ] [ i ] = n_infty ( v_val  );
    ion [ p_Kdr_io ] [ i ] = p_infty ( v_val  );
    ion [ x_K_io   ] [ i ] = x_infty ( v_val  );
    ion [ r_CaH_io ] [ i ] = r_infty ( v_val  );
    ion [ q_H_io   ] [ i ] = q_infty ( v_val  );

    double I_Ca1 = cond [ g_CaH_io ] [ i ] * ion [ r_CaH_io ] [ i ] * ion [ r_CaH_io ] [ i ] * ( v_val - V_Ca_IO );
    elem [ Ca    ] [ i ] = -I_Ca1 * CA_PH_IO / BETA_CA_IO;

    ion [ s_KCa_io ] [ i ] = s_infty ( elem [ Ca ] [ i ] );
    ion [ m_Na_io  ] [ i ] = m_infty ( v_val  );
    ion [ h_Na_io  ] [ i ] = h_infty ( v_val, l_comp );                 
  }
}
