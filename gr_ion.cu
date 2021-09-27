#include "gr_ion.cuh"

__host__ __device__ static double alpha_n_KV ( const double v ) {
  return ( fabs ( v + 25.0 ) > 1.0e-6 ) ? -0.017 * ((v + 25.0) / (exp(-(v + 25.0) / 10.0) - 1.0)) : -0.17 * (1.0 + (v + 25.0) / 20.0);
}
__host__ __device__ static double beta_n_KV(const double v) { return 0.2223 * exp(-(v + 35.0) / 80.0); }
__host__ __device__ static double inf_n_KV(const double v) { return (alpha_n_KV(v) / (alpha_n_KV(v) + beta_n_KV(v))); }
__host__ __device__ static double tau_n_KV(const double v) { return (1.0 / (alpha_n_KV(v) + beta_n_KV(v))); }

__host__ __device__ static double alpha_a_KA(const double v) { return 0.771 / (1.0 + exp(-(v + 9.17203) / 23.32708)); }
__host__ __device__ static double beta_a_KA(const double v) { return 0.1567 / exp((v + 18.27914) / 19.47175); }
__host__ __device__ static double inf_a_KA(const double v) { return 1.0 / (1.0 + exp(-(v + 38.0) / 17.0)); }
__host__ __device__ static double tau_a_KA(const double v) { return (1.0 / (alpha_a_KA(v) + beta_a_KA(v))); }

__host__ __device__ static double alpha_b_KA(const double v) { return 0.0348 / (1.0 + exp((v + 111.33208) / 12.8433)); }
__host__ __device__ static double beta_b_KA(const double v) { return 0.0327 / (1.0 + exp(-(v + 49.9537) / 8.90123)); }
__host__ __device__ static double inf_b_KA(const double v) { return 1.0 / (1.0 + exp((v + 78.8) / 8.4)); }
__host__ __device__ static double tau_b_KA(const double v) { return (1.0 / (alpha_b_KA(v) + beta_b_KA(v))); }

__host__ __device__ static double alpha_ir_KIR(const double v) { return 0.2302 * exp(-(v + 83.94) / 24.3902); }
__host__ __device__ static double beta_ir_KIR(const double v) { return 0.2943 * exp((v + 83.94) / 35.714); }
__host__ __device__ static double inf_ir_KIR(const double v) { return (alpha_ir_KIR(v) / (alpha_ir_KIR(v) + beta_ir_KIR(v))); }
__host__ __device__ static double tau_ir_KIR(const double v) { return (1.0 / (alpha_ir_KIR(v) + beta_ir_KIR(v))); }

__host__ __device__ static double alpha_s_KM(const double v) { return 0.0046 * exp((v + 30.0) / 40.0); }
__host__ __device__ static double beta_s_KM(const double v) { return 0.0046 * exp(-(v + 30.0) / 20.0); }
__host__ __device__ static double inf_s_KM(const double v) { return 1.0 / (1.0 + exp(-(v + 35.0) / 6.0)); }
__host__ __device__ static double tau_s_KM(const double v) { return (1.0 / (alpha_s_KM(v) + beta_s_KM(v))); }

__host__ __device__ static double alpha_c_KCa(const double v, const double ca) { return 1.4433 / (1.0 + (1.5e-3 * exp(-v / 11.765)) / ca); }
__host__ __device__ static double beta_c_KCa(const double v, const double ca) { return 0.8660 / (1.0 + ca / (1.5e-4 * exp(-v / 11.765))); }
__host__ __device__ static double inf_c_KCa(const double v, const double ca) { return alpha_c_KCa(v, ca) / (alpha_c_KCa(v, ca) + beta_c_KCa(v, ca)); }
__host__ __device__ static double tau_c_KCa(const double v, const double ca) { return (1.0 / (alpha_c_KCa(v, ca) + beta_c_KCa(v, ca))); }

__host__ __device__ static double alpha_ch_Ca(const double v) { return 0.0856 * exp((v + 29.6) / 15.873); }
__host__ __device__ static double beta_ch_Ca(const double v) { return 0.1437 * exp(-(v + 18.66) / 25.641); }
__host__ __device__ static double inf_ch_Ca(const double v) { return (alpha_ch_Ca(v) / (alpha_ch_Ca(v) + beta_ch_Ca(v))); }
__host__ __device__ static double tau_ch_Ca(const double v) { return (1.0 / (alpha_ch_Ca(v) + beta_ch_Ca(v))); }

__host__ __device__ static double alpha_ci_Ca(const double v) { return 0.0023 * exp(-(v + 48.0) / 18.183); }
__host__ __device__ static double beta_ci_Ca(const double v) { return 0.0025 * exp((v + 48.0) / 83.33); }
__host__ __device__ static double inf_ci_Ca(const double v) { return (alpha_ci_Ca(v) / (alpha_ci_Ca(v) + beta_ci_Ca(v))); }
__host__ __device__ static double tau_ci_Ca(const double v) { return (1.0 / (alpha_ci_Ca(v) + beta_ci_Ca(v))); }

__global__ void gr_Na_update_2order ( const int nc, const double *l_v_new, const double *l_v, const double DT, const double *compart,
  double *o,  double *c1, double *c2, double *c3, double *c4, double *c5,
  double *i1, double *i2, double *i3, double *i4, double *i5, double *i6 )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;  
  int i_comp =  ( int ) ( compart [ id ] );
  if ( ( id < nc ) && ( i_comp >= GR_hill * 1.0 ) )
  {
    // Debug
    if ( i_comp < GR_hill || GR_pf < i_comp  ) { printf ( "GR Na kinetic error\n" ); return; }

    //GR Na channel parameters
    const double Na_n1 [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 5.422, 5.422, 5.422, 5.422 };
    const double Na_n2 [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 3.279, 3.279, 3.279, 3.279 };
    const double Na_n3 [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.83 , 1.83 , 1.83 , 1.83 };
    const double Na_n4 [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 0.738, 0.738, 0.738, 0.738 };

    const double Na_Aalpha [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 8.85,   8.85,   8.85,   8.85 };
    const double Na_Abeta  [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 0.0318, 0.0318, 0.0318, 0.0318 };
    const double Na_Vshift [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, -22.0, -22.0, -25.0, -25.0 };

    const double Na_Con  [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 0.003464, 0.003464, 0.09353,  0.09353 };
    const double Na_Coff [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 0.4157,   0.4157,   0.04676,  0.04676 };
    const double Na_Oon  [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.8186,   1.8186,  2.5981, 2.5981 };
    const double Na_Ooff [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 0.006928, 0.006928, 0.000173,  0.000173};

    //const double Na_Vshift [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, -22.0, -22.0, -22.0, -22.0 };

    //const double Na_Con  [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 0.003464, 0.003464, 0.003464, 0.003464 };
    //const double Na_Coff [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 0.4157,   0.4157,   0.4157,  0.4157 };
    //const double Na_Oon  [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.8186,   1.8186,   1.8186,   1.8186 };
    //const double Na_Ooff [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 0.006928, 0.006928, 0.006928, 0.006928 };

    const double n1 = Na_n1 [ i_comp ];
    const double n2 = Na_n2 [ i_comp ];
    const double n3 = Na_n3 [ i_comp ];
    const double n4 = Na_n4 [ i_comp ];
    const double Con  = Na_Con  [i_comp ];
    const double Coff = Na_Coff [ i_comp ];
    const double Oon  = Na_Oon  [ i_comp ];
    const double Ooff = Na_Ooff [ i_comp ];
    double alpha = Na_Aalpha [ i_comp ] * exp (   ( l_v [ id ] - Na_Vshift [ i_comp ] ) / 10.0 );
    double beta  = Na_Abeta  [ i_comp ] * exp ( - ( l_v [ id ] - Na_Vshift [ i_comp ] ) / 10.0 );
    const double a = pow ( ( Na_Oon  [ i_comp ] / Na_Con  [ i_comp ] ), 0.25 );
    const double b = pow ( ( Na_Ooff [ i_comp ] / Na_Coff [ i_comp ] ), 0.25 );

    // create Matrix
    
    double vca_n [ 12 ] [ 12 ] = {
      { -(n1*alpha + Con - 2.0/DT ), n4*beta, 0.0, 0.0, 0.0, Coff, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      { n1*alpha, -(n4*beta + n2 * alpha + Con * a - 2.0/DT ), n3*beta, 0.0, 0.0, 0.0, Coff*b, 0.0, 0.0, 0.0, 0.0, 0.0  },
      { 0.0, n2*alpha, -(n3*beta + n3 * alpha + Con * a*a - 2.0/DT ), n2*beta, 0.0, 0.0, 0.0, Coff*b*b, 0.0, 0.0, 0.0, 0.0  },
      { 0.0, 0.0, n3*alpha, -(n2*beta + n4 * alpha + Con * a*a*a - 2.0/DT ), n1*beta, 0.0, 0.0, 0.0, Coff*b*b*b, 0.0, 0.0, 0.0  },
      { 0.0, 0.0, 0.0, n4*alpha, -(n1*beta + 259.8 + Con * pow(a,4.0) - 2.0/DT ), 0.0, 0.0, 0.0, 0.0, Coff*pow(b,4.0), 0.0, 69.3 },
      { Con, 0.0, 0.0, 0.0, 0.0, -(Coff + n1 * alpha*a - 2.0/DT ), n4*beta*b, 0.0, 0.0, 0.0, 0.0, 0.0 },
      { 0.0, Con*a, 0.0, 0.0, 0.0, n1*alpha*a, -(Coff*b + n4 * beta*b + n2 * alpha*a - 2.0/DT ), n3*beta*b, 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, Con*a*a, 0.0, 0.0, 0.0, n2*alpha*a, -(Coff*b*b + n3 * beta*b + n3 * alpha*a - 2.0/DT ), n2*beta*b, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, Con*a*a*a, 0.0, 0.0, 0.0, n3*alpha*a, -(Coff*b*b*b + n2 * beta*b + n4 * alpha*a - 2.0/DT ), n1*beta*b, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0, Con*pow(a,4.0), 0.0, 0.0, 0.0, n4*alpha*a, -(Coff*pow(b,4.0) + n1 * beta*b + 259.8 - 2.0/DT ), 69.3, 0.0  },
      { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 259.8, -(Ooff + 69.3 - 2.0/DT ), Oon },
      { 0.0, 0.0, 0.0, 0.0, 259.8, 0.0, 0.0, 0.0, 0.0, 0.0, Ooff, -(69.3 + Oon - 2.0/DT ) }
    };
    double vc [ 12 ] = { c1[ id ] , c2[ id ], c3[ id ], c4[ id ], c5[ id ],
                         i1[ id ], i2[ id ], i3[ id ], i4[ id ], i5[ id ], i6[ id ], o[ id ] };
    double vcb [ 12 ] = { };
    
    for ( int i = 0; i < 12; i++ )
    {
      for ( int j = 0; j < 12; j++ )
      {
        vcb [ i ] -= vc [ j ] * vca_n [ i ] [ j ];
      }
    }
    alpha = Na_Aalpha [ i_comp ] * exp (   ( l_v_new [ id ] - Na_Vshift [ i_comp ] ) / 10.0 );
    beta  = Na_Abeta  [ i_comp ] * exp ( - ( l_v_new [ id ] - Na_Vshift [ i_comp ] ) / 10.0 );

    double vca [ 12 ] [ 12 ] = {
      { -(n1*alpha + Con + 2.0/DT ), n4*beta, 0.0, 0.0, 0.0, Coff, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      { n1*alpha, -(n4*beta + n2 * alpha + Con * a + 2.0/DT ), n3*beta, 0.0, 0.0, 0.0, Coff*b, 0.0, 0.0, 0.0, 0.0, 0.0  },
      { 0.0, n2*alpha, -(n3*beta + n3 * alpha + Con * a*a + 2.0/DT ), n2*beta, 0.0, 0.0, 0.0, Coff*b*b, 0.0, 0.0, 0.0, 0.0  },
      { 0.0, 0.0, n3*alpha, -(n2*beta + n4 * alpha + Con * a*a*a + 2.0/DT ), n1*beta, 0.0, 0.0, 0.0, Coff*b*b*b, 0.0, 0.0, 0.0  },
      { 0.0, 0.0, 0.0, n4*alpha, -(n1*beta + 259.8 + Con * pow(a,4.0) + 2.0/DT ), 0.0, 0.0, 0.0, 0.0, Coff*pow(b,4.0), 0.0, 69.3 },
      { Con, 0.0, 0.0, 0.0, 0.0, -(Coff + n1 * alpha*a + 2.0/DT ), n4*beta*b, 0.0, 0.0, 0.0, 0.0, 0.0 },
      { 0.0, Con*a, 0.0, 0.0, 0.0, n1*alpha*a, -(Coff*b + n4 * beta*b + n2 * alpha*a + 2.0/DT ), n3*beta*b, 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, Con*a*a, 0.0, 0.0, 0.0, n2*alpha*a, -(Coff*b*b + n3 * beta*b + n3 * alpha*a + 2.0/DT ), n2*beta*b, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, Con*a*a*a, 0.0, 0.0, 0.0, n3*alpha*a, -(Coff*b*b*b + n2 * beta*b + n4 * alpha*a + 2.0/DT ), n1*beta*b, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0, Con*pow(a,4.0), 0.0, 0.0, 0.0, n4*alpha*a, -(Coff*pow(b,4.0) + n1 * beta*b + 259.8 + 2.0/DT ), 69.3, 0.0  },
      { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 259.8, -(Ooff + 69.3 + 2.0/DT ), Oon },
      { 0.0, 0.0, 0.0, 0.0, 259.8, 0.0, 0.0, 0.0, 0.0, 0.0, Ooff, -(69.3 + Oon + 2.0/DT ) }
    };

    //////////////////// Pivot selection //////////////////
    double temp;
    for ( int i = 0; i < 11; i++ ) 
    {
      int pivot = i;
      double p_max = fabs ( vca [ i ] [ i ] );
      for ( int j = i + 1; j < 12; j++ )
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
        for ( int j = i; j < 12; j++ )
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
      for ( int j = i + 1; j < 12; j++ ) 
      {
        double w = vca [ j ] [ i ] / vca [ i ] [ i ];
        vca [ j ] [ i ] = 0.0;
        // Multiply the ith line by -a[j][i]/a[i][i] and add it to the jth line
        for ( int k = i + 1; k < 12; k++ ) 
        {
           vca [ j ] [ k ] = vca [ j ] [ k ] - vca [ i ] [ k ] * w;
        }
        vcb [ j ] = vcb [ j ] - vcb [ i ] * w;
      }
    }
    //////////////////// Backward elimination //////////////////   
    for ( int i = 12 - 1; i >= 0; i-- )
    {
      for( int j = i + 1; j < 12; j++)
      {
         vcb [ i ] = vcb [ i ] - vca [ i ] [ j ] * vcb [ j ];
         vca [ i ] [ j ] = 0.0;
      }
      vcb [ i ] = vcb [ i ] / vca [ i ] [ i ];
      vca [ i ] [ i ] = 1.0;
   }
      
    
    c1 [ id ] = vcb [ 0 ];
    c2 [ id ] = vcb [ 1 ]; 
    c3 [ id ] = vcb [ 2 ];
    c4 [ id ] = vcb [ 3 ];
    c5 [ id ] = vcb [ 4 ];
    i1 [ id ] = vcb [ 5 ];
    i2 [ id ] = vcb [ 6 ];
    i3 [ id ] = vcb [ 7 ];
    i4 [ id ] = vcb [ 8 ];
    i5 [ id ] = vcb [ 9 ];
    i6 [ id ] = vcb [ 10 ];
    o  [ id ] = vcb [ 11 ];

    const double sum = c1 [ id ] + c2 [ id ] + c3 [ id ] + c4 [ id ] + c5 [ id ] + i1 [ id ] + i2 [ id ] + i3 [ id ] + i4 [ id ] + i5 [ id ] + i6 [ id ] + o [ id ];
    if ( ( sum < 0.99 ) || ( sum > 1.01 ) ) { printf ( "KAHP error %.15f\n", sum ); } // Debug
  }
}

__global__ void gr_Na_update ( const int nc, const double *l_v, const double DT, const double *compart,
  double *o,  double *c1, double *c2, double *c3, double *c4, double *c5,
  double *i1, double *i2, double *i3, double *i4, double *i5, double *i6 )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;  
  int i_comp =  ( int ) ( compart [ id ] );
  if ( ( id < nc ) && ( i_comp >= GR_hill * 1.0 ) )
  {
    // Debug
    if ( i_comp < GR_hill || GR_pf < i_comp  ) { printf ( "GR Na kinetic error\n" ); return; }

    //GR Na channel parameters
    const double Na_n1 [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 5.422, 5.422, 5.422, 5.422 };
    const double Na_n2 [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 3.279, 3.279, 3.279, 3.279 };
    const double Na_n3 [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.83 , 1.83 , 1.83 , 1.83 };
    const double Na_n4 [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 0.738, 0.738, 0.738, 0.738 };

    const double Na_Aalpha [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 8.85,   8.85,   8.85,   8.85 };
    const double Na_Abeta  [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 0.0318, 0.0318, 0.0318, 0.0318 };
    const double Na_Vshift [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, -22.0, -22.0, -25.0, -25.0 };

    const double Na_Con  [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 0.003464, 0.003464, 0.09353,  0.09353 };
    const double Na_Coff [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 0.4157,   0.4157,   0.04676,  0.04676 };
    const double Na_Oon  [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.8186,   1.8186,  2.5981, 2.5981 };
    const double Na_Ooff [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 0.006928, 0.006928, 0.000173,  0.000173};

    //const double Na_Vshift [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, -22.0, -22.0, -22.0, -22.0 };

    //const double Na_Con  [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 0.003464, 0.003464, 0.003464, 0.003464 };
    //const double Na_Coff [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 0.4157,   0.4157,   0.4157,  0.4157 };
    //const double Na_Oon  [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.8186,   1.8186,   1.8186,   1.8186 };
    //const double Na_Ooff [ ] = { 1.0, 1.0, 1.0, 1.0, 1.0, 0.006928, 0.006928, 0.006928, 0.006928 };

    const double n1 = Na_n1 [ i_comp ];
    const double n2 = Na_n2 [ i_comp ];
    const double n3 = Na_n3 [ i_comp ];
    const double n4 = Na_n4 [ i_comp ];
    const double Con  = Na_Con  [i_comp ];
    const double Coff = Na_Coff [ i_comp ];
    const double Oon  = Na_Oon  [ i_comp ];
    const double Ooff = Na_Ooff [ i_comp ];
    const double alpha = Na_Aalpha [ i_comp ] * exp (   ( l_v [ id ] - Na_Vshift [ i_comp ] ) / 10.0 );
    const double beta  = Na_Abeta  [ i_comp ] * exp ( - ( l_v [ id ] - Na_Vshift [ i_comp ] ) / 10.0 );
    const double a = pow ( ( Na_Oon  [ i_comp ] / Na_Con  [ i_comp ] ), 0.25 );
    const double b = pow ( ( Na_Ooff [ i_comp ] / Na_Coff [ i_comp ] ), 0.25 );

    // create Matrix
    
    double vca [ 12 ] [ 12 ] = {
      { -(n1*alpha + Con), n4*beta, 0.0, 0.0, 0.0, Coff, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      { n1*alpha, -(n4*beta + n2 * alpha + Con * a), n3*beta, 0.0, 0.0, 0.0, Coff*b, 0.0, 0.0, 0.0, 0.0, 0.0  },
      { 0.0, n2*alpha, -(n3*beta + n3 * alpha + Con * a*a), n2*beta, 0.0, 0.0, 0.0, Coff*b*b, 0.0, 0.0, 0.0, 0.0  },
      { 0.0, 0.0, n3*alpha, -(n2*beta + n4 * alpha + Con * a*a*a), n1*beta, 0.0, 0.0, 0.0, Coff*b*b*b, 0.0, 0.0, 0.0  },
      { 0.0, 0.0, 0.0, n4*alpha, -(n1*beta + 259.8 + Con * pow(a,4.0)), 0.0, 0.0, 0.0, 0.0, Coff*pow(b,4.0), 0.0, 69.3 },
      { Con, 0.0, 0.0, 0.0, 0.0, -(Coff + n1 * alpha*a), n4*beta*b, 0.0, 0.0, 0.0, 0.0, 0.0 },
      { 0.0, Con*a, 0.0, 0.0, 0.0, n1*alpha*a, -(Coff*b + n4 * beta*b + n2 * alpha*a), n3*beta*b, 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, Con*a*a, 0.0, 0.0, 0.0, n2*alpha*a, -(Coff*b*b + n3 * beta*b + n3 * alpha*a), n2*beta*b, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, Con*a*a*a, 0.0, 0.0, 0.0, n3*alpha*a, -(Coff*b*b*b + n2 * beta*b + n4 * alpha*a), n1*beta*b, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0, Con*pow(a,4.0), 0.0, 0.0, 0.0, n4*alpha*a, -(Coff*pow(b,4.0) + n1 * beta*b + 259.8), 69.3, 0.0  },
      { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 259.8, -(Ooff + 69.3), Oon },
      { 0.0, 0.0, 0.0, 0.0, 259.8, 0.0, 0.0, 0.0, 0.0, 0.0, Ooff, -(69.3 + Oon) }
    };
    double vcb [ 12 ] = { c1[ id ] , c2[ id ], c3[ id ], c4[ id ], c5[ id ],
                         i1[ id ], i2[ id ], i3[ id ], i4[ id ], i5[ id ], i6[ id ], o[ id ] };

    for ( int i = 0; i < 12; i++ )
      for ( int j = 0; j < 12; j++ ) 
        vca [ i ] [ j ] *= - DT;
    
    for ( int i = 0; i < 12; i++ ) { vca [ i ] [ i ] += 1.0; }
    
    //////////////////// Pivot selection //////////////////
    double temp;
    for ( int i = 0; i < 11; i++ ) 
    {
      int pivot = i;
      double p_max = fabs ( vca [ i ] [ i ] );
      for ( int j = i + 1; j < 12; j++ )
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
        for ( int j = i; j < 12; j++ )
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
      for ( int j = i + 1; j < 12; j++ ) 
      {
        double w = vca [ j ] [ i ] / vca [ i ] [ i ];
        vca [ j ] [ i ] = 0.0;
        // Multiply the ith line by -a[j][i]/a[i][i] and add it to the jth line
        for ( int k = i + 1; k < 12; k++ ) 
        {
           vca [ j ] [ k ] = vca [ j ] [ k ] - vca [ i ] [ k ] * w;
        }
        vcb [ j ] = vcb [ j ] - vcb [ i ] * w;
      }
    }
    //////////////////// Backward elimination //////////////////   
    for ( int i = 12 - 1; i >= 0; i-- )
    {
      for( int j = i + 1; j < 12; j++)
      {
         vcb [ i ] = vcb [ i ] - vca [ i ] [ j ] * vcb [ j ];
         vca [ i ] [ j ] = 0.0;
      }
      vcb [ i ] = vcb [ i ] / vca [ i ] [ i ];
      vca [ i ] [ i ] = 1.0;
   }
      
    
    c1 [ id ] = vcb [ 0 ];
    c2 [ id ] = vcb [ 1 ]; 
    c3 [ id ] = vcb [ 2 ];
    c4 [ id ] = vcb [ 3 ];
    c5 [ id ] = vcb [ 4 ];
    i1 [ id ] = vcb [ 5 ];
    i2 [ id ] = vcb [ 6 ];
    i3 [ id ] = vcb [ 7 ];
    i4 [ id ] = vcb [ 8 ];
    i5 [ id ] = vcb [ 9 ];
    i6 [ id ] = vcb [ 10 ];
    o  [ id ] = vcb [ 11 ];

    const double sum = c1 [ id ] + c2 [ id ] + c3 [ id ] + c4 [ id ] + c5 [ id ] + i1 [ id ] + i2 [ id ] + i3 [ id ] + i4 [ id ] + i5 [ id ] + i6 [ id ] + o [ id ];
    if ( ( sum < 0.999999 ) || ( sum > 1.000001 ) ) { printf ( "KAHP error %.15f\n", sum ); } // Debug
  }
}


__host__ __device__ double dmdt ( const double m, const double inf_m, const double tau_m  )
{
  return ( 1.0 / tau_m ) * ( - m + inf_m );
}

__global__ void gr_update_ion_RKC_RK4 ( neuron_t *d_gr, neuron_solve_t *d_gr_solve,  double *elem_v, const double DT )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  double **elem = d_gr -> elem;
  double **ion  = d_gr -> ion;
  double **cond = d_gr -> cond;
  
  if ( id < d_gr -> nc)
  {
    double v_val = elem [ v ] [ id ];
    double Ca_val = elem [ Ca ] [ id ];
    //Imp2
    //double l_ch_Ca = ion [ ch_Ca  ] [ id ];
    //double l_ci_Ca = ion [ ci_Ca  ] [ id ];
    
    double dndt1  = dmdt ( ion [ n_KV   ] [ id ],  inf_n_KV   ( v_val ), tau_n_KV   ( v_val ) );
    double dadt1  = dmdt ( ion [ a_KA   ] [ id ],  inf_a_KA   ( v_val ), tau_a_KA   ( v_val ) );
    double dbdt1  = dmdt ( ion [ b_KA   ] [ id ],  inf_b_KA   ( v_val ), tau_b_KA   ( v_val ) );
    double dirdt1 = dmdt ( ion [ ir_KIR ] [ id ],  inf_ir_KIR ( v_val ), tau_ir_KIR ( v_val ) );
    double dsdt1  = dmdt ( ion [ s_KM   ] [ id ],  inf_s_KM   ( v_val ), tau_s_KM   ( v_val ) );
    double dchdt1 = dmdt ( ion [ ch_Ca  ] [ id ],  inf_ch_Ca  ( v_val ), tau_ch_Ca  ( v_val ) );
    double dcidt1 = dmdt ( ion [ ci_Ca  ] [ id ],  inf_ci_Ca  ( v_val ), tau_ci_Ca  ( v_val ) );
    double dcdt1  = dmdt ( ion [ c_KCa  ] [ id ],  inf_c_KCa ( v_val, Ca_val ), tau_c_KCa ( v_val, Ca_val ) );
    
    double I_Ca1 = ( 1e-3 * cond [ g_Ca ] [ id ] / elem [ area ] [ id ] * ion [ ch_Ca ] [ id ] * ion [ ch_Ca ] [ id ] * ion [ ci_Ca ] [ id ] * ( v_val - V_Ca_GR ) ); //I_Ca [mA/cm^2]
    double dCadt1 = - I_Ca1 / ( 2.0 * F * SHELL1_D ) - B_Ca1 * ( Ca_val - Ca1_0 );
    
    double dndt2  = dmdt ( ion [ n_KV   ] [ id ] + DT * dndt1  / 2.0,  inf_n_KV   ( v_val ), tau_n_KV   ( v_val ) );
    double dadt2  = dmdt ( ion [ a_KA   ] [ id ] + DT * dadt1  / 2.0,  inf_a_KA   ( v_val ), tau_a_KA   ( v_val ) );
    double dbdt2  = dmdt ( ion [ b_KA   ] [ id ] + DT * dbdt1  / 2.0,  inf_b_KA   ( v_val ), tau_b_KA   ( v_val ) );
    double dirdt2 = dmdt ( ion [ ir_KIR ] [ id ] + DT * dirdt1 / 2.0,  inf_ir_KIR ( v_val ), tau_ir_KIR ( v_val ) );
    double dsdt2  = dmdt ( ion [ s_KM   ] [ id ] + DT * dsdt1  / 2.0,  inf_s_KM   ( v_val ), tau_s_KM   ( v_val ) );
    double dchdt2 = dmdt ( ion [ ch_Ca  ] [ id ] + DT * dchdt1 / 2.0,  inf_ch_Ca  ( v_val ), tau_ch_Ca  ( v_val ) );
    double dcidt2 = dmdt ( ion [ ci_Ca  ] [ id ] + DT * dcidt1 / 2.0,  inf_ci_Ca  ( v_val ), tau_ci_Ca  ( v_val ) );
    double dcdt2  = dmdt ( ion [ c_KCa  ] [ id ] + DT * dcdt1  / 2.0 ,  inf_c_KCa ( v_val, Ca_val + DT * dCadt1 / 2.0 ), tau_c_KCa ( v_val, Ca_val + DT * dCadt1 / 2.0 ) );
    //double dcdt2  = dmdt ( ion [ c_KCa  ] [ id ] + DT * dcdt1  / 2.0,  inf_c_KCa ( v_val, Ca_val ), tau_c_KCa ( v_val, Ca_val ) );
    
    double I_Ca2 = ( 1e-3 * cond [ g_Ca ] [ id ] / elem [ area ] [ id ] * ( ion [ ch_Ca ] [ id ] * DT * dchdt1 / 2.0 ) * ( ion [ ch_Ca ] [ id ] * DT * dchdt1 / 2.0 ) * ( ion [ ci_Ca ] [ id ] * DT * dcidt1 / 2.0 ) * ( v_val - V_Ca_GR ) ); //I_Ca [mA/cm^2]
    double dCadt2 = - I_Ca2 / ( 2.0 * F * SHELL1_D ) - B_Ca1 * ( Ca_val + DT * dCadt1 / 2.0 - Ca1_0 );
    
    double dndt3  = dmdt ( ion [ n_KV   ] [ id ] + DT * dndt2  / 2.0,  inf_n_KV   ( v_val ), tau_n_KV   ( v_val ) );
    double dadt3  = dmdt ( ion [ a_KA   ] [ id ] + DT * dadt2  / 2.0,  inf_a_KA   ( v_val ), tau_a_KA   ( v_val ) );
    double dbdt3  = dmdt ( ion [ b_KA   ] [ id ] + DT * dbdt2  / 2.0,  inf_b_KA   ( v_val ), tau_b_KA   ( v_val ) );
    double dirdt3 = dmdt ( ion [ ir_KIR ] [ id ] + DT * dirdt2 / 2.0,  inf_ir_KIR ( v_val ), tau_ir_KIR ( v_val ) );
    double dsdt3  = dmdt ( ion [ s_KM   ] [ id ] + DT * dsdt2  / 2.0,  inf_s_KM   ( v_val ), tau_s_KM   ( v_val ) );
    double dchdt3 = dmdt ( ion [ ch_Ca  ] [ id ] + DT * dchdt2 / 2.0,  inf_ch_Ca  ( v_val ), tau_ch_Ca  ( v_val ) );
    double dcidt3 = dmdt ( ion [ ci_Ca  ] [ id ] + DT * dcidt2 / 2.0,  inf_ci_Ca  ( v_val ), tau_ci_Ca  ( v_val ) );
    double dcdt3  = dmdt ( ion [ c_KCa  ] [ id ] + DT * dcdt2  / 2.0 ,  inf_c_KCa ( v_val, Ca_val + DT * dCadt2 / 2.0 ), tau_c_KCa ( v_val, Ca_val + DT * dCadt2 / 2.0 ) );
    //double dcdt3  = dmdt ( ion [ c_KCa  ] [ id ] + DT * dcdt2  / 2.0,  inf_c_KCa ( v_val, Ca_val ), tau_c_KCa ( v_val, Ca_val ) );
    
    double I_Ca3 = ( 1e-3 * cond [ g_Ca ] [ id ] / elem [ area ] [ id ] * ( ion [ ch_Ca ] [ id ] * DT * dchdt2 / 2.0 ) * ( ion [ ch_Ca ] [ id ] * DT * dchdt2 / 2.0 ) * ( ion [ ci_Ca ] [ id ] * DT * dcidt2 / 2.0 ) * ( v_val - V_Ca_GR ) ); //I_Ca [mA/cm^2]
    double dCadt3 = - I_Ca3 / ( 2.0 * F * SHELL1_D ) - B_Ca1 * ( Ca_val + DT * dCadt2 / 2.0 - Ca1_0 );
    
    double dndt4  = dmdt ( ion [ n_KV   ] [ id ] + DT * dndt3 ,  inf_n_KV   ( v_val ), tau_n_KV   ( v_val ) );
    double dadt4  = dmdt ( ion [ a_KA   ] [ id ] + DT * dadt3 ,  inf_a_KA   ( v_val ), tau_a_KA   ( v_val ) );
    double dbdt4  = dmdt ( ion [ b_KA   ] [ id ] + DT * dbdt3 ,  inf_b_KA   ( v_val ), tau_b_KA   ( v_val ) );
    double dirdt4 = dmdt ( ion [ ir_KIR ] [ id ] + DT * dirdt3,  inf_ir_KIR ( v_val ), tau_ir_KIR ( v_val ) );
    double dsdt4  = dmdt ( ion [ s_KM   ] [ id ] + DT * dsdt3 ,  inf_s_KM   ( v_val ), tau_s_KM   ( v_val ) );
    double dchdt4 = dmdt ( ion [ ch_Ca  ] [ id ] + DT * dchdt3,  inf_ch_Ca  ( v_val ), tau_ch_Ca  ( v_val ) );
    double dcidt4 = dmdt ( ion [ ci_Ca  ] [ id ] + DT * dcidt3,  inf_ci_Ca  ( v_val ), tau_ci_Ca  ( v_val ) );
    double dcdt4  = dmdt ( ion [ c_KCa  ] [ id ] + DT * dcdt3  ,  inf_c_KCa ( v_val, Ca_val + DT * dCadt3 ), tau_c_KCa ( v_val, Ca_val + DT * dCadt3 ) );
    //double dcdt4  = dmdt ( ion [ c_KCa  ] [ id ] + DT * dcdt3,  inf_c_KCa ( v_val, Ca_val ), tau_c_KCa ( v_val, Ca_val ) );
    
    double I_Ca4 = ( 1e-3 * cond [ g_Ca ] [ id ] / elem [ area ] [ id ] * ( ion [ ch_Ca ] [ id ] * DT * dchdt3 ) * ( ion [ ch_Ca ] [ id ] * DT * dchdt3 ) * ( ion [ ci_Ca ] [ id ] * DT * dcidt3 ) * ( v_val - V_Ca_GR ) ); //I_Ca [mA/cm^2]
    double dCadt4 = - I_Ca4 / ( 2.0 * F * SHELL1_D ) - B_Ca1 * ( Ca_val + DT * dCadt3 - Ca1_0 );
    
    ion [ n_KV   ] [ id ] += DT * ( dndt1  + dndt2  * 2.0 + dndt3   * 2.0 + dndt4  ) / 6.0;
    ion [ a_KA   ] [ id ] += DT * ( dadt1  + dadt2  * 2.0 + dadt3   * 2.0 + dadt4  ) / 6.0;
    ion [ b_KA   ] [ id ] += DT * ( dbdt1  + dbdt2  * 2.0 + dbdt3   * 2.0 + dbdt4  ) / 6.0;
    ion [ ir_KIR ] [ id ] += DT * ( dirdt1 + dirdt2 * 2.0 + dirdt3  * 2.0 + dirdt4 ) / 6.0;
    ion [ s_KM   ] [ id ] += DT * ( dsdt1  + dsdt2  * 2.0 + dsdt3   * 2.0 + dsdt4  ) / 6.0;
    ion [ ch_Ca  ] [ id ] += DT * ( dchdt1 + dchdt2 * 2.0 + dchdt3  * 2.0 + dchdt4 ) / 6.0;
    ion [ ci_Ca  ] [ id ] += DT * ( dcidt1 + dcidt2 * 2.0 + dcidt3  * 2.0 + dcidt4 ) / 6.0;
    ion [ c_KCa  ] [ id ] += DT * ( dcdt1  + dcdt2  * 2.0 + dcdt3   * 2.0 + dcdt4  ) / 6.0;
    elem [ Ca ] [ id ]    += DT * ( dCadt1 + dCadt2 * 2.0 + dCadt3  * 2.0 + dCadt4 ) / 6.0;
    
    elem [ v ] [ id ] = elem_v [ id ];    
  }
}
__global__ void gr_update_ion_exp_imp ( neuron_t *d_gr, neuron_solve_t *d_gr_solve, const double DT )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  double **elem = d_gr -> elem;
  double **ion  = d_gr -> ion;
  //double **cond = d_gr -> cond;
  
  if ( id < d_gr -> nc)
  {
    double v_val  = ( elem [ v ] [ id ] + d_gr_solve -> vec [ cn_v_old ] [ id ] ) / 2.0;
    double Ca_val = elem [ Ca ] [ id ];

    ion [ n_KV   ] [ id ] = ( 2.0 * DT * inf_n_KV   ( v_val ) + ( 2.0 * tau_n_KV   ( v_val ) - DT ) * ion [ n_KV   ] [ id ] ) / ( 2.0 * tau_n_KV   ( v_val ) + DT );
    ion [ a_KA   ] [ id ] = ( 2.0 * DT * inf_a_KA   ( v_val ) + ( 2.0 * tau_a_KA   ( v_val ) - DT ) * ion [ a_KA   ] [ id ] ) / ( 2.0 * tau_a_KA   ( v_val ) + DT );
    ion [ b_KA   ] [ id ] = ( 2.0 * DT * inf_b_KA   ( v_val ) + ( 2.0 * tau_b_KA   ( v_val ) - DT ) * ion [ b_KA   ] [ id ] ) / ( 2.0 * tau_b_KA   ( v_val ) + DT );
    ion [ ir_KIR ] [ id ] = ( 2.0 * DT * inf_ir_KIR ( v_val ) + ( 2.0 * tau_ir_KIR ( v_val ) - DT ) * ion [ ir_KIR ] [ id ] ) / ( 2.0 * tau_ir_KIR ( v_val ) + DT );
    ion [ s_KM   ] [ id ] = ( 2.0 * DT * inf_s_KM   ( v_val ) + ( 2.0 * tau_s_KM   ( v_val ) - DT ) * ion [ s_KM   ] [ id ] ) / ( 2.0 * tau_s_KM   ( v_val ) + DT );

    ion [ c_KCa  ] [ id ] = ( 2.0 * DT * inf_c_KCa  ( v_val, Ca_val ) + ( 2.0 * tau_c_KCa ( v_val, Ca_val ) - DT ) * ion [ c_KCa   ] [ id ] ) / ( 2.0 * tau_c_KCa ( v_val, Ca_val ) + DT );    
    d_gr_solve -> vec [ cn_v_old ] [ id ]  = elem [ v ] [ id ] ;
  }
}

__global__ void gr_update_ion_RKC_exp_imp ( neuron_t *d_gr, neuron_solve_t *d_gr_solve, double *vnew, const double DT )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  double **elem = d_gr -> elem;
  double **ion  = d_gr -> ion;
  double **cond = d_gr -> cond;
  
  if ( id < d_gr -> nc)
  {
    double v_val  = ( elem [ v ] [ id ] + vnew [ id ] ) / 2.0;
    double Ca_val = elem [ Ca ] [ id ];
    ion [ n_KV   ] [ id ] = ( 2.0 * DT * inf_n_KV   ( v_val ) + ( 2.0 * tau_n_KV   ( v_val ) - DT ) * ion [ n_KV   ] [ id ] ) / ( 2.0 * tau_n_KV   ( v_val ) + DT );
    ion [ a_KA   ] [ id ] = ( 2.0 * DT * inf_a_KA   ( v_val ) + ( 2.0 * tau_a_KA   ( v_val ) - DT ) * ion [ a_KA   ] [ id ] ) / ( 2.0 * tau_a_KA   ( v_val ) + DT );
    ion [ b_KA   ] [ id ] = ( 2.0 * DT * inf_b_KA   ( v_val ) + ( 2.0 * tau_b_KA   ( v_val ) - DT ) * ion [ b_KA   ] [ id ] ) / ( 2.0 * tau_b_KA   ( v_val ) + DT );
    ion [ ir_KIR ] [ id ] = ( 2.0 * DT * inf_ir_KIR ( v_val ) + ( 2.0 * tau_ir_KIR ( v_val ) - DT ) * ion [ ir_KIR ] [ id ] ) / ( 2.0 * tau_ir_KIR ( v_val ) + DT );
    ion [ s_KM   ] [ id ] = ( 2.0 * DT * inf_s_KM   ( v_val ) + ( 2.0 * tau_s_KM   ( v_val ) - DT ) * ion [ s_KM   ] [ id ] ) / ( 2.0 * tau_s_KM   ( v_val ) + DT );

    ion [ c_KCa  ] [ id ] = ( 2.0 * DT * inf_c_KCa  ( v_val, Ca_val ) + ( 2.0 * tau_c_KCa ( v_val, Ca_val ) - DT ) * ion [ c_KCa   ] [ id ] ) / ( 2.0 * tau_c_KCa ( v_val, Ca_val ) + DT );    
    elem [ v ] [ id ] = vnew [ id ];
  }
}
__global__ void gr_update_ion_RK2 ( neuron_t *d_gr, neuron_solve_t *d_gr_solve, const double DT )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  double **elem = d_gr -> elem;
  double **ion  = d_gr -> ion;
  double **cond = d_gr -> cond;
  
  if ( id < d_gr -> nc)
  {
    double v_val = elem [ v ] [ id ];
    double Ca_val = elem [ Ca ] [ id ];
    double I_Ca = ( 1e-3 * cond [ g_Ca ] [ id ] / elem [ area ] [ id ] * ion [ ch_Ca ] [ id ] * ion [ ch_Ca ] [ id ] * ion [ ci_Ca ] [ id ] * ( v_val - V_Ca_GR ) ); //I_Ca [mA/cm^2]
    
    double dndt1  = dmdt ( ion [ n_KV   ] [ id ],  inf_n_KV   ( v_val ), tau_n_KV   ( v_val ) );
    double dadt1  = dmdt ( ion [ a_KA   ] [ id ],  inf_a_KA   ( v_val ), tau_a_KA   ( v_val ) );
    double dbdt1  = dmdt ( ion [ b_KA   ] [ id ],  inf_b_KA   ( v_val ), tau_b_KA   ( v_val ) );
    double dirdt1 = dmdt ( ion [ ir_KIR ] [ id ],  inf_ir_KIR ( v_val ), tau_ir_KIR ( v_val ) );
    double dsdt1  = dmdt ( ion [ s_KM   ] [ id ],  inf_s_KM   ( v_val ), tau_s_KM   ( v_val ) );
    double dchdt1 = dmdt ( ion [ ch_Ca  ] [ id ],  inf_ch_Ca  ( v_val ), tau_ch_Ca  ( v_val ) );
    double dcidt1 = dmdt ( ion [ ci_Ca  ] [ id ],  inf_ci_Ca  ( v_val ), tau_ci_Ca  ( v_val ) );
    double dcdt1  = dmdt ( ion [ c_KCa  ] [ id ],  inf_c_KCa ( v_val, Ca_val ), tau_c_KCa ( v_val, Ca_val ) );
    
    double dndt2  = dmdt ( ion [ n_KV   ] [ id ] + DT * dndt1 ,  inf_n_KV   ( v_val ), tau_n_KV   ( v_val ) );
    double dadt2  = dmdt ( ion [ a_KA   ] [ id ] + DT * dadt1 ,  inf_a_KA   ( v_val ), tau_a_KA   ( v_val ) );
    double dbdt2  = dmdt ( ion [ b_KA   ] [ id ] + DT * dbdt1 ,  inf_b_KA   ( v_val ), tau_b_KA   ( v_val ) );
    double dirdt2 = dmdt ( ion [ ir_KIR ] [ id ] + DT * dirdt1,  inf_ir_KIR ( v_val ), tau_ir_KIR ( v_val ) );
    double dsdt2  = dmdt ( ion [ s_KM   ] [ id ] + DT * dsdt1 ,  inf_s_KM   ( v_val ), tau_s_KM   ( v_val ) );
    double dchdt2 = dmdt ( ion [ ch_Ca  ] [ id ] + DT * dchdt1,  inf_ch_Ca  ( v_val ), tau_ch_Ca  ( v_val ) );
    double dcidt2 = dmdt ( ion [ ci_Ca  ] [ id ] + DT * dcidt1,  inf_ci_Ca  ( v_val ), tau_ci_Ca  ( v_val ) );
    double dcdt2  = dmdt ( ion [ c_KCa  ] [ id ] + DT * dcdt1 ,  inf_c_KCa ( v_val, Ca_val ), tau_c_KCa ( v_val, Ca_val ) );
    
    ion [ n_KV   ] [ id ] += 0.5 * DT * ( dndt1  + dndt2  );
    ion [ a_KA   ] [ id ] += 0.5 * DT * ( dadt1  + dadt2  );
    ion [ b_KA   ] [ id ] += 0.5 * DT * ( dbdt1  + dbdt2  );
    ion [ ir_KIR ] [ id ] += 0.5 * DT * ( dirdt1 + dirdt2 );
    ion [ s_KM   ] [ id ] += 0.5 * DT * ( dsdt1  + dsdt2  );
    ion [ ch_Ca  ] [ id ] += 0.5 * DT * ( dchdt1 + dchdt2 );
    ion [ ci_Ca  ] [ id ] += 0.5 * DT * ( dcidt1 + dcidt2 );
    ion [ c_KCa  ] [ id ] += 0.5 * DT * ( dcdt1  + dcdt2  );
    //elem [ Ca ] [ id ]    += 0.5 * DT * ( dCadt1 + dCadt2 );
    // RK4    
    double k1 = DT * ( - I_Ca / ( 2.0 * F * SHELL1_D ) - B_Ca1 * ( Ca_val            - Ca1_0 ) ); //[mA*mol/cm^3*sec*A]=[M/sec] **[1mM = 1mol/m^3]**
    double k2 = DT * ( - I_Ca / ( 2.0 * F * SHELL1_D ) - B_Ca1 * ( Ca_val + k1 / 2.0 - Ca1_0 ) );
    double k3 = DT * ( - I_Ca / ( 2.0 * F * SHELL1_D ) - B_Ca1 * ( Ca_val + k2 / 2.0 - Ca1_0 ) );
    double k4 = DT * ( - I_Ca / ( 2.0 * F * SHELL1_D ) - B_Ca1 * ( Ca_val + k3       - Ca1_0 ) );
    elem [ Ca ] [ id ] += ( k1 + k2 * 2.0 + k3 * 2.0 + k4 ) / 6.0;
  }
}

__global__ void gr_update_ion_RKC_RK2 ( neuron_t *d_gr, neuron_solve_t *d_gr_solve,  double *elem_v, const double DT )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  double **elem = d_gr -> elem;
  double **ion  = d_gr -> ion;
  double **cond = d_gr -> cond;
  
  if ( id < d_gr -> nc)
  {
    double v_val = elem_v [ id ];
    double Ca_val = elem [ Ca ] [ id ];
    double I_Ca = ( 1e-3 * cond [ g_Ca ] [ id ] / elem [ area ] [ id ] * ion [ ch_Ca ] [ id ] * ion [ ch_Ca ] [ id ] * ion [ ci_Ca ] [ id ] * ( v_val - V_Ca_GR ) ); //I_Ca [mA/cm^2]

    double dndt1  = dmdt ( ion [ n_KV   ] [ id ],  inf_n_KV   ( v_val ), tau_n_KV   ( v_val ) );
    double dadt1  = dmdt ( ion [ a_KA   ] [ id ],  inf_a_KA   ( v_val ), tau_a_KA   ( v_val ) );
    double dbdt1  = dmdt ( ion [ b_KA   ] [ id ],  inf_b_KA   ( v_val ), tau_b_KA   ( v_val ) );
    double dirdt1 = dmdt ( ion [ ir_KIR ] [ id ],  inf_ir_KIR ( v_val ), tau_ir_KIR ( v_val ) );
    double dsdt1  = dmdt ( ion [ s_KM   ] [ id ],  inf_s_KM   ( v_val ), tau_s_KM   ( v_val ) );
    double dchdt1 = dmdt ( ion [ ch_Ca  ] [ id ],  inf_ch_Ca  ( v_val ), tau_ch_Ca  ( v_val ) );
    double dcidt1 = dmdt ( ion [ ci_Ca  ] [ id ],  inf_ci_Ca  ( v_val ), tau_ci_Ca  ( v_val ) );
    double dcdt1  = dmdt ( ion [ c_KCa  ] [ id ],  inf_c_KCa ( v_val, Ca_val ), tau_c_KCa ( v_val, Ca_val ) );
    
    double dndt2  = dmdt ( ion [ n_KV   ] [ id ] + DT * dndt1 ,  inf_n_KV   ( v_val ), tau_n_KV   ( v_val ) );
    double dadt2  = dmdt ( ion [ a_KA   ] [ id ] + DT * dadt1 ,  inf_a_KA   ( v_val ), tau_a_KA   ( v_val ) );
    double dbdt2  = dmdt ( ion [ b_KA   ] [ id ] + DT * dbdt1 ,  inf_b_KA   ( v_val ), tau_b_KA   ( v_val ) );
    double dirdt2 = dmdt ( ion [ ir_KIR ] [ id ] + DT * dirdt1,  inf_ir_KIR ( v_val ), tau_ir_KIR ( v_val ) );
    double dsdt2  = dmdt ( ion [ s_KM   ] [ id ] + DT * dsdt1 ,  inf_s_KM   ( v_val ), tau_s_KM   ( v_val ) );
    double dchdt2 = dmdt ( ion [ ch_Ca  ] [ id ] + DT * dchdt1,  inf_ch_Ca  ( v_val ), tau_ch_Ca  ( v_val ) );
    double dcidt2 = dmdt ( ion [ ci_Ca  ] [ id ] + DT * dcidt1,  inf_ci_Ca  ( v_val ), tau_ci_Ca  ( v_val ) );
    double dcdt2  = dmdt ( ion [ c_KCa  ] [ id ] + DT * dcdt1 ,  inf_c_KCa ( v_val, Ca_val ), tau_c_KCa ( v_val, Ca_val ) );
    
    ion [ n_KV   ] [ id ] += 0.5 * DT * ( dndt1  + dndt2  );
    ion [ a_KA   ] [ id ] += 0.5 * DT * ( dadt1  + dadt2  );
    ion [ b_KA   ] [ id ] += 0.5 * DT * ( dbdt1  + dbdt2  );
    ion [ ir_KIR ] [ id ] += 0.5 * DT * ( dirdt1 + dirdt2 );
    ion [ s_KM   ] [ id ] += 0.5 * DT * ( dsdt1  + dsdt2  );
    ion [ ch_Ca  ] [ id ] += 0.5 * DT * ( dchdt1 + dchdt2 );
    ion [ ci_Ca  ] [ id ] += 0.5 * DT * ( dcidt1 + dcidt2 );
    ion [ c_KCa  ] [ id ] += 0.5 * DT * ( dcdt1  + dcdt2  );
    //elem [ Ca ] [ id ]    += 0.5 * DT * ( dCadt1 + dCadt2 );
    // RK4    
    double k1 = DT * ( - I_Ca / ( 2.0 * F * SHELL1_D ) - B_Ca1 * ( Ca_val            - Ca1_0 ) ); //[mA*mol/cm^3*sec*A]=[M/sec] **[1mM = 1mol/m^3]**
    double k2 = DT * ( - I_Ca / ( 2.0 * F * SHELL1_D ) - B_Ca1 * ( Ca_val + k1 / 2.0 - Ca1_0 ) );
    double k3 = DT * ( - I_Ca / ( 2.0 * F * SHELL1_D ) - B_Ca1 * ( Ca_val + k2 / 2.0 - Ca1_0 ) );
    double k4 = DT * ( - I_Ca / ( 2.0 * F * SHELL1_D ) - B_Ca1 * ( Ca_val + k3       - Ca1_0 ) );
    elem [ Ca ] [ id ] += ( k1 + k2 * 2.0 + k3 * 2.0 + k4 ) / 6.0;
  }
}
__global__ void gr_update_ion ( neuron_t *d_gr, neuron_solve_t *d_gr_solve, const double DT )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  double **elem = d_gr -> elem;
  double **ion  = d_gr -> ion;
  //double **cond = d_gr -> cond;
  
  if ( id < d_gr -> nc)
  {
    double v_val = elem [ v ] [ id ];
    double Ca_val = elem [ Ca ] [ id ];
    //double I_Ca1 = ( 1e-3 * cond [ g_Ca ] [ id ] / elem [ area ] [ id ] * ion [ ch_Ca ] [ id ] * ion [ ch_Ca ] [ id ] * ion [ ci_Ca ] [ id ] * ( v_val - V_Ca_GR ) ); //I_Ca [mA/cm^2]
    
    ion [ n_KV   ] [ id ] = inf_n_KV   ( v_val ) + ( ion [ n_KV   ] [ id ] - inf_n_KV   ( v_val ) ) * exp ( - DT / tau_n_KV   ( v_val ) );
    ion [ a_KA   ] [ id ] = inf_a_KA   ( v_val ) + ( ion [ a_KA   ] [ id ] - inf_a_KA   ( v_val ) ) * exp ( - DT / tau_a_KA   ( v_val ) );
    ion [ b_KA   ] [ id ] = inf_b_KA   ( v_val ) + ( ion [ b_KA   ] [ id ] - inf_b_KA   ( v_val ) ) * exp ( - DT / tau_b_KA   ( v_val ) );
    ion [ ir_KIR ] [ id ] = inf_ir_KIR ( v_val ) + ( ion [ ir_KIR ] [ id ] - inf_ir_KIR ( v_val ) ) * exp ( - DT / tau_ir_KIR ( v_val ) );
    ion [ s_KM   ] [ id ] = inf_s_KM   ( v_val ) + ( ion [ s_KM   ] [ id ] - inf_s_KM   ( v_val ) ) * exp ( - DT / tau_s_KM   ( v_val ) );
    ion [ c_KCa  ] [ id ] = ( inf_c_KCa ( v_val, Ca_val ) 
			     + ( ion [ c_KCa ] [ id ] - inf_c_KCa ( v_val, Ca_val ) ) * exp ( - DT / tau_c_KCa ( v_val, Ca_val ) ) );
  }
}

__global__ void gr_update_ion_RKC ( neuron_t *d_gr, neuron_solve_t *d_gr_solve,  double *elem_v, const double DT )
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  double **elem = d_gr -> elem;
  double **ion  = d_gr -> ion;
  double **cond = d_gr -> cond;
  
  if ( id < d_gr -> nc)
  {
    double v_val = elem_v [ id ];
    double Ca_val = elem [ Ca ] [ id ];
    double I_Ca1 = ( 1e-3 * cond [ g_Ca ] [ id ] / elem [ area ] [ id ] * ion [ ch_Ca ] [ id ] 
      * ion [ ch_Ca ] [ id ] * ion [ ci_Ca ] [ id ] * ( v_val - V_Ca_GR ) ); //I_Ca [mA/cm^2]
    
    ion [ n_KV   ] [ id ] = inf_n_KV   ( v_val ) + ( ion [ n_KV   ] [ id ] - inf_n_KV   ( v_val ) ) * exp ( - DT / tau_n_KV   ( v_val ) );
    ion [ a_KA   ] [ id ] = inf_a_KA   ( v_val ) + ( ion [ a_KA   ] [ id ] - inf_a_KA   ( v_val ) ) * exp ( - DT / tau_a_KA   ( v_val ) );
    ion [ b_KA   ] [ id ] = inf_b_KA   ( v_val ) + ( ion [ b_KA   ] [ id ] - inf_b_KA   ( v_val ) ) * exp ( - DT / tau_b_KA   ( v_val ) );
    ion [ ir_KIR ] [ id ] = inf_ir_KIR ( v_val ) + ( ion [ ir_KIR ] [ id ] - inf_ir_KIR ( v_val ) ) * exp ( - DT / tau_ir_KIR ( v_val ) );
    ion [ s_KM   ] [ id ] = inf_s_KM   ( v_val ) + ( ion [ s_KM   ] [ id ] - inf_s_KM   ( v_val ) ) * exp ( - DT / tau_s_KM   ( v_val ) );
    ion [ c_KCa  ] [ id ] = ( inf_c_KCa ( v_val, Ca_val ) 
			     + ( ion [ c_KCa ] [ id ] - inf_c_KCa ( v_val, Ca_val ) ) * exp ( - DT / tau_c_KCa ( v_val, Ca_val ) ) );
    ion [ ch_Ca  ] [ id ] = inf_ch_Ca  ( v_val ) + ( ion [ ch_Ca  ] [ id ] - inf_ch_Ca  ( v_val ) ) * exp ( - DT / tau_ch_Ca  ( v_val ) );
    ion [ ci_Ca  ] [ id ] = inf_ci_Ca  ( v_val ) + ( ion [ ci_Ca  ] [ id ] - inf_ci_Ca  ( v_val ) ) * exp ( - DT / tau_ci_Ca  ( v_val ) );

    
    // Integral method
    I_Ca1 = ( 1e-3 * cond [ g_Ca ] [ id ] / elem [ area ] [ id ] * ion [ ch_Ca ] [ id ] * ion [ ch_Ca ] [ id ] * ion [ ci_Ca ] [ id ] * ( v_val - V_Ca_GR ) ); //I_Ca [mA/cm^2]
    double Cinf = Ca1_0 - I_Ca1 / ( 2.0 * 9.6485e4 * 0.2e-4 * B_Ca1 );
    elem [ Ca ] [ id ]  = Cinf - (Cinf - Ca_val) * exp ( - DT * B_Ca1 );

  }
}

__host__ void gr_initialize_ion ( neuron_t *gr )
{
  double **elem = gr -> elem;
  double **ion  = gr -> ion;
  double init_v_rand = 0.0;
  for ( int i = 0; i < gr -> nc; i++) {
    
    if ( i % GR_COMP == 0 ) { init_v_rand = ( ( ( double ) rand ( ) + 1.0 ) / ( ( double ) RAND_MAX + 2.0) ) - 0.5; }

    elem [ v     ] [ i ] = V_INIT_GR + 2.0 * init_v_rand;
    elem [ Ca    ] [ i ] = Ca1_0;
    elem [ i_ext ] [ i ] = 0.0;    
    double v_val = elem [ v  ] [ i ];
    double ca_val = elem [ Ca ] [ i ];

    ion [ n_KV ]   [ i ] = inf_n_KV   ( v_val );
    ion [ a_KA ]   [ i ] = inf_a_KA   ( v_val );
    ion [ b_KA ]   [ i ] = inf_b_KA   ( v_val );
    ion [ ir_KIR ] [ i ] = inf_ir_KIR ( v_val );
    ion [ s_KM ]   [ i ] = inf_s_KM   ( v_val );
    ion [ c_KCa ]  [ i ] = inf_c_KCa  ( v_val, ca_val );
    ion [ ch_Ca ]  [ i ] = inf_ch_Ca  ( v_val );
    ion [ ci_Ca ]  [ i ] = inf_ci_Ca  ( v_val );

    ion [ o_Na  ] [ i ] = 0.0;
    ion [ c2_Na ] [ i ] = 0.0;
    ion [ c3_Na ] [ i ] = 0.0;
    ion [ c4_Na ] [ i ] = 0.0;
    ion [ c5_Na ] [ i ] = 0.0;
    ion [ i1_Na ] [ i ] = 0.0;
    ion [ i2_Na ] [ i ] = 0.0;
    ion [ i3_Na ] [ i ] = 0.0;
    ion [ i4_Na ] [ i ] = 0.0;
    ion [ i5_Na ] [ i ] = 0.0;
    ion [ i6_Na ] [ i ] = 0.0;
    ion [ c1_Na ]  [ i ] = 1.0;
  }
}
