#include "output.cuh"

void output_time ( const int n_gr, const char *type1, const int n_go, const char *type2, 
                   const int n_pkj, const char *type3, const int n_io, const char *type4, 
                   const double comp_time, const int sim_time )
{  
  FILE *time_plot;
  time_plot = fopen ( "comptime.csv", "a" );
  if ( time_plot == NULL ) { printf("go_time_file_open_error \n"); }
  else 
  {
    fprintf ( time_plot, "Gr,%d,", n_gr ); int l_err1 = fputs ( type1, time_plot ); 
    fprintf ( time_plot, ",Go,%d,",  n_go  ); int l_err2 = fputs ( type2, time_plot ); 
    fprintf ( time_plot, ",Pkj,%d,", n_pkj ); int l_err3 = fputs ( type3, time_plot ); 
    fprintf ( time_plot, ",Io,%d,",  n_io  ); int l_err4 = fputs ( type4, time_plot ); 
    fprintf ( time_plot, ",CompTime,%lf,SimTime,%d\n", comp_time, sim_time ); 
  }
  fclose ( time_plot );
}

void output_time2 ( const int n_gr, const char *type1, const int n_go, const char *type2, 
  const int n_pkj, const char *type3, const int n_io, const char *type4, 
  const double comp_time, const double comp_time_half, const int sim_time )
{  
FILE *time_plot;
time_plot = fopen ( "comptime.csv", "a" );
if ( time_plot == NULL ) { printf("go_time_file_open_error \n"); }
else 
{
fprintf ( time_plot, "Gr,%d,", n_gr ); int l_err1 = fputs ( type1, time_plot ); 
fprintf ( time_plot, ",Go,%d,",  n_go  ); int l_err2 = fputs ( type2, time_plot ); 
fprintf ( time_plot, ",Pkj,%d,", n_pkj ); int l_err3 = fputs ( type3, time_plot ); 
fprintf ( time_plot, ",Io,%d,",  n_io  ); int l_err4 = fputs ( type4, time_plot ); 
fprintf ( time_plot, ",AllCompTime,%lf,0.5s-endCompTime,%lf,SimTime,%d\n", comp_time, comp_time_half, sim_time ); 
}
fclose ( time_plot );
}

void raster_plot ( const int n_go, const char *type ){
  if ( n_go > 0 ){
    FILE *fp, *fw;
    if ( 0 == strncmp ( type, "BACKWARD_EULER", 14 ) ) {   fp = fopen ( "go_v_by_BEm.csv", "r" ); fw = fopen ( "go_raster_by_BEm.csv", "w" ); }
    else if ( 0 == strncmp ( type, "CRANK_NICOLSON", 14 ) ) {   fp = fopen ( "go_v_by_CNm.csv", "r" ); fw = fopen ( "go_raster_by_CNm.csv", "w" ); }
    else if ( 0 == strncmp ( type, "FORWARD_EULER", 13 ) ) {   fp = fopen ( "go_v_by_FEm.csv", "r" );  fw = fopen ( "go_raster_by_FEm.csv", "w" );}
    else if ( 0 == strncmp ( type, "RUNGE_KUTTA_4", 13 ) ) {   fp = fopen ( "go_v_by_RK4m.csv", "r" ); fw = fopen ( "go_raster_by_RK4.csv", "w" ); }
    else if ( 0 == strncmp ( type, "RKC", 3 ) ) {   fp = fopen ( "go_v_by_RKC.csv", "r" );  fw = fopen ( "go_raster_by_RKC.csv", "w" );}
    else { printf ("go_initialization_error \n"); exit ( 1 ); }
    
    double *old_v = ( double * ) malloc ( ( n_go ) * sizeof ( double ) );
    double *v = ( double * ) malloc ( ( n_go ) * sizeof ( double ) );
    for ( int i = 0; i < n_go; i++ ) { v [ i ] = old_v [ i ] = -70.0; }
    double t;//, dam;
    
    printf ( "Debug for output \n" );    
    while ( ! feof ( fp ) ) {
      if ( fscanf ( fp, "%lf,", &t ) == ( EOF ) ){
        printf ( "raster_FILE_ERROR1\n" ); exit ( 1 );
      }
      fprintf ( fw, "%lf,", t ); 

      for ( int i = 0; i < n_go; i++ ) {
        if ( fscanf ( fp, "%lf,", &( v [ i ] ) )== ( EOF ) ){
          printf ( "raster_FILE_ERROR2\n" ); exit ( 1 );
        }
        //fprintf ( fw, "%lf,", v [ i ]  );
        if ( ( v [ i ] > 0.0 ) && ( old_v [ i ] < 0.0 ) ) { fprintf ( fw, "%d,", i  ); printf ( "%d,", i  ); }
        else { fprintf ( fw, "," ); }
        old_v [ i ] = v [ i ]; 
      }
     // fscanf ( fp, "%lf,", &dam );
      fprintf ( fw, "\n" );
    }
    fclose ( fp );
    fclose ( fw );
    free ( v );
    free ( old_v );
  }
}