#ifndef _OUTPUT_CUH_
#define _OUTPUT_CUH_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void output_time ( const int, const char *, const int, const char *, const int, const char *, const int, const char *, const double, const int );
extern void output_time2 ( const int, const char *, const int, const char *, const int, const char *, const int, const char *, const double,  const double, const int );
extern void raster_plot ( const int, const char * );

#endif // _OUTPUT_CUH_