/* --------------------------------------------------------
 * FHDef.h ---- C Util library ----
 *      Written by F.Horie.
 * -------------------------------------------------------- */


#ifndef     FHDEF_H
#define     FHDEF_H


#include    <memory.h>


typedef     unsigned char   uchar;
typedef     unsigned short  ushort;


#define     FH_INLINE      static inline
#define     FH_BUF_LEN     256


#define     M_PIF   3.14159265358979323846f
#define     FH_USHORT_MAX  65536
#define     FH_UCHAR_MAX   256
#define     FH_UINT_MAX    4294967296

#define     FHMax(x, y)   ((x)>(y)?(x):(y))
#define     FHMin(x, y)   ((x)<(y)?(x):(y))
#define     FHSquare(x)   ((x)*(x)) 
#define     FHStep(x)     ((x)>=0?1:0)


#ifdef  __cplusplus

#define     FHMalloc(type, size)  new type[size]
#define     FHFree(x)             do{ if ( x ){ delete [](x); x = NULL; } }while(0)

#else

#include    <stdlib.h>
#define     FHMalloc(type, size)  (type*)malloc(sizeof(type)*(size))
#define     FHFree(x)             do{ if ( x ){ free((void*)(x)); x = NULL; } }while(0)

#endif

#define     FHSwap(type, x, y)    do{ type tmp = x; x = y; y = tmp;}while(0)
#define     FHInside(a, x, b)     ((a)<=(x) && (x)<=(b))
#define     FHOutside(a, x, b)    ((x)<=(a) && (b)<=(x))
#define     FHClamp(a, x, b)      (FHMin((b), FHMax((a), (x))))

#define     FHCp(x, y, size)      memcpy(y, x, sizeof(x[0])*size)

#define     FHSubAll(x, size, val)     do{ \
    int     i; \
    for ( i=0 ; i<size ; i++ ){ x[i] = val; } \
}while(0)

#define     FHIs2Pow(n)         !((n) & ((n)-1))


#endif      /* FHDEF_H */
