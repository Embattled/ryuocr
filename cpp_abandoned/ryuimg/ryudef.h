#pragma once
#ifndef _RYUDEF_H
#define _RYUDEF_H
#include <new>


#define     DEFMAX(x,y)     ((x)>(y)?(x):(y))
#define     DEFMIN(x,y)     ((x)<(y)?(x):(y))
#define     DEFSQUARE(x)    ((x)*(x)) 


#define     DEFIS2POW(n)    !((n) & ((n)-1))



#ifdef  __cplusplus

#define     DEFNEW(type, size)  new (std::nothrow) type[size]
#define     DEFFREE(x)          do{ if ( x ){ delete [](x); x = NULL; } }while(0)

#else

#define     DEFNEW(type, size)  (type*)malloc(sizeof(type)*(size))
#define     DEFFREE(x)          do{ if ( x ){ free((void*)(x)); x = NULL; } }while(0)

#endif

# endif