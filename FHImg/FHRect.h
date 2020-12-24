/* --------------------------------------------------------
 * FHRect.h
 *      Written by F.Horie.
 * -------------------------------------------------------- */


#ifndef     FHRECT_H
#define     FHRECT_H


#include    "FHImg.h"


/* --------------------------------------------------------
 * Rectangle (FHRect.cpp)
 * -------------------------------------------------------- */

typedef struct FHRect {
    int     x0;
    int     y0;
    int     x1;
    int     y1;
} FHRect;


inline FHRect FHRectSet(int x0, int y0, int x1, int y1){
    FHRect  rect;
    if ( x0>x1 || y0>y1 ){ 
        rect.x0 = -1;
        rect.y0 = -1;
        rect.x1 = -1;
        rect.y1 = -1;
        return(rect); 
    }
    rect.x0 = x0;
    rect.y0 = y0;
    rect.x1 = x1;
    rect.y1 = y1;
    return(rect);
}


extern int      FHTrim(FHImg *src, FHImg *dst, FHRect rect);



#endif      /* FHRECT_H */
