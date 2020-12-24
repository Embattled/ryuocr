/* --------------------------------------------------------
 * FHRect.cpp
 *      Written by F.Horie.
 * -------------------------------------------------------- */


#include    "FHRect.h"


/* --------------------------------------------------------
 * Rectangle
 * -------------------------------------------------------- */

int FHTrim(FHImg *src, FHImg *dst, FHRect rect){
    int     x, y;
    uchar   *sp, *dp;
    FHImg     imgtmp;

    FHImgInit(&imgtmp);
    if ( FHImgOpen(&imgtmp, rect.y1-rect.y0+1, rect.x1-rect.x0+1, src->ch) ){ return(-1); }
    for ( y=0 ; y<imgtmp.h ; y++ ){
        sp = FHImgPtr(src, rect.y0+y);
        dp = FHImgPtr(&imgtmp, y);
        for ( x=0 ; x<imgtmp.linesize ; x++ ){
            dp[x] = sp[rect.x0*imgtmp.ch+x];
        }
    }
    FHImgMv(&imgtmp, dst);
    FHImgClose(&imgtmp);
    return(0);
}
