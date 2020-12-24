/* --------------------------------------------------------
 * FHImg/resize.cpp
 *      Written by F.Horie.
 * -------------------------------------------------------- */


#include    <assert.h>
#include    "FHImg.h"


/* --------------------------------------------------------
 * Resizing by bilinear interpolation
 * -------------------------------------------------------- */

int FHResize(FHImg *src, FHImg *dst, float xs, float ys){
    int     x, y;
    float   dx, dy;
    uchar   *dp;
    FHImg   imgtmp;

    FHImgInit(&imgtmp);

    if ( xs<=0 || ys<=0 ){ return(-1); }
    if ( FHImgOpen(&imgtmp, (int)((float)src->h*ys), (int)((float)src->w*xs), src->ch) ){
        return(-1);
    }

    for ( y=0 ; y<imgtmp.h ; y++ ){
        dp = FHImgPtr(&imgtmp, y);
        dy = (float)y / ys;
        dy = FHMin(dy, (float)src->h-2);
        for ( x=0 ; x<imgtmp.linesize ; x++ ){
            /* RGB -> Gray */
            dx = (float)((float)((int)x/src->ch) / xs);
            dx = FHMin(dx, (float)src->w-2);

            /* Interpolation */
            dp[x] = FHImgInterp(src, dy, dx, x%src->ch);
        }
    }
    FHImgMv(&imgtmp, dst);
    FHImgFinish(&imgtmp);
    return(0);
}


int FHResizeSquare(FHImg *src, FHImg *dst, int size){
    float    xs, ys;
    if ( size<=0 ){ return(-1); }
    if ( src->w==size && src->h==size ){ FHImgCp(src, dst); return(0); }
    xs = (float)size/(float)src->w;
    if ( (int)(xs*(float)src->w)!=size ){ xs+=1.0f/(2.0f*(float)src->w); }
    ys = (float)size/(float)src->h;
    if ( (int)(ys*(float)src->h)!=size ){ ys+=1.0f/(2.0f*(float)src->h); }
    return(FHResize(src, dst, xs, ys));
}


int FHResizeInside(FHImg *src, FHImg *dst, int size){
    float   s, xs, ys;
    if ( size<=0 ){ return(-1); }
    if ( src->w==size && src->h==size ){ FHImgCp(src, dst); return(0); }
    xs = (float)size/(float)src->w;
    if ( (int)(xs*(float)src->w)!=size ){ xs+=1.0f/(2.0f*(float)src->w); }
    ys = (float)size/(float)src->h;
    if ( (int)(ys*(float)src->h)!=size ){ ys+=1.0f/(2.0f*(float)src->h); }
    s = FHMin(xs, ys);
    return(FHResize(src, dst, s, s));
}
