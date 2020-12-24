/* --------------------------------------------------------
 * FHImgBase.cpp ---- Base of image processing ----
 *      Written by F.Horie.
 * -------------------------------------------------------- */


#include    <assert.h>
#include    <stdlib.h>
#include    <memory.h>
#include    <math.h>
#include    "FHImg.h"


/* --------------------------------------------------------
 * Open / Close
 * -------------------------------------------------------- */

void FHImgInit(FHImg *src){
    src->d = 0;
    src->h = src->w = src->ch = src->size = src->linesize = 0;
    return;
}


int FHImgOpen(FHImg *src, int h, int w, int ch){
    if ( ch!=1 && ch!=3 ){ return(-1); }
    if ( h<=0 || w<=0 ){ return(-1); }
    if ( h==src->h && w==src->w && ch==src->ch ){ return(0); }
    FHImgFinish(src);
    src->d = FHMalloc(uchar, h*w*ch);
    if ( src->d==0 ){ return(-1); }
    src->h = h;
    src->w = w;
    src->ch = ch;
    src->linesize = w * ch;
    src->size = h * src->linesize;
    return(0);
}


void FHImgClose(FHImg *src){
    FHFree(src->d);
    FHImgInit(src);
    return;
}


void FHImgFinish(FHImg *src){
    FHImgClose(src);
    return;
}


uchar *FHImgPtr(FHImg *src, int y){ 
    return(&(src->d[y*src->linesize])); 
}


/* --------------------------------------------------------
 * Copy / Move
 * -------------------------------------------------------- */

int FHImgCp(FHImg *src, FHImg *dst){
    if ( src==dst ){ return(0); }
    if ( FHImgOpen(dst, src->h, src->w, src->ch) ){ return(-1); }
    if ( NULL==FHCp(src->d, dst->d, src->size) ){ return(-1); }
    return(0);
}


void FHImgMv(FHImg *src, FHImg *dst){
    if ( src==dst ){ return; }
    FHImgFinish(dst);
    *dst = *src;
    FHImgInit(src);
    return;
}


/* --------------------------------------------------------
 * Useful
 * -------------------------------------------------------- */

float linear_interp(float x1, float x1_d, float x2_d){
    // return(x1 * x2_d + (1.0f - x1) * x1_d);
    return((x2_d - x1_d) * x1 + x1_d);
}


#define     Decimal(x)  ((x)-floorf((x)))


uchar FHImgInterp(FHImg *src, float y, float x, int color){
    float   x1, x2;
    float   dec;
    int     ch = src->ch;
    int     res;
    int     xx1, xx2, yy1, yy2;
    uchar   *p1, *p2;

    dec = Decimal(x);

    xx1 = (int)x * ch + color;
    xx2 = FHMin((int)(x+1), src->w-1);
    xx2 = xx2 * ch + color;
    yy1 = (int)y;
    yy2 = FHMin((int)(y+1), src->h-1);

    // p1 = FHImgPtr(src, (int)y);
    // x1 = linear_interp(dec, p1[(int)x*ch+color], p1[(int)(x+1)*ch+color]);
    // p2 = FHImgPtr(src, (int)y+1);
    // x2 = linear_interp(dec, p2[(int)x*ch+color], p2[(int)(x+1)*ch+color]);

    p1 = FHImgPtr(src, yy1);
    x1 = linear_interp(dec, p1[xx1], p1[xx2]);
    p2 = FHImgPtr(src, yy2);
    x2 = linear_interp(dec, p2[xx1], p2[xx2]);

    dec = Decimal(y);

    res = (int)(linear_interp(dec, x1, x2) + 0.5f);
    if ( res>255 ){ res = 255; }
    if ( res<0 ){ res = 0; }
    return((uchar)res);
}


// int FHImgSum(FHImg *src){
//     int     sumval = 0;
//     int     i;

//     for ( i=0 ; i<src->size ; i++ ){ sumval+=src->d[i]; }
//     return(sumval);
// }


// void FHImgSubAll(FHImg *src, uchar x){
//     int     i;
//     for ( i=0 ; i<src->size ; i++ ){ src->d[i] = x; }
//     return;
// }


// int FHImgCheckSize(FHImg *img1, FHImg *img2){
//     if ( img1->w==img2->w && img1->h==img2->h && img1->ch==img2->ch ){ return(1); }
//     else{ return(0); }
// }
