/* --------------------------------------------------------
 * FHRGB.cpp
 *      Written by F.Horie.
 * -------------------------------------------------------- */


#include    <assert.h>
#include    <stdio.h>
#include    "FHRGB.h"


uchar FHToGrayPix(uchar x1, uchar x2, uchar x3){
    return((uchar)(x1*0.299+x2*0.587+x3*0.114));
}


int FHLUT(FHImg *src, FHImg *dst, uchar *lut){
    int     i;
    FHImg   imgtmp;

    FHImgInit(&imgtmp);
    if ( FHImgOpen(&imgtmp, src->h, src->w, src->ch) ){ return(-1); }
    for ( i=0 ; i<imgtmp.size ; i++ ){ imgtmp.d[i] = lut[src->d[i]]; }
    FHImgMv(&imgtmp, dst);
    FHImgFinish(&imgtmp);
    return(0);
}



int FHToGray(FHImg *src, FHImg *dst){
    int     i;
    int     size;
    FHImg   imgtmp;

    FHImgInit(&imgtmp);
    if ( src->ch==1 ){ 
        return(FHImgCp(src, dst));
    }
    if ( FHImgOpen(&imgtmp, src->h, src->w, 1) ){ return(-1); }
    size = imgtmp.size;

    for ( i=0 ; i<size ; i++ ){
        imgtmp.d[i] = FHToGrayPix(src->d[i*3], src->d[i*3+1], src->d[i*3+2]);
    }
    FHImgMv(&imgtmp, dst);
    FHImgFinish(&imgtmp);
    return(0);
}


int FHToRGB(FHImg *src, FHImg *dst){
    int     i;
    FHImg   imgtmp;

    FHImgInit(&imgtmp);
    if ( src->ch==3 ){ 
        return(FHImgCp(src, dst)); 
    }
    if ( FHImgOpen(&imgtmp, src->h, src->w, 3) ){ return(-1); }

    for ( i=0 ; i<imgtmp.size ; i+=3 ){
        imgtmp.d[i  ] = src->d[i/3];
        imgtmp.d[i+1] = src->d[i/3];
        imgtmp.d[i+2] = src->d[i/3];
    }
    FHImgMv(&imgtmp, dst);
    FHImgFinish(&imgtmp);
    return(0);
}


int FHChangeLight(FHImg *src, FHImg *dst, int light){
    int     i;
    uchar   lut[256];

    if ( light<-255 || 255<light ){ return(-1); }
    if ( light>=0 ){
        for ( i=0 ; i<256-light ; i++ ){ lut[i] = (uchar)(i + light); }
        for ( i=256-light ; i<256 ; i++ ){ lut[i] = 255; }
    }
    else{
        for ( i=0 ; i<-light ; i++ ){ lut[i] = 0; }
        for ( i=-light ; i<256 ; i++ ){ lut[i] = (uchar)(i + light); }
    }

    return(FHLUT(src, dst, lut));
}


int FHExtractRGB(FHImg *src, FHImg *dst, int ch){
    int     i;
    FHImg   imgtmp;

    if ( src->ch==1 ){ return(-1); }
    if ( !(0<=ch && ch<=2) ){ return(-1); }

    FHImgInit(&imgtmp);
    if ( FHImgOpen(&imgtmp, src->h, src->w, 1) ){ return(-1); }

    for ( i=0 ; i<imgtmp.size ; i++ ){ imgtmp.d[i] = src->d[i*3+ch]; }
    FHImgMv(&imgtmp, dst);
    FHImgFinish(&imgtmp);
    return(0);
}


int FHImgInverse(FHImg *src, FHImg *dst){
    int     i;
    uchar   lut[256];

    for ( i=0 ; i<256 ; i++ ){ lut[i] = (uchar)(255 - i); }
    return(FHLUT(src, dst, lut));
}


static float ImgCalcVar(uchar *x, int size){
    int     i;
    float   mean;
    float   var;

    mean = 0;
    for ( i=0 ; i<size ; i++ )  mean += x[i];
    mean /= (float)size;
    var = 0;
    for ( i=0 ; i<size ; i++ )  var += (x[i] - mean) * (x[i] - mean);
    var /= (float)size;
    return(var);
}


int FHMaxVarColor(FHImg *src, FHImg *dst){
    int     i;
    int     ch = 0;
    float   maxvar = 0;
    float   vartmp;
    FHImg   imgtmp[3];

    if ( src->ch==1 ){ return(-1); }
    for ( i=0 ; i<3 ; i++ ){ 
        FHImgInit(&imgtmp[i]);
        if ( FHExtractRGB(src, &imgtmp[i], i) ){ return(-1); }
        vartmp = ImgCalcVar(imgtmp[i].d, imgtmp[i].size);
        if ( maxvar<vartmp ){ ch = i; maxvar = vartmp; }
    }
    FHImgMv(&imgtmp[ch], dst);
    for ( i=0 ; i<3 ; i++ ){ FHImgFinish(&imgtmp[i]); }
    return(0);
}


int FHPosterize(FHImg *src, FHImg *dst, int grad){
    int     i;
    int     diff = 256 / grad;
    uchar   lut[256];

    if ( !(2<grad && grad<256) ){ return(-1); }
    for ( i=0 ; i<256 ; i++ ){ lut[i] = (uchar)((int)(i / diff) * diff); }
    return(FHLUT(src, dst, lut));
}


int FHMaxRGB(FHImg *src, FHImg *dst){
    int     i, j;
    int     maxid = 0, minid = 0;
    FHImg   imgtmp;

    FHImgInit(&imgtmp);
    FHImgOpen(&imgtmp, src->h, src->w, src->ch);
    if ( src->ch==1 ){
        for ( i=1 ; i<src->size ; i++ ){
            if ( src->d[maxid]<src->d[i] ){ maxid = i; }
        }
        for ( i=1 ; i<src->size ; i++ ){
            if ( src->d[minid]>src->d[i] ){ minid = i; }
        }
        for ( i=0 ; i<src->size ; i++ ){
            imgtmp.d[i] = (float)(src->d[i]-src->d[minid]) / \
                          (src->d[maxid]-src->d[minid]) * 255.0;
        }
    }
    else if ( src->ch==3 ){
        for ( j=0 ; j<3 ; j++ ){
            maxid = minid = j;
            for ( i=j+3 ; i<src->size ; i+=3 ){
                if ( src->d[maxid]<src->d[i] ){ maxid = i; }
            }
            for ( i=j+3 ; i<src->size ; i+=3 ){
                if ( src->d[minid]>src->d[i] ){ minid = i; }
            }
            for ( i=j ; i<src->size ; i+=3 ){
                imgtmp.d[i] = (float)(src->d[i]-src->d[minid]) / \
                              (src->d[maxid]-src->d[minid]) * 255.0;
            }
        }
    }
    FHImgMv(&imgtmp, dst);
    FHImgFinish(&imgtmp);
    return(0);
}
