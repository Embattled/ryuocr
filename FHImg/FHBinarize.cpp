/* --------------------------------------------------------
 * FHBinarizea.cpp
 *      Written by F.Horie.
 * -------------------------------------------------------- */


#include    "FHImg.h"
#include    "FHRGB.h"


/* --------------------------------------------------------
 * Caltulation of Otsu threshold
 * -------------------------------------------------------- */

uchar FHOtsuThresh(uchar *src, int size){
    int     i, j;
    int     histgram[256] = {0};
    int     sum = 0;
    uchar   result = 0;
    int     min_no = 0, max_no = 256;
    long    lbuf = 0;
    float   average, average1, average2;
    float   max = 0.0;
    int     count1, count2;
    float   breakup1, breakup2;
    float   class1, class2;
    float   tmp;

    for ( i=0 ; i<size ; i++ ){ histgram[src[i]]++; sum += src[i]; }
    for ( i=0 ; i<256 ; i++ ){ 
        if ( histgram[i]!=0 ){ min_no = i; break; } }
    for ( i=256-1 ; i>=0 ; i-- ){
        if ( histgram[i]!=0 ){ max_no = i+1; break; } }

    average = (float)sum / (float)size;
    average1 = average2 = 0;
    count1 = count2 = 0;
    lbuf = 0;
    breakup1 = breakup2 = 0.0;	

    for ( i=min_no ; i<max_no ; i++ ){
        breakup1 = breakup2 = 0.0;

        count1 += histgram[i];

        lbuf += histgram[i] * i;

        average1 = (float)lbuf / (float)count1;
        for( j=min_no ; j<i ; j++ ){
            breakup1 += ((float)j - average1) * \
                        ((float)j - average1) * (float)histgram[j]; }

        count2 = size - count1;

        average2 = (float)(sum-lbuf) / (float)count2;
        for ( j=i ; j<max_no ; j++ ){
            breakup2 += ((float)j - average2) * \
                        ((float)j - average2) * (float)histgram[j]; }

        class1 = breakup1 + breakup2;
        class2 = (float)count1 * (average1 - average) * (average1 - average) + \
                 (float)count2 * (average2 - average) * (average2 - average);

        tmp = class2 / class1;

        if ( max<tmp ){ max = tmp; result = (uchar)i; }
    }
    return(result);
}


/* --------------------------------------------------------
 * Binarization
 * -------------------------------------------------------- */

int FHBinarize(FHImg *src, FHImg *dst, uchar thresh){
    int     i;
    uchar   lut[256];

    if ( src->ch==3 ){ return(-1); }
    for ( i=0 ; i<thresh ; i++ ){ lut[i] = 0; }
    for ( i=thresh ; i<256 ; i++ ){ lut[i] = 255; }

    return(FHLUT(src, dst, lut));
}


/* --------------------------------------------------------
 * Binarization by Otsu method
 * -------------------------------------------------------- */

int FHBinarizeOtsu(FHImg *src, FHImg *dst){
    uchar   thresh;

    if ( src->ch==3 ){ return(-1); }
    thresh = FHOtsuThresh(src->d, src->size);
    return(FHBinarize(src, dst, thresh));
}


/* --------------------------------------------------------
 * Check whether a image is binarized
 * If a image is binarized, return 1,
 * else return 0.
 * -------------------------------------------------------- */

int FHIsBinarized(FHImg *src){
    int     i;

    if ( src->ch==3 ){ return(-1); }
    for ( i=0 ; i<src->size ; i++ ){
        if ( src->d[i]!=0 && src->d[i]!=255 ){ return(0); }
    }
    return(1);
}
