/* --------------------------------------------------------
 * FHFilter.h
 *      Written by F.Horie.
 * -------------------------------------------------------- */


#ifndef     FHFILTER_H
#define     FHFILTER_H


#include    "FHImg.h"


/* --------------------------------------------------------
 * Filtering (FHFilter.cpp)
 * -------------------------------------------------------- */

extern int      FHImgMagnitude(FHImg *src1, FHImg *src2, FHImg *dst);
extern int      FHFilter3(FHImg *src, FHImg *dst, float kernel[][3]);
extern int      FHFilter5(FHImg *src, FHImg *dst, float kernel[][5]);
extern int      FHSobelEx(FHImg *src, FHImg *dst, float sobelv[][3], float sobelh[][3]);
extern int      FHSobel(FHImg *src, FHImg *dst);
extern int      FHLaplacian(FHImg *src, FHImg *dst);
extern int      FHMeanFilter3(FHImg *src, FHImg *dst);
extern int      FHMeanFilter5(FHImg *src, FHImg *dst);
extern int      FHGaussianFilter3(FHImg *src, FHImg *dst, float sigma);
inline int      FHGaussianFilter3(FHImg *src, FHImg *dst){
    return(FHGaussianFilter3(src, dst, 1.0));
}
extern int      FHGaussianFilter5(FHImg *src, FHImg *dst);
extern int      FHGaussianFilter5(FHImg *src, FHImg *dst, float sigma);
extern int      FHMedianFilter(FHImg *src, FHImg *dst);
extern int      FHImgPadding(FHImg *src, FHImg *dst, int type);
extern int      FHMeanPooling(FHImg *src, FHImg *dst);
extern int      FHMaxPooling(FHImg *src, FHImg *dst);


#endif      /* FHFILTER_H */
