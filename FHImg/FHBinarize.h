/* --------------------------------------------------------
 * FHBinarize.h
 *      Written by F.Horie.
 * -------------------------------------------------------- */


#ifndef     FHBINARIZE_H
#define     FHBINARIZE_H


#include    "FHImg.h"


/* --------------------------------------------------------
 * Binarizing (FHBinarizea.cpp)
 * -------------------------------------------------------- */

extern uchar    FHOtsuThresh(uchar *src, int size);
extern int      FHBinarize(FHImg *src, FHImg *dst, uchar thresh);
extern int      FHBinarizeOtsu(FHImg *src, FHImg *dst);
extern int      FHIsBinarized(FHImg *src);


#endif      /* FHBINARIZE_H */
