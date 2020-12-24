/* --------------------------------------------------------
 * FHRGB.h
 *      Written by F.Horie.
 * -------------------------------------------------------- */


#ifndef     FHRGB_H
#define     FHRGB_H


#include    "FHImg.h"


/* --------------------------------------------------------
 * Color changer (FHImgRGB.cpp)
 * -------------------------------------------------------- */

extern uchar    FHToGrayPix(uchar x1, uchar x2, uchar x3);
extern int      FHLUT(FHImg *src, FHImg *dst, uchar *lut);
extern int      FHToGray(FHImg *src, FHImg *dst);
extern int      FHToRGB(FHImg *src, FHImg *dst);
extern int      FHChangeLight(FHImg *src, FHImg *dst, int light);
extern int      FHImgInverse(FHImg *src, FHImg *dst);
extern int      FHExtractRGB(FHImg *src, FHImg *dst, int ch);
extern int      FHMaxVarColor(FHImg *src, FHImg *dst);
extern int      FHPosterize(FHImg *src, FHImg *dst, int grad);
extern int      FHMaxRGB(FHImg *src, FHImg *dst);
extern int      FHGammaCorrection(FHImg *src, FHImg *dst, float gamma);


#endif      /* FHRGB_H */
