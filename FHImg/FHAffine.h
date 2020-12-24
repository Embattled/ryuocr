/* --------------------------------------------------------
 * FHAffine.h - Function group about affine transformation -
 *      Written by F.Horie.
 * -------------------------------------------------------- */


#ifndef     FHAFFINE_H
#define     FHAFFINE_H


#include    "FHImg.h"


/* --------------------------------------------------------
 * Affine transformation (FHAffine.cpp)
 * -------------------------------------------------------- */

extern int      FHAffine(FHImg *src, FHImg *dst, float mat[][3]);
extern int      FHCenterAffine(FHImg *src, FHImg *dst, float mat[][2]);
extern int      FHSkewH(FHImg *src, FHImg *dst, float angle);
extern int      FHSkewV(FHImg *src, FHImg *dst, float angle);
extern int      FHRotate(FHImg *src, FHImg *dst, float angle);
extern int      FHAffineMove(FHImg *src, FHImg *dst, float dx, float dy);
extern int      FHAffineResize(FHImg *src, FHImg *dst, float xs, float ys);
extern int      FHAffine3D(FHImg *src, FHImg *dst, float mat[][4], uchar bgcolor[]);


#endif      /* FHAFFINE_H */
