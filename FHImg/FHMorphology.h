/* --------------------------------------------------------
 * FHMorphology.h
 *      Written by F.Horie.
 * -------------------------------------------------------- */


#ifndef     FHMORPHOLOGY_H
#define     FHMORPHOLOGY_H


#include    "FHImg.h"


/* --------------------------------------------------------
 * Morphology operation (FHImgMorphology.cpp)
 * -------------------------------------------------------- */

extern int      FHMinkowskiAdd3(FHImg *src, FHImg *dst, int kernel[][3]);
extern int      FHMinkowskiPro3(FHImg *src, FHImg *dst, int kernel[][3]);
extern int      FHMinkowskiAdd5(FHImg *src, FHImg *dst, int kernel[][5]);
extern int      FHMinkowskiPro5(FHImg *src, FHImg *dst, int kernel[][5]);
extern int      FHErosion3(FHImg *src, FHImg *dst);
extern int      FHDilation3(FHImg *src, FHImg *dst);
extern int      FHErosion5(FHImg *src, FHImg *dst);
extern int      FHDilation5(FHImg *src, FHImg *dst);
extern int      FHImgOpening3(FHImg *src, FHImg *dst);
extern int      FHImgClosing3(FHImg *src, FHImg *dst);
extern int      FHOCFilter3(FHImg *src, FHImg *dst);
extern int      FHCOFilter3(FHImg *src, FHImg *dst);


#endif      /* FHMORPHOLOGY_H */
