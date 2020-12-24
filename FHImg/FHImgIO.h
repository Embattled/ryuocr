/* --------------------------------------------------------
 * FHImgIO.h
 *      Written by F.Horie.
 * -------------------------------------------------------- */


#ifndef     FHIMGIO_H
#define     FHIMGIO_H


#include    <stdio.h>
#include    "FHImg.h"


extern int      FHImgRead(FHImg *img, FILE *fp);
extern int      FHImgWrite(FHImg *img, FILE *fp);
extern int      FHImgLoad(FHImg *img, const char *fname);
extern int      FHImgSave(FHImg *img, const char *fname);


#endif      /* FHIMGIO_H */
