/* --------------------------------------------------------
 * FHImg.h - Tiny image processing library -
 *      Written by F.Horie.
 * --------------------------------------------------------
 * Introduction
 *      This library includes above functions.
 *      - Binarizing:               FHBinarizea.cpp
 *      - Color change:             FHImgRGB.cpp
 *      - Filtering:                FHFilter.cpp
 *      - Affine transformation:    FHAffine.cpp
 *      - Resizing:                 FHResize.cpp
 *      - Morphology operations     FHMorphology.cpp
 *      - Triming:                  FHRect.cpp
 * -------------------------------------------------------- */


#ifndef     FHIMG_H
#define     FHIMG_H


#include    <assert.h>
#include    "FHDef.h"


/* --------------------------------------------------------
 * Image object class
 * -------------------------------------------------------- */

typedef struct FHImg {
    uchar   *d;
    int     h;
    int     w;
    int     ch;
    int     linesize;
    int     size;
    FHImg(void);
    virtual ~FHImg(void);
} FHImg;


/* --------------------------------------------------------
 * Basic functions (FHImg.cpp)
 * -------------------------------------------------------- */

#define         FHImgAt(img, y, x)  (img)->d[(y)*(img)->linesize+(x)]
extern void     FHImgInit(FHImg *img);
extern int      FHImgOpen(FHImg *img, int h, int w, int ch);
extern void     FHImgClose(FHImg *img);
extern void     FHImgFinish(FHImg *img);
extern uchar    *FHImgPtr(FHImg *img, int y);
extern int      FHImgCp(FHImg *src, FHImg *dst);
extern void     FHImgMv(FHImg *src, FHImg *dst);
extern void     FHImgSubAll(FHImg *img, int x);
extern uchar    FHImgInterp(FHImg *img, float y, float x, int color);
extern int      FHImgSum(FHImg *img);
extern void     FHImgPrint(FHImg *img);
extern int      FHImgCheckSize(FHImg *img1, FHImg *img2);


inline FHImg :: FHImg(void){ FHImgInit(this); }
inline FHImg :: ~FHImg(void){ assert(d==0); }


#endif      /* FHIMG_H */
