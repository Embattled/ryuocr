/* --------------------------------------------------------
 * FHIDic.h
 *      Written by F.Horie.
 * -------------------------------------------------------- */


#ifndef     FHIDIC_H
#define     FHIDIC_H


#include    <stdio.h>
#include    <assert.h>
#include    "FHImg.h"


/* --------------------------------------------------------
 * image dictionary structure
 * -------------------------------------------------------- */

typedef struct FHIDic {
    FHImg   *img;
    int     *label;
    int     num;
    FHIDic(void);
    virtual ~FHIDic(void);
} FHIDic;


extern void     FHIDicInit(FHIDic *idic);
extern int      FHIDicOpen(FHIDic *idic, int num);
extern void     FHIDicClose(FHIDic *idic);
extern void     FHIDicFinish(FHIDic *idic);
extern int      FHIDicCp(FHIDic *src, FHIDic *dst);
extern void     FHIDicMv(FHIDic *src, FHIDic *dst);
extern int      FHIDicBootstrap(FHIDic *src, FHIDic *dst, int num_boot);
extern int      FHIDicJoint(FHIDic *src1, FHIDic *src2, FHIDic *dst);
extern int      FHIDicRead(FHIDic *idic, FILE *fp);
extern int      FHIDicWrite(FHIDic *idic, FILE *fp);
extern int      FHIDicLoad(FHIDic *idic, const char *fname);
extern int      FHIDicSave(FHIDic *idic, const char *fname);


inline FHIDic :: FHIDic(void){ FHIDicInit(this); }
inline FHIDic :: ~FHIDic(void){ assert(img==0 && label==0); }


#endif      /* FHIDIC_H */
