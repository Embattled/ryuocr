/* --------------------------------------------------------
 * FHIDic.cpp
 *      Written by F.Horie.
 * -------------------------------------------------------- */


#include    <memory.h>
#include    "FHRnd.h"
#include    "FHIDic.h"
#include    "FHImgIO.h"


/* --------------------------------------------------------
 * Open / Close
 * -------------------------------------------------------- */

// void FHIDicInit(FHIDic *idic){
//     idic->num = 0;
//     idic->label = 0;
//     idic->img = 0;

// }


// int FHIDicOpen(FHIDic *idic, int num){
//     int     i;
//     if ( num<=0 ){ return(-1); }
//     if ( idic->num==num ){ return(0); }
//     FHIDicFinish(idic);
//     idic->img = FHMalloc(FHImg, num);
//     if ( idic->img==0 ){ FHIDicFinish(idic); return(-1); }
//     for ( i=0 ; i<num ; i++ ){ FHImgInit(&idic->img[i]); }
//     idic->label = FHMalloc(int, num);
//     if ( idic->label==0 ){ FHIDicFinish(idic); return(-1); }
//     idic->num = num;
//     return(0);
// }


// void FHIDicClose(FHIDic *idic){
//     int     i;
//     if ( idic->img ){ 
//         for ( i=0 ; i<idic->num ; i++ ){ FHImgFinish(&idic->img[i]); }
//         FHFree(idic->img);
//     }
//     FHFree(idic->label);
//     idic->num = 0;
//     return;
// }


// void FHIDicFinish(FHIDic *idic){
//     FHIDicClose(idic);
// }


/* --------------------------------------------------------
 * Move
 * -------------------------------------------------------- */

void FHIDicMv(FHIDic *src, FHIDic *dst){
    if ( src==dst ){ return; }
    FHIDicFinish(dst);
    *dst = *src;
    src->img = 0;
    src->label = 0;
    FHIDicFinish(src);
    return;
}


/* --------------------------------------------------------
 * Copy
 * -------------------------------------------------------- */

int FHIDicCp(FHIDic *src, FHIDic *dst){
    int     i;
    if ( src==dst ){ return(0); }
    if ( FHIDicOpen(dst, src->num) ){ return(-1); }
    for ( i=0 ; i<src->num ; i++ ){
        if ( FHImgCp(&src->img[i], &dst->img[i]) ){ return(-1); }
        dst->label[i] = src->label[i];
    }
    return(0);
}


/* --------------------------------------------------------
 * Joint
 * -------------------------------------------------------- */

int FHIDicJoint(FHIDic *src1, FHIDic *src2, FHIDic *dst){
    int     i;

    if ( FHIDicOpen(dst, src1->num+src2->num) ){ return(-1); }

    for ( i=0 ; i<src1->num ; i++ ){
        FHImgCp(&src1->img[i], &dst->img[i]);
        dst->label[i] = src1->label[i];
    }

    for ( i=0 ; i<src2->num ; i++ ){
        FHImgCp(&src2->img[i], &dst->img[i+src1->num]);
        dst->label[i+src1->num] = src2->label[i];
    }

    return(0);
}


/* --------------------------------------------------------
 * Bootstrap selection
 * -------------------------------------------------------- */

int FHIDicBootstrap(FHIDic *src, FHIDic *dst, int boot_num){
    int     i;
    int     itmp;
    FHIDic  idictmp;

    FHIDicInit(&idictmp);
    if ( FHIDicOpen(&idictmp, boot_num) ){ return(-1); }
    for ( i=0 ; i<idictmp.num ; i++ ){
        itmp = FHRndI(0, src->num-1);
        if ( FHImgCp(&src->img[itmp], &idictmp.img[i]) ){ return(-1); }
        idictmp.label[i] = src->label[itmp];
    }
    FHIDicMv(&idictmp, dst);
    FHIDicFinish(&idictmp);
    return(0);
}


/* --------------------------------------------------------
 * I/O
 * -------------------------------------------------------- */

int FHIDicRead(FHIDic *idic, FILE *fp){
    int     i;
    int     num0;
    if ( 1!=fread(&num0, sizeof(int), 1, fp) ){ return(-1); }
    if ( FHIDicOpen(idic, num0) ){ return(-1); }
    for ( i=0 ; i<num0 ; i++ ){
        if ( 1!=fread(&idic->label[i], sizeof(int), 1, fp) ){ \
            FHIDicFinish(idic); return(-1); }
        if ( FHImgRead(&idic->img[i], fp) ){ FHIDicFinish(idic); return(-1); }
    }
    return(0);
}


int FHIDicWrite(FHIDic *idic, FILE *fp){
    int     i;
    if ( 1!=fwrite(&idic->num, sizeof(int), 1, fp) ){ return(-1); }
    for ( i=0 ; i<idic->num ; i++ ){
        if ( 1!=fwrite(&idic->label[i], sizeof(int), 1, fp) ){ return(-1); }
        if ( FHImgWrite(&idic->img[i], fp) ){ return(-1); }
    }
    return(0);
}


int FHIDicLoad(FHIDic *idic, const char *fname){
    FILE    *fp = fopen(fname, "rb");
    if ( NULL==fp ){ return(-1); }
    if ( FHIDicRead(idic, fp) ){ fclose(fp); return(-1); }
    if ( EOF==fclose(fp) ){ return(-1); }
    return(0);
}


int FHIDicSave(FHIDic *idic, const char *fname){
    FILE    *fp = fopen(fname, "wb");
    if ( NULL==fp ){ return(-1); }
    if ( FHIDicWrite(idic, fp) ){ fclose(fp); return(-1); }
    if ( EOF==fclose(fp) ){ return(-1); }
    return(0);
}
