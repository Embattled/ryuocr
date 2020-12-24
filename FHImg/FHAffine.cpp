/* --------------------------------------------------------
 * FHAffine.cpp - Function group about affine transformation -
 *      Written by F.Horie.
 * -------------------------------------------------------- */


#include    <assert.h>
#include    <stdio.h>
#include    <math.h>
#include    <algorithm>

#include    "FHImg.h"


/* --------------------------------------------------------
 *  Calculation of median of edge
 *  This function calculates the median of the edge pixels of an image.
 *  The median value is inseted into ZERO space created by 
 * the affine transformation.
 *  ImgEdgeMedian1Ch: Calculation of the median value on 1 
 * channel images.
 *  ImgEdgeMedian3Ch: Calculation of the median value on 3 
 * channels images.
 * -------------------------------------------------------- */

static void ImgEdgeMedian1Ch(FHImg *src, uchar *mdata){
    int     i;
    int     esize = 2 * (src->w + src->h);
    int     offset;
    uchar   *edge = new uchar[esize];

    offset = 0;
    for ( i=0 ; i<src->linesize ; i++ ){ edge[i+offset] = FHImgAt(src, 0, i); }
    offset += src->w;
    for ( i=0 ; i<src->linesize ; i++ ){ edge[i+offset] = FHImgAt(src, src->h-1, i); }
    offset += src->w;
    for ( i=0 ; i<src->h ; i++ ){ edge[i+offset] = FHImgAt(src, i, 0); }
    offset += src->h;
    for ( i=0 ; i<src->h ; i++ ){ edge[i+offset] = FHImgAt(src, i, src->linesize-1); }
    std::sort(&edge[0], &edge[esize-1]);
    *mdata = (uchar)((edge[src->w+src->h] + edge[src->w+src->h+1]) / 2);

    delete []edge;
    return;
}



static void ImgEdgeMedian3Ch(FHImg *src, uchar *rdata, uchar *gdata, uchar *bdata){
    int     i;
    int     esize = 2 * (src->w + src->h);
    int     offset;
    int     color;
    uchar   *edge = new uchar[esize];

    offset = 0;
    color = 0;
    for ( i=color ; i<src->linesize ; i+=3 ){ edge[(i-color)/3+offset] = FHImgAt(src, 0, i); }
    offset += src->w;
    for ( i=color ; i<src->linesize ; i+=3 ){ edge[(i-color)/3+offset] = FHImgAt(src, src->h-1, i); }
    offset += src->w;
    for ( i=0 ; i<src->h ; i++ ){ edge[i+offset] = FHImgAt(src, i, color); }
    offset += src->h;
    for ( i=0 ; i<src->h ; i++ ){ edge[i+offset] = FHImgAt(src, i, src->linesize-3+color); }
    std::sort(&edge[0], &edge[esize-1]);
    *rdata = (uchar)((edge[src->w+src->h] + edge[src->w+src->h+1]) / 2);

    offset = 0;
    color = 1;
    for ( i=color ; i<src->linesize ; i+=3 ){ edge[(i-color)/3+offset] = FHImgAt(src, 0, i); }
    offset += src->w;
    for ( i=color ; i<src->linesize ; i+=3 ){ edge[(i-color)/3+offset] = FHImgAt(src, src->h-1, i); }
    offset += src->w;
    for ( i=0 ; i<src->h ; i++ ){ edge[i+offset] = FHImgAt(src, i, color); }
    offset += src->h;
    for ( i=0 ; i<src->h ; i++ ){ edge[i+offset] = FHImgAt(src, i, src->linesize-3+color); }
    std::sort(&edge[0], &edge[esize-1]);
    *gdata = (uchar)((edge[src->w+src->h] + edge[src->w+src->h+1]) / 2);

    offset = 0;
    color = 2;
    for ( i=color ; i<src->linesize ; i+=3 ){ edge[(i-color)/3+offset] = FHImgAt(src, 0, i); }
    offset += src->w;
    for ( i=color ; i<src->linesize ; i+=3 ){ edge[(i-color)/3+offset] = FHImgAt(src, src->h-1, i); }
    offset += src->w;
    for ( i=0 ; i<src->h ; i++ ){ edge[i+offset] = FHImgAt(src, i, color); }
    offset += src->h;
    for ( i=0 ; i<src->h ; i++ ){ edge[i+offset] = FHImgAt(src, i, src->linesize-3+color); }
    std::sort(&edge[0], &edge[esize-1]);
    *bdata = (uchar)((edge[src->w+src->h] + edge[src->w+src->h+1]) / 2);

    delete []edge;
    return;
}


/* --------------------------------------------------------
 * 2D-Affine
 * -------------------------------------------------------- */

static void InverseMat3(float src[][3], float dst[][3]){
    int     i, j, k;
    int     n = 3;
    float   fbuf = 0;

    for ( i=0 ; i<n ; i++ ){
        for ( j=0 ; j<n ; j++ ){ dst[i][j] = (i==j) ? 1.0:0.0; }
    }

    for ( i=0 ; i<n ; i++ ){
        fbuf = 1 / src[i][i];
        for ( j=0 ; j<n ; j++ ){
            src[i][j] *= fbuf;
            dst[i][j] *= fbuf;
        }
        for ( j=0 ; j<n ; j++ ){
            if ( i!=j ){
                fbuf = src[j][i];
                for ( k=0 ; k<n ; k++ ){
                    src[j][k] -= src[i][k] * fbuf;
                    dst[j][k] -= dst[i][k] * fbuf;
                }
            }
        }
    }
    return;
}


static void InnerProductMV3(float mat[][3], float vec[], float dst[]){
    int     i, j;
    int     n = 3;

    for ( i=0 ; i<n ; i++ ){
        dst[i] = 0;
        for ( j=0 ; j<n ; j++ ){ dst[i] += mat[i][j] * vec[j]; }
    }
    return;
}


static void InnerProductMM3(float src1[][3], float src2[][3], float dst[][3]){
    int     i, j, k;
    int     n = 3;

    for ( i=0 ; i<n ; i++ ){
        for ( j=0 ; j<n ; j++ ){
            dst[i][j] = 0;
            for ( k=0 ; k<n ; k++ ){ dst[i][j] += src1[i][k] * src2[k][j]; }
        }
    }
    return;
}


/* --------------------------------------------------------
 * Calculation of center affine matrix
 * -------------------------------------------------------- */

void FHCenterMat2D(float src[][2], float dst[][3], int h, int w){
    float   mattmp1[3][3] = {
        {1, 0, (float)w/2.0f},
        {0, 1, (float)h/2.0f},
        {0, 0, 1}
    };
    float   mattmp2[3][3] = {
        {1, 0, -(float)w/2.0f},
        {0, 1, -(float)h/2.0f},
        {0, 0, 1}
    };
    float   srctmp[3][3] = {
        {src[0][0], src[0][1],  0},
        {src[1][0], src[1][1],  0},
        {0,         0,          1}
    };
    float   dsttmp1[3][3];
    float   dsttmp2[3][3];

    InnerProductMM3(mattmp1, srctmp, dsttmp1);
    InnerProductMM3(dsttmp1, mattmp2, dsttmp2);
    InverseMat3(dsttmp2, dst);
    return;
}


/* --------------------------------------------------------
 *  Affine transformation without completion of the
 * background color (bilinear interpolation)
 * -------------------------------------------------------- */

static int FHAffineBG(FHImg *src, FHImg *dst, float mat[][3], uchar bgcolor[]){
    int     x, y, i;
    int     ch = src->ch;
    float   dx, dy;
    FHImg   imgtmp;
    uchar   *p;
    float   vec0[3], vec1[3];

    if ( src->d==0 ){ return(-1); }

    FHImgInit(&imgtmp);
    if ( FHImgCp(src, &imgtmp) ){ return(-1); }
    vec0[2] = 1.0f;

    for ( y=0 ; y<imgtmp.h ; y++ ){
        p = FHImgPtr(&imgtmp, y);
        for ( x=0 ; x<imgtmp.w ; x++ ){
            vec0[0] = (float)x; vec0[1] = (float)y;
            InnerProductMV3(mat, vec0, vec1);
            dx = vec1[0]; dy = vec1[1];

            if ( (int)dy==src->h-1 ){ dy = (float)src->h-2.0f; }
            if ( (int)dx==src->w-1 ){ dx = (float)src->w-2.0f; }
            if ( dx<0 || dy<0 || src->h-1<dy || src->w-1<dx ){ 
                for ( i=0 ; i<ch ; i++ ){ p[ch*x+i] = bgcolor[i]; }
            }
            else{
                for ( i=0 ; i<ch ; i++ ){ 
                    p[ch*x+i] = FHImgInterp(src, dy, dx, i); 
                }
            }
        }
    }
    FHImgMv(&imgtmp, dst);
    FHImgFinish(&imgtmp);
    return(0);
}


/* --------------------------------------------------------
 * Affine transformation with background taken into account
 * -------------------------------------------------------- */

int FHAffine(FHImg *src, FHImg *dst, float mat[][3]){
    uchar   bgcolor[3] = {0};
    if ( src->ch==3 ){ 
        ImgEdgeMedian3Ch(src, &bgcolor[0], &bgcolor[1], &bgcolor[2]); 
    }
    else if ( src->ch==1 ){ ImgEdgeMedian1Ch(src, &bgcolor[0]); }
    else{ return(-1); }
    if ( FHAffineBG(src, dst, mat, bgcolor) ){ return(-1); }
    return(0);
}


/* --------------------------------------------------------
 * Affine transformation using the center affine matrix.
 * -------------------------------------------------------- */

int FHCenterAffine(FHImg *src, FHImg *dst, float mat[][2]){
    float   kernel[3][3];
    FHCenterMat2D(mat, kernel, src->h, src->w);
    return(FHAffine(src, dst, kernel));
}


static inline float degree2rad(float x){ return(M_PIF*x/180.0f); }


/* --------------------------------------------------------
 * Extra functions
 * -------------------------------------------------------- */

int FHSkewH(FHImg *src, FHImg *dst, float angle){
    float   tana = tanf(degree2rad(angle));
    float   mat[2][2] = {
        {1, tana},
        {0, 1}
    };
    return(FHCenterAffine(src, dst, mat));
}


int FHSkewV(FHImg *src, FHImg *dst, float angle){
    float   tana = tanf(degree2rad(angle));
    float   mat[2][2] = {
        {1, 0},
        {tana, 1}
    };
    return(FHCenterAffine(src, dst, mat));
}


int FHRotate(FHImg *src, FHImg *dst, float angle){
    float   cosa = cosf(degree2rad(angle));
    float   sina = sinf(degree2rad(angle));
    float   mat[2][2] = {
        {cosa, sina},
        {-sina, cosa}
    };
    return(FHCenterAffine(src, dst, mat));
}


int FHAffineResize(FHImg *src, FHImg *dst, float xs, float ys){
    float   mat[2][2] = {
        {xs, 0},
        {0, ys}
    };
    return(FHCenterAffine(src, dst, mat));
}


/* --------------------------------------------------------
 * 3D-Affine
 * -------------------------------------------------------- */

static void InverseMat4(float src[][4], float dst[][4]){
    int     i, j, k;
    int     n = 4;
    float   fbuf = 0;

    for ( i=0 ; i<n ; i++ ){
        for ( j=0 ; j<n ; j++ ){ dst[i][j] = (i==j) ? 1.0:0.0; }
    }

    for ( i=0 ; i<n ; i++ ){
        fbuf = 1 / src[i][i];
        for ( j=0 ; j<n ; j++ ){
            src[i][j] *= fbuf;
            dst[i][j] *= fbuf;
        }
        for ( j=0 ; j<n ; j++ ){
            if ( i!=j ){
                fbuf = src[j][i];
                for ( k=0 ; k<n ; k++ ){
                    src[j][k] -= src[i][k] * fbuf;
                    dst[j][k] -= dst[i][k] * fbuf;
                }
            }
        }
    }
    return;
}


static void InnerProductMV4(float mat[][4], float vec[], float dst[]){
    int     i, j;
    int     n = 4;

    for ( i=0 ; i<n ; i++ ){
        dst[i] = 0;
        for ( j=0 ; j<n ; j++ ){ dst[i] += mat[i][j] * vec[j]; }
    }
    return;
}


static void InnerProductMM4(float src1[][4], float src2[][4], float dst[][4]){
    int     i, j, k;
    int     n = 4;

    for ( i=0 ; i<n ; i++ ){
        for ( j=0 ; j<n ; j++ ){
            dst[i][j] = 0;
            for ( k=0 ; k<n ; k++ ){ dst[i][j] += src1[i][k] * src2[k][j]; }
        }
    }
    return;
}


void FHImgCenterMat3D(float src[][3], float dst[][4], int h, int w){
    float   mattmp1[4][4] = {
        {1, 0, 0, (float)w/2.0f},
        {0, 1, 0, (float)h/2.0f},
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    };
    float   mattmp2[4][4] = {
        {1, 0, 0, -(float)w/2.0f},
        {0, 1, 0, -(float)h/2.0f},
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    };
    float   srctmp[4][4] = {
        {src[0][0], src[0][1],  src[0][2],  0},
        {src[1][0], src[1][1],  src[1][2],  0},
        {src[2][0], src[2][1],  src[2][2],  0},
        {0,         0,          0,          1}
    };
    float   dsttmp1[4][4];
    float   dsttmp2[4][4];

    InnerProductMM4(mattmp1, srctmp, dsttmp1);
    InnerProductMM4(dsttmp1, mattmp2, dsttmp2);
    InverseMat4(dsttmp2, dst);
    return;
}


int FHAffine3D(FHImg *src, FHImg *dst, float mat[][4], uchar bgcolor[]){
    int     x, y, i;
    int     ch = src->ch;
    float   dx, dy;
    FHImg   imgtmp;
    uchar   *p;
    float   vec0[4] = {0, 0, 1, 1}, vec1[4];

    FHImgInit(&imgtmp);
    if ( FHImgCp(src, &imgtmp) ){ return(-1); }

    for ( y=0 ; y<imgtmp.h ; y++ ){
        p = FHImgPtr(&imgtmp, y);
        for ( x=0 ; x<imgtmp.w ; x++ ){
            vec0[0] = (float)x; vec0[1] = (float)y;
            InnerProductMV4(mat, vec0, vec1);
            dx = vec1[0]/vec1[2]; dy = vec1[1]/vec1[2];

            if ( (int)dy==src->h-1 ){ dy = (float)src->h-2.0f; }
            if ( (int)dx==src->w-1 ){ dx = (float)src->w-2.0f; }

            if ( dx<0 || dy<0 || src->h-1<dy || src->w-1<dx ){ 
                for ( i=0 ; i<ch ; i++ ){ p[ch*x+i] = bgcolor[i]; }
            }
            else{
                for ( i=0 ; i<ch ; i++ ){ 
                    p[ch*x+i] = FHImgInterp(src, dy, dx, i); 
                }
            }
        }
    }

    FHImgMv(&imgtmp, dst);
    return(0);
}


int FHAffineMove(FHImg *src, FHImg *dst, float dx, float dy){
    float   mat[2][3] = {
        {1, 0, -dx},
        {0, 1, -dy}
    };
    return(FHAffine(src, dst, mat));
}
