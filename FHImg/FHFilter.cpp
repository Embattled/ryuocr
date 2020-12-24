/* --------------------------------------------------------
 * FHImg/filter.cpp
 *      Written by F.Horie.
 * -------------------------------------------------------- */


#include    <math.h>
#include    <assert.h>
#include    "FHImg.h"
#include    "FHSort.h"


/* --------------------------------------------------------
 * Padding
 * -------------------------------------------------------- */

int FHImgPadding(FHImg *src, FHImg *dst, int type){
    int     y, i;
    int     ch = src->ch;
    FHImg   imgtmp;

    FHImgInit(&imgtmp);
    if ( FHImgOpen(&imgtmp, src->h+2, src->w+2, src->ch) ){ return(-1); }
    for ( i=0 ; i<imgtmp.size ; i++ ){ imgtmp.d[i] = 0; }
    for ( y=1 ; y<imgtmp.h-1 ; y++ ){
        uchar   *sp = FHImgPtr(src, y-1);
        uchar   *dp = FHImgPtr(&imgtmp, y);
        if ( NULL==memcpy(dp+ch, sp, imgtmp.linesize-2*ch) ){ return(-1); }
    }
    if ( type==1 ){
        /* y==0 */
        uchar   *p1 = FHImgPtr(&imgtmp, 0);
        uchar   *p2 = FHImgPtr(&imgtmp, 1);
        if ( NULL==memcpy(p1, p2, imgtmp.linesize) ){ return(-1); }
        /* y==imgtmp.h-1 */
        p1 = FHImgPtr(&imgtmp, imgtmp.h-1);
        p2 = FHImgPtr(&imgtmp, imgtmp.h-2);
        if ( NULL==memcpy(p1, p2, imgtmp.linesize) ){ return(-1); }
        /* x==0 and x==imgtmp.w-1 */
        for ( y=0 ; y<imgtmp.h ; y++ ){ 
            for ( i=0 ; i<ch ; i++ ){
                imgtmp.d[y*imgtmp.linesize+i] = imgtmp.d[y*imgtmp.linesize+ch+i]; 
                imgtmp.d[(y+1)*imgtmp.linesize-ch+i] = \
                    imgtmp.d[(y+1)*imgtmp.linesize-2*ch+i]; 
            }
        }
    }
    FHImgMv(&imgtmp, dst);
    FHImgFinish(&imgtmp);
    return(0);
}


/* --------------------------------------------------------
 * Image Filter 
 * -------------------------------------------------------- */

int FHFilter3(FHImg *src, FHImg *dst, float kernel[][3]){
    int     x, y;
    int     ch = src->ch;
    int     val;
    float   g;
    uchar   *dp, *p1, *p2, *p3;
    FHImg   srcimg, dstimg;

    FHImgInit(&srcimg);
    FHImgInit(&dstimg);
    if ( FHImgOpen(&dstimg, src->h, src->w, src->ch) ){ return(-1); }
    if ( FHImgPadding(src, &srcimg, 1) ){ return(-1); }

    for ( y=1 ; y<srcimg.h-1 ; y++ ){
        dp = FHImgPtr(&dstimg, y-1);
        p1 = FHImgPtr(&srcimg, y-1);
        p2 = FHImgPtr(&srcimg, y  );
        p3 = FHImgPtr(&srcimg, y+1);
        for ( x=ch ; x<srcimg.linesize-ch ; x++ ){
            g = 0;
            g += p1[x-ch] * kernel[0][0];
            g += p1[x   ] * kernel[0][1];
            g += p1[x+ch] * kernel[0][2];
            g += p2[x-ch] * kernel[1][0];
            g += p2[x   ] * kernel[1][1];
            g += p2[x+ch] * kernel[1][2];
            g += p3[x-ch] * kernel[2][0];
            g += p3[x   ] * kernel[2][1];
            g += p3[x+ch] * kernel[2][2];
            val = (int)(g+0.5);
            if ( val<0 ){ val = 0; }else if ( val>255 ){ val = 255; }
            dp[x-ch] = (uchar)val;
        }
    }

    FHImgMv(&dstimg, dst);
    FHImgFinish(&srcimg);
    FHImgFinish(&dstimg);
    return(0);
}


int FHFilter5(FHImg *src, FHImg *dst, float kernel[][5]){
    int     x, y, dx, dy, val;
    float   g;
    int     ch = src->ch;
    FHImg   srcimg, dstimg;
    uchar   *dp, *sp[5];

    FHImgInit(&srcimg);
    FHImgInit(&dstimg);
    if ( FHImgOpen(&dstimg, src->h, src->w, src->ch) ){ return(-1); }

    if ( FHImgPadding(src, &srcimg, 1) ){ return(-1); }
    if ( FHImgPadding(&srcimg, &srcimg, 1) ){ return(-1); }
    for ( y=2 ; y<srcimg.h-2 ; y++ ){
        dp = FHImgPtr(&dstimg, y-2);
        sp[0] = FHImgPtr(&srcimg, y-2);
        sp[1] = FHImgPtr(&srcimg, y-1);
        sp[2] = FHImgPtr(&srcimg, y  );
        sp[3] = FHImgPtr(&srcimg, y+1);
        sp[4] = FHImgPtr(&srcimg, y+2);
        for ( x=ch*2 ; x<srcimg.linesize-ch*2 ; x++ ){
            g = 0;
            for ( dy=0 ; dy<5 ; dy++ ){
                for ( dx=0 ; dx<5 ; dx++ ){
                    g += sp[dy][x+(dx-2)*ch] * kernel[dy][dx];
                }
            }
            val = (int)g;
            if ( val<0 ){ val = 0; } if ( val>255 ){ val = 255; }
            dp[x-2*ch] = (uchar)val;
        }
    }
    FHImgMv(&dstimg, dst);

    FHImgClose(&dstimg);
    FHImgClose(&srcimg);
    return(0);
}


/* --------------------------------------------------------
 * Edge Detection
 * -------------------------------------------------------- */

int FHImgMagnitude(FHImg *src1, FHImg *src2, FHImg *dst){
    int     i;
    float   m;
    FHImg   dstimg;

    if ( FHImgCheckSize(src1, src2)!=1 ){ return(-1); }
    FHImgInit(&dstimg);
    FHImgOpen(&dstimg, src1->h, src1->w, src1->ch);
    for ( i=0 ; i<src1->size ; i++ ){
         m = hypotf(src1->d[i], src2->d[i]);
         if ( m<0 ){ m = 0; }
         if ( m>255 ){ m = 255; }
         dstimg.d[i] = (uchar)round(m);
    }

    FHImgMv(&dstimg, dst);
    FHImgFinish(&dstimg);
    return(0);
}


int FHSobelEx(FHImg *src, FHImg *dst, float sobelv[][3], float sobelh[][3]){
    int     x, y;
    int     ch = src->ch;
    float   inth = 0, intv = 0, g;
    uchar   *dp, *p1, *p2, *p3;
    FHImg   srcimg, dstimg;

    FHImgInit(&srcimg);
    FHImgInit(&dstimg);
    if ( FHImgOpen(&dstimg, src->h, src->w, src->ch) ){ return(-1); }

    if ( FHImgPadding(src, &srcimg, 1) ){ return(-1); }

    for ( y=1 ; y<srcimg.h-1 ; y++ ){
        dp = FHImgPtr(&dstimg, y-1);
        p1 = FHImgPtr(&srcimg, y-1);
        p2 = FHImgPtr(&srcimg, y  );
        p3 = FHImgPtr(&srcimg, y+1);
        for ( x=ch ; x<srcimg.linesize-ch ; x++ ){
            inth = intv = 0;
            inth += sobelh[0][0] * p1[x-ch];
            intv += sobelv[0][0] * p1[x-ch];
            inth += sobelh[0][1] * p1[x];
            intv += sobelv[0][1] * p1[x];
            inth += sobelh[0][2] * p1[x+ch];
            intv += sobelv[0][2] * p1[x+ch];
            inth += sobelh[1][0] * p2[x-ch];
            intv += sobelv[1][0] * p2[x-ch];
            inth += sobelh[1][1] * p2[x];
            intv += sobelv[1][1] * p2[x];
            inth += sobelh[1][2] * p2[x+ch];
            intv += sobelv[1][2] * p2[x+ch];
            inth += sobelh[2][0] * p3[x-ch];
            intv += sobelv[2][0] * p3[x-ch];
            inth += sobelh[2][1] * p3[x];
            intv += sobelv[2][1] * p3[x];
            inth += sobelh[2][2] * p3[x+ch];
            intv += sobelv[2][2] * p3[x+ch];
            g = (hypotf(inth, intv)+0.5f);
            if ( g<0 ){ g = 0; } if ( g>255 ){ g = 255; }
            dp[x-ch] = (uchar)g;
        }
    }
    FHImgMv(&dstimg, dst);
    FHImgFinish(&dstimg);
    FHImgFinish(&srcimg);
    return(0);
}


int FHSobel(FHImg *src, FHImg *dst){
    float   sobelh[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    float   sobelv[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };
    return(FHSobelEx(src, dst, sobelv, sobelh));
}


int FHLaplacian(FHImg *src, FHImg *dst){
    float   kernel[3][3] = {{1, 1, 1}, {1, -8, 1}, {1, 1, 1}};
    return(FHFilter3(src, dst, kernel));
}


/* --------------------------------------------------------
 * Gaussian Filter
 * -------------------------------------------------------- */

#if     0
void FHGaussianFilter3(FHImg *src, FHImg *dst){
    float   kernel[3][3] = {
        { 1.0/16, 2.0/16, 1.0/16 },
        { 2.0/16, 4.0/16, 2.0/16 },
        { 1.0/16, 2.0/16, 1.0/16 },
    };
    FHFilter3(src, dst, kernel);
}
#endif


int FHGaussianFilter3(FHImg *src, FHImg *dst, float sigma){
    int     i, j;
    int     x, y;
    float   kernel[3][3];
    float   sum = 0;

    if ( sigma==0 ){ return(FHImgCp(src, dst)); }

    for ( i=0 ; i<3 ; i++ ){
        for ( j=0 ; j<3 ; j++ ){
            x = j - 1; y = i - 1;
            sum += kernel[i][j] = 1.0f / (2.0f * M_PIF * FHSquare(sigma)) * \
                           expf(-(FHSquare((float)x)+FHSquare((float)y)) / \
                                   (2.0f * FHSquare(sigma)));
        }
    }

    for ( i=0 ; i<3 ; i++ ){
        for ( j=0 ; j<3 ; j++ ){
            kernel[i][j] /= sum;
        }
    }

    return(FHFilter3(src, dst, kernel));
}


#if     0
void FHGaussianFilter5_usual(FHImg *src, FHImg *dst){
    float   kernel[5][5] = {
        { 1.0/256, 4.0/256, 6.0/256, 4.0/256, 1.0/256 },
        { 4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256 },
        { 6.0/256, 24.0/256, 36.0/256, 24.0/256, 6.0/256 },
        { 4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256 },
        { 1.0/256, 4.0/256, 6.0/256, 4.0/256, 1.0/256 }
    };
    FHFilter5(src, dst, kernel);
}
#endif


int FHGaussianFilter5(FHImg *src, FHImg *dst, float sigma){
    int     i, j;
    int     x, y;
    float   kernel[5][5];
    float   sum = 0;

    for ( i=0 ; i<5 ; i++ ){
        for ( j=0 ; j<5 ; j++ ){
            x = j - 1; y = i - 1;
            sum += kernel[i][j] = 1.0f / (2.0f * M_PIF * FHSquare(sigma)) * \
                           expf(-(FHSquare((float)x)+FHSquare((float)y)) / \
                                   (2.0f * FHSquare(sigma)));
        }
    }

    for ( i=0 ; i<5 ; i++ ){
        for ( j=0 ; j<5 ; j++ ){
            kernel[i][j] /= sum;
        }
    }

    return(FHFilter5(src, dst, kernel));
}


/* --------------------------------------------------------
 * Mean Filter
 * -------------------------------------------------------- */

int FHMeanFilter3(FHImg *src, FHImg *dst){
    int     x, y;
    int     ch = src->ch;
    float   g;
    uchar   *dp, *p1, *p2, *p3;
    FHImg   srcimg, dstimg;

    FHImgInit(&srcimg);
    FHImgInit(&dstimg);
    if ( FHImgOpen(&dstimg, src->h, src->w, src->ch) ){ return(-1); }
    if ( FHImgPadding(src, &srcimg, 1) ){ return(-1); }

    for ( y=1 ; y<srcimg.h-1 ; y++ ){
        dp = FHImgPtr(&dstimg, y-1);
        p1 = FHImgPtr(&srcimg, y-1);
        p2 = FHImgPtr(&srcimg, y  );
        p3 = FHImgPtr(&srcimg, y+1);
        for ( x=ch ; x<srcimg.linesize-ch ; x++ ){
            g = 0;
            g += p1[x-ch];
            g += p1[x   ];
            g += p1[x+ch];
            g += p2[x-ch];
            g += p2[x   ];
            g += p2[x+ch];
            g += p3[x-ch];
            g += p3[x   ];
            g += p3[x+ch];
            dp[x-ch] = (uchar)((float)g / 9 + 0.5);
        }
    }

    FHImgMv(&dstimg, dst);
    FHImgFinish(&srcimg);
    FHImgFinish(&dstimg);
    return(0);
}


int FHMeanFilter5(FHImg *src, FHImg *dst){
    int     i, j;
    float   kernel[5][5];
    for ( i=0 ; i<5 ; i++ ){
        for ( j=0 ; j<5 ; j++ ){ kernel[i][j] = 1.0f / 25.0f; }
    }
    return(FHFilter5(src, dst, kernel));
}


/* --------------------------------------------------------
 * Median Filter
 * -------------------------------------------------------- */

int FHMedianFilter(FHImg *src, FHImg *dst){
    int     x, y;
    int     itmp[9];
    FHImg   imgtmp;
    uchar   *dp, *p1, *p2, *p3;
    int     ch = src->ch;

    FHImgInit(&imgtmp);
    if ( FHImgCp(src, &imgtmp) ){ return(-1); }

    for ( y=1 ; y<imgtmp.h-1 ; y++ ){
        dp = FHImgPtr(&imgtmp, y  );
        p1 = FHImgPtr(src, y-1);
        p2 = FHImgPtr(src, y  );
        p3 = FHImgPtr(src, y+1);
        for ( x=ch ; x<imgtmp.linesize-ch ; x++ ){
            itmp[0] = p1[x-ch];
            itmp[1] = p1[x];
            itmp[2] = p1[x+ch];
            itmp[3] = p2[x-ch];
            itmp[4] = p2[x];
            itmp[5] = p2[x+ch];
            itmp[6] = p3[x-ch];
            itmp[7] = p3[x];
            itmp[8] = p3[x+ch];
            FHSortI(itmp, 9);
            dp[x] = (uchar)itmp[4];
        }
    }
    FHImgMv(&imgtmp, dst);
    FHImgClose(&imgtmp);
    return(0);
}


/* --------------------------------------------------------
 * Pooling
 * -------------------------------------------------------- */

int FHMeanPooling(FHImg *src, FHImg *dst){
    int     y;
    int     ch = src->ch;
    FHImg   dstimg;

    FHImgInit(&dstimg);

    if ( src->h%2!=0 || src->w%2!=0 ){ return(-1); }
    if ( FHImgOpen(&dstimg, src->h/2, src->w/2, src->ch) ){ return(-1); }

    for ( y=0 ; y<src->h ; y+=2 ){
        int     x;
        uchar   *dp, *p1, *p2;
        dp = FHImgPtr(&dstimg, y/2);
        p1 = FHImgPtr(src, y  );
        p2 = FHImgPtr(src, y+1);
        for ( x=0 ; x<src->linesize ; x+=2*ch ){
            int     i;
            int     val;
            for ( i=0 ; i<ch ; i++ ){
                val = 0;
                val += p1[x+i];
                val += p2[x+i];
                val += p1[x+ch+i];
                val += p2[x+ch+i];
                val = (int)((float)val/4.0f+0.5);
                if ( val<0 ){ val = 0; } if ( val>255 ){ val = 255; }
                dp[x/2+i] = (uchar)val;
            }
        }
    }

    FHImgMv(&dstimg, dst);
    FHImgFinish(&dstimg);
    return(0);
}


int FHMaxPooling(FHImg *src, FHImg *dst){
    int     y;
    int     ch = src->ch;
    FHImg   dstimg;

    FHImgInit(&dstimg);

    if ( src->h%2!=0 || src->w%2!=0 ){ return(-1); }
    if ( FHImgOpen(&dstimg, src->h/2, src->w/2, src->ch) ){ return(-1); }

    for ( y=0 ; y<src->h ; y+=2 ){
        int     x;
        uchar   *dp, *p1, *p2;
        dp = FHImgPtr(&dstimg, y/2);
        p1 = FHImgPtr(src, y  );
        p2 = FHImgPtr(src, y+1);
        for ( x=0 ; x<src->linesize ; x+=2*ch ){
            int     i;
            int     val;
            for ( i=0 ; i<ch ; i++ ){
                val = 0;
                val = FHMax(val, p1[x+i]);
                val = FHMax(val, p2[x+i]);
                val = FHMax(val, p1[x+ch+i]);
                val = FHMax(val, p2[x+ch+i]);
                dp[x/2+i] = (uchar)val;
            }
        }
    }

    FHImgMv(&dstimg, dst);
    FHImgFinish(&dstimg);
    return(0);
}
