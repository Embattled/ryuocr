/* --------------------------------------------------------
 * FHMorphology.cpp
 *      Written by F.Horie.
 * -------------------------------------------------------- */


#include    <stdio.h>
#include    "FHImg.h"


/* --------------------------------------------------------
 * Morphology Operation
 * -------------------------------------------------------- */

int FHMinkowskiAdd3(FHImg *src, FHImg *dst, int kernel[][3]){
    int     x, y;
    int     bufint;
    FHImg   imgtmp;
    uchar   *dp, *p1, *p2, *p3;
    int     ch = src->ch;


    for ( y=0 ; y<3 ; y++ ){
        for ( x=0 ; x<3 ; x++ ){
            if ( kernel[y][x]!=0 && kernel[y][x]!=1 ){ return(-1); }
        }
    }

    FHImgInit(&imgtmp);
    if ( FHImgCp(src, &imgtmp) ){ return(-1); }

    for ( y=1 ; y<imgtmp.h-1 ; y++ ){
        dp = FHImgPtr(&imgtmp, y  );
        p1 = FHImgPtr(src, y-1);
        p2 = FHImgPtr(src, y  );
        p3 = FHImgPtr(src, y+1);
        for ( x=ch ; x<imgtmp.linesize-ch ; x++ ){
            bufint = 0;
            bufint = FHMax(p1[x-ch]*kernel[0][0], bufint);
            bufint = FHMax(p1[x   ]*kernel[0][1], bufint);
            bufint = FHMax(p1[x+ch]*kernel[0][2], bufint);
            bufint = FHMax(p2[x-ch]*kernel[1][0], bufint);
            bufint = FHMax(p2[x   ]*kernel[1][1], bufint);
            bufint = FHMax(p2[x+ch]*kernel[1][2], bufint);
            bufint = FHMax(p3[x-ch]*kernel[2][0], bufint);
            bufint = FHMax(p3[x   ]*kernel[2][1], bufint);
            bufint = FHMax(p3[x+ch]*kernel[2][2], bufint);
            dp[x] = (uchar)bufint;
        }
    }

    FHImgMv(&imgtmp, dst);
    FHImgClose(&imgtmp);
    return(0);
}


int FHMinkowskiAdd5(FHImg *src, FHImg *dst, int kernel[][5]){
    int     x, y;
    FHImg   imgtmp;
    uchar   *dp;
    int     ch = src->ch;

    for ( y=0 ; y<5 ; y++ ){
        for ( x=0 ; x<5 ; x++ ){
            if ( kernel[y][x]!=0 && kernel[y][x]!=1 ){ return(-1); }
        }
    }

    FHImgInit(&imgtmp);
    if ( FHImgCp(src, &imgtmp) ){ return(-1); }

    for ( y=2 ; y<imgtmp.h-2 ; y++ ){
        dp = FHImgPtr(&imgtmp, y  );
        for ( x=ch*2 ; x<imgtmp.linesize-ch*2 ; x++ ){
            int     i, j;
            int     bufint = 0;
            for ( i=0 ; i<4 ; i++ ){
                for ( j=0 ; j<4 ; j++ ){
                    bufint = FHMax(FHImgAt(src, y-(i-2), x-ch*(j-2))*kernel[i][j], bufint);
                }
            }
            dp[x] = (uchar)bufint;
        }
    }

    FHImgMv(&imgtmp, dst);
    FHImgClose(&imgtmp);
    return(0);
}


int FHMinkowskiAdd(FHImg *src, FHImg *dst, int str[][2], int num_p){
    int     x, y, i;
    int     bufint;
    int     lx, ly;
    FHImg   imgtmp;
    uchar   *dp;
    int     ch = src->ch;

    FHImgInit(&imgtmp);
    if ( FHImgOpen(&imgtmp, src->h, src->w, src->ch) ){ return(-1); }

    for ( y=0 ; y<imgtmp.h ; y++ ){
        dp = FHImgPtr(&imgtmp, y  );
        for ( x=0 ; x<imgtmp.linesize ; x++ ){
            bufint = 0;
            for ( i=0 ; i<num_p ; i++ ){
                /* Local pixel */
                lx = ch*str[i][0] + x;
                ly = str[i][1] + y;
                if ( lx<0 || imgtmp.linesize-1<lx ){ continue; }
                if ( ly<0 || imgtmp.h-1<ly ){ continue; }
                bufint = FHMax(FHImgAt(src, ly, lx), bufint);
            }
            dp[x] = (uchar)bufint;
        }
    }

    FHImgMv(&imgtmp, dst);
    FHImgFinish(&imgtmp);
    return(0);
}


int FHMinkowskiPro3(FHImg *src, FHImg *dst, int kernel[][3]){
    int     x, y;
    int     bufint;
    FHImg   imgtmp;
    uchar   *dp, *p1, *p2, *p3;
    int     ch = src->ch;
    int     kernelcp[3][3];

    for ( y=0 ; y<3 ; y++ ){
        for ( x=0 ; x<3 ; x++ ){
            if ( kernel[y][x]!=0 && kernel[y][x]!=1 ){ return(-1); }
            if ( 0==kernel[y][x] ){ kernelcp[y][x] = 255; }
            else{ kernelcp[y][x] = 0; }
        }
    }

    FHImgInit(&imgtmp);
    if ( FHImgCp(src, &imgtmp) ){ return(-1); }

    for ( y=1 ; y<imgtmp.h-1 ; y++ ){
        dp = FHImgPtr(&imgtmp, y  );
        p1 = FHImgPtr(src, y-1);
        p2 = FHImgPtr(src, y  );
        p3 = FHImgPtr(src, y+1);
        for ( x=ch ; x<imgtmp.linesize-ch ; x++ ){
            bufint = 255;
            bufint = FHMin(p1[x-ch]+kernelcp[0][0], bufint);
            bufint = FHMin(p1[x   ]+kernelcp[0][1], bufint);
            bufint = FHMin(p1[x+ch]+kernelcp[0][2], bufint);
            bufint = FHMin(p2[x-ch]+kernelcp[1][0], bufint);
            bufint = FHMin(p2[x   ]+kernelcp[1][1], bufint);
            bufint = FHMin(p2[x+ch]+kernelcp[1][2], bufint);
            bufint = FHMin(p3[x-ch]+kernelcp[2][0], bufint);
            bufint = FHMin(p3[x   ]+kernelcp[2][1], bufint);
            bufint = FHMin(p3[x+ch]+kernelcp[2][2], bufint);
            dp[x] = (uchar)bufint;
        }
    }
    FHImgMv(&imgtmp, dst);
    FHImgClose(&imgtmp);
    return(0);
}


int FHMinkowskiPro5(FHImg *src, FHImg *dst, int kernel[][5]){
    int     x, y;
    FHImg   imgtmp;
    uchar   *dp;
    int     ch = src->ch;
    int     kernelcp[5][5];

    for ( y=0 ; y<5 ; y++ ){
        for ( x=0 ; x<5 ; x++ ){
            if ( kernel[y][x]!=0 && kernel[y][x]!=1 ){ return(-1); }
            if ( 0==kernel[y][x] ){ kernelcp[y][x] = 255; }
            else { kernelcp[y][x] = 0; }
        }
    }

    FHImgInit(&imgtmp);
    if ( FHImgCp(src, &imgtmp) ){ return(-1); }

    for ( y=2 ; y<imgtmp.h-2 ; y++ ){
        dp = FHImgPtr(&imgtmp, y  );
        for ( x=ch*2 ; x<imgtmp.linesize-ch*2 ; x++ ){
            int     i, j;
            int     bufint = 255;
            for ( i=0 ; i<4 ; i++ ){
                for ( j=0 ; j<4 ; j++ ){
                    bufint = FHMin(FHImgAt(src, \
                                y-(i-2), x-ch*(j-2))+kernelcp[i][j], bufint);
                }
            }
            dp[x] = (uchar)bufint;
        }
    }

    FHImgMv(&imgtmp, dst);
    FHImgClose(&imgtmp);
    return(0);
}


int FHMinkowskiPro(FHImg *src, FHImg *dst, int str[][2], int num_p){
    int     x, y, i;
    int     bufint;
    int     lx, ly;
    FHImg   imgtmp;
    uchar   *dp;
    int     ch = src->ch;

    FHImgInit(&imgtmp);
    if ( FHImgOpen(&imgtmp, src->h, src->w, src->ch) ){ return(-1); }

    for ( y=0 ; y<imgtmp.h ; y++ ){
        dp = FHImgPtr(&imgtmp, y  );
        for ( x=0 ; x<imgtmp.linesize ; x++ ){
            bufint = 0;
            for ( i=0 ; i<num_p ; i++ ){
                /* Local pixel */
                lx = ch*str[i][0] + x;
                ly = str[i][1] + y;
                if ( lx<0 || imgtmp.linesize-1<lx ){ continue; }
                if ( ly<0 || imgtmp.h-1<ly ){ continue; }
                bufint = FHMin(FHImgAt(src, ly, lx), bufint);
            }
            dp[x] = (uchar)bufint;
        }
    }

    FHImgMv(&imgtmp, dst);
    FHImgFinish(&imgtmp);
    return(0);
}


int FHErosion3(FHImg *src, FHImg *dst){
    int     kernel[3][3] = { {1, 1, 1}, {1, 1, 1}, {1, 1, 1} };
    return(FHMinkowskiPro3(src, dst, kernel));
}


int FHDilation3(FHImg *src, FHImg *dst){
    int     kernel[3][3] = { {1, 1, 1}, {1, 1, 1}, {1, 1, 1} };
    return(FHMinkowskiAdd3(src, dst, kernel));
    // int     str[9][2] = {
    //     {-1, -1},
    //     {-1,  0},
    //     {-1,  1}, 
    //     { 0, -1},
    //     { 0,  0},
    //     { 0,  1}, 
    //     { 1, -1},
    //     { 1,  0},
    //     { 1,  1}
    // };
    // return(FHMinkowskiAdd(src, dst, str, 9));
}


int FHErosion5(FHImg *src, FHImg *dst){
    int     kernel[5][5] = { 
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1}
    };
    return(FHMinkowskiPro5(src, dst, kernel));
}


int FHDilation5(FHImg *src, FHImg *dst){
    int     kernel[5][5] = { 
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1}
    };
    return(FHMinkowskiAdd5(src, dst, kernel));
}


int FHImgOpening3(FHImg *src, FHImg *dst){
    if ( FHErosion3(src, dst) ){ return(-1); }
    if ( FHDilation3(dst, dst) ){ return(-1); }
    return(0);
}


int FHImgClosing3(FHImg *src, FHImg *dst){
    if ( FHDilation3(src, dst) ){ return(-1); }
    if ( FHErosion3(dst, dst) ){ return(-1); }
    return(0);
}


int FHImgOpening5(FHImg *src, FHImg *dst){
    if ( FHErosion5(src, dst) ){ return(-1); }
    if ( FHDilation5(dst, dst) ){ return(-1); }
    return(0);
}


int FHImgClosing5(FHImg *src, FHImg *dst){
    if ( FHDilation5(src, dst) ){ return(-1); }
    if ( FHErosion5(dst, dst) ){ return(-1); }
    return(0);
}


int FHOCFilter3(FHImg *src, FHImg *dst){
    if ( FHImgOpening3(src, dst) ){ return(-1); }
    if ( FHImgClosing3(dst, dst) ){ return(-1); }
    return(0);
}


int FHCOFilter3(FHImg *src, FHImg *dst){
    if ( FHImgClosing3(src, dst) ){ return(-1); }
    if ( FHImgOpening3(dst, dst) ){ return(-1); }
    return(0);
}
