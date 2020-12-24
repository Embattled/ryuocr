/* --------------------------------------------------------
 * FHImgIO.cpp
 *      Written by F.Horie.
 * -------------------------------------------------------- */


#include    <stdio.h>
#include    <assert.h>
#include    "FHRGB.h"
#include    "FHStr.h"


#define     STB_IMAGE_WRITE_STATIC
#define     STB_IMAGE_STATIC
#define     STB_IMAGE_IMPLEMENTATION
#define     STB_IMAGE_WRITE_IMPLEMENTATION
#include    "stb_image.h"
#include    "stb_image_write.h"


/* --------------------------------------------------------
 * IO
 * -------------------------------------------------------- */

// int FHImgRead(FHImg *src, FILE *fp){
//     int     h0, w0, ch0;
//     if ( 1!=fread(&h0, sizeof(int), 1, fp) ){ return(-1); }
//     if ( 1!=fread(&w0, sizeof(int), 1, fp) ){ return(-1); }
//     if ( 1!=fread(&ch0, sizeof(int), 1, fp) ){ return(-1); }
//     FHImgClose(src);
//     if ( FHImgOpen(src, h0, w0, ch0) ){ return(-1); }
//     if ( (unsigned)src->size!=fread(src->d, sizeof(uchar), src->size, fp) ){ return(-1); }
//     return(0);
// }


int FHImgWrite(FHImg *src, FILE *fp){
    if ( 1!=fwrite(&src->h, sizeof(int), 1, fp) ){ return(-1); }
    if ( 1!=fwrite(&src->w, sizeof(int), 1, fp) ){ return(-1); }
    if ( 1!=fwrite(&src->ch, sizeof(int), 1, fp) ){ return(-1); }
    if ( (unsigned)src->size!=fwrite(src->d, sizeof(uchar), src->size, fp) ){ return(-1); }
    return(0);
}


void FHImgPrint(FHImg *src){
    int     x, y;

    printf("%dx%dx%d\n", src->w, src->h, src->ch);
    for ( y=0 ; y<src->h ; y++ ){
        uchar   *p = FHImgPtr(src, y);
        for ( x=0 ; x<src->linesize ; x++ ){ printf("%d ", p[x]); }
        putchar('\n');
    }
}


int FHImgReadPNMH(FILE *fp, int *width, int *height, int filetype){
    char    bufch[FH_BUF_LEN];
    char    magic[3];
    int     maxint;

    if ( 2!=sprintf(magic, "P%d", filetype) ){ return(-1); }
    if ( NULL==FHRead1Line(fp, bufch, FH_BUF_LEN) ){ return(-1); }
    if ( strcmp(magic, bufch) ){ return(-1); }
    do{
        if ( NULL==FHRead1Line(fp, bufch, FH_BUF_LEN) ){ return(-1); }
    }while ( bufch[0]=='#' );
    if ( 2!=sscanf(bufch, "%d %d", width, height) ){ return(-1); }
    if ( filetype!=4 ){
        if ( NULL==FHRead1Line(fp, bufch, FH_BUF_LEN) ){ return(-1); }
        if ( 1!=sscanf(bufch, "%d", &maxint) ){ return(-1); }
        if ( *width<1 || *height<1 || maxint!=255 ){ return(-1); }
    }
    else{
        if ( *width<1 || *height<1 ){ return(-1); }
    }

    return(0);
}


int FHImgReadPNMD(FILE *fp, int width, int height, uchar *data, int filetype){
    int     i, j, k;
    uchar   bufuc[width/8+1];
    const int     size = width * height * (int)((filetype - 4) * 3.0 / 2);

    if ( filetype==4 ){
        for ( i=0 ; i<height ; i++ ){
            if ( (uint)width/8 \
                    != fread(bufuc, sizeof(uchar), width/8, fp) ){ return(-1); }
            for ( j=0 ; j<width ; j+=8 ){
                for ( k=0 ; k<8 ; k++ ){
                    if ( j+k < width ){
                        data[i*width+j+k] =  (uchar)((~bufuc[j/8]>>(7-k)&0x01)*255);
                    }
                }
            }
        }
    }
    else if ( filetype==5 || filetype==6 ){
        if ( (uint)size != fread(data, sizeof(uchar), size, fp) ){ return(-1); }
    }
    else{ return(-1); }
    return(0);
}


int FHImgWritePNMH(FILE *fp, int width, int height, int filetype){
    if ( filetype==4 ){
        if ( 1 > fprintf(fp, "P%01d\n%04d %04d\n", filetype, width, height) ){ 
            return(-1);
        }
    }
    else{
        if ( 1 > fprintf(fp, "P%01d\n%04d %04d\n%04d\n", filetype, width, height, 255) ){
            return(-1);
        }
    }
    return(0);
}




int FHImgWritePNMD(FILE *fp, int width, int height, uchar *data, int filetype){
    int     i, j, k;
    int     bufw;
    uchar   bufuc[width/8+1];
    const int     size = width * height * (int)((filetype - 4) * 3.0 / 2);

    if ( filetype==4 ){
        for ( i=0 ; i<height ; i++ ){
            if ( width%8==0 ){ bufw = width; }
            else{ bufw = width + (8 - width % 8); }
            for ( j=0 ; j<bufw ; j+=8 ){
                bufuc[j/8] = 0x00;
                for ( k=0 ; k<8 ; k++ ){
                    if ( k+j < width ){ bufuc[j/8] |= (uchar)(~data[i*width+j+k] & 0x80 >> k); }
                }
            }
            if ( (uint)bufw/8 != fwrite(bufuc, sizeof(uchar), bufw/8, fp) ){ return(-1); }
        }
    }
    else if ( filetype==5 || filetype==6 ){
        if ( (uint)size!=fwrite(data, sizeof(uchar), size, fp) ){ return(-1); }
    }
    else{ return(-1); }
    return(0);
}


int FHImgLoad(FHImg *src, const char *fname){
    int     width, height, channels;
    uchar   *data;
    const char *ext = FHGetExt(fname);
    FILE    *fp;

    if ( 0==strcmp(ext, "PBM") || 0==strcmp(ext, "pbm") ){
        if ( NULL==(fp=fopen(fname, "rb")) ){ return(-1); }
        if ( FHImgReadPNMH(fp, &width, &height, 4) ){ fclose(fp); return(-1); }
        if ( FHImgOpen(src, height, width, 1) ){ fclose(fp); return(-1); }
        if ( FHImgReadPNMD(fp, src->w, src->h, src->d, 4) ){ 
            FHImgClose(src); fclose(fp); return(-1); 
        }
        if ( EOF==fclose(fp) ){ FHImgClose(src); return(-1); }
        return(0);
    }
    data = stbi_load(fname, &width, &height, &channels, 0);
    if ( NULL==data ){ return(-1); }
    if ( channels!=1 && channels!=3 ){ stbi_image_free(data); return(-1); }
    if ( FHImgOpen(src, height, width, channels) ){ stbi_image_free(data); return(-1); }
    if ( NULL==memcpy(src->d, data, src->size) ){ 
        FHImgFinish(src);
        stbi_image_free(data); return(-1); 
    }
    stbi_image_free(data);
    return(0);
}


int FHImgSave(FHImg *src, const char *fname){
    int     filetype = 0, n = 0;
    FILE    *fp;
    FHImg   imgtmp;
    const char  *ext = FHGetExt(fname);

    FHImgInit(&imgtmp);

    if ( 0==src->d ){ return(-1); }

    if ( 0==strcmp(ext, "pbm") || 0==strcmp(ext, "PBM") ){ filetype = 4; n = 1; }
    else if ( 0==strcmp(ext, "pgm") || 0==strcmp(ext, "PGM")  ){ filetype = 5; n = 1; }
    else if ( 0==strcmp(ext, "ppm") || 0==strcmp(ext, "PPM")  ){ filetype = 6; n = 3; }

    /* ---- PNM write ---- */

    if ( filetype!=0 ){
        if ( n==1 && src->ch==3 ){ 
            if ( FHToGray(src, &imgtmp) ){ FHImgFinish(&imgtmp); return(-1); }
        }
        else if ( n==3 && src->ch==1 ){ 
            if ( FHToRGB(src, &imgtmp) ){ FHImgFinish(&imgtmp); return(-1); }
        }
        else{ 
            if ( FHImgCp(src, &imgtmp) ){ FHImgFinish(&imgtmp); return(-1); }
        }

        if ( NULL==(fp=fopen(fname, "w")) ){ FHImgFinish(&imgtmp); return(-1); }

        if ( FHImgWritePNMH(fp, imgtmp.w, imgtmp.h, filetype) ){
            fclose(fp); FHImgFinish(&imgtmp); return(-1); }

        if ( FHImgWritePNMD(fp, imgtmp.w, imgtmp.h, imgtmp.d, filetype) ){
            fclose(fp); FHImgFinish(&imgtmp); return(-1); }

        if ( EOF==fclose(fp) ){ FHImgFinish(&imgtmp); return(-1); }
    }

    /* ---- BMP write ---- */

    else if ( 0==strcmp(ext, "bmp") || 0==strcmp(ext, "BMP") ){
        if ( 0==stbi_write_bmp(fname, src->w, src->h, src->ch, src->d) ){ 
            FHImgFinish(&imgtmp); return(-1); 
        }
    }

    /* ---- JPG write ---- */

    else if ( 0==strcmp(ext, "jpg") || 0==strcmp(ext, "JPG") ||
            0==strcmp(ext, "jpeg") || 0==strcmp(ext, "JPEG") ){
        if ( 0==stbi_write_jpg(fname, src->w, src->h, src->ch, src->d, 100) ){ 
            FHImgFinish(&imgtmp); return(-1); 
        }
    }

    /* ---- PNG write ---- */

    else if ( 0==strcmp(ext, "png") || 0==strcmp(ext, "PNG") ){
        if ( 0==stbi_write_png(fname, src->w, src->h, src->ch, src->d, src->linesize) ){
            FHImgFinish(&imgtmp); return(-1); 
        }
    }

    else{ FHImgFinish(&imgtmp); return(-1); }

    FHImgFinish(&imgtmp); 
    return(0);
}
