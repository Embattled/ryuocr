/* --------------------------------------------------------
 * FHResize.h
 *      Written by F.Horie.
 * -------------------------------------------------------- */


#ifndef     FHRESIZE_H
#define     FHRESIZE_H


/* --------------------------------------------------------
 * Resizing (FHResize.cpp)
 * -------------------------------------------------------- */

extern int      FHResize(FHImg *src, FHImg *dst, float xs, float ys);
extern int      FHResizeSquare(FHImg *src, FHImg *dst, int sz);
extern int      FHResizeInside(FHImg *src, FHImg *dst, int sz);


#endif      /* FHRESIZE_H */
