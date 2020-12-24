/* --------------------------------------------------------
 * FHRecTest.h
 *      Written by F.Horie.
 * -------------------------------------------------------- */


#ifndef     FHRECTEST_H
#define     FHRECTEST_H


#ifdef  __cplusplus
extern "C" {
#endif


extern int      FHRecTestDebug;
extern int      FHRecTestCorrect;


extern void     FHRecTestStart(void);
extern void     FHRecTestEnd(int ans, int label);
extern void     FHRecTestReset(void);
extern double   FHRecTestAccuracy(void);
extern double   FHRecTestTime(void);


#ifdef  __cplusplus
}
#endif


#endif      /* FHRECTEST_H */
