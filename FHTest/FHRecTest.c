/* --------------------------------------------------------
 * FHRecTest.c
 *      Written by F.Horie.
 * -------------------------------------------------------- */


#include    <stdio.h>
#include    "FHRecTest.h"
#include    "FHTime.h"


#ifdef  __cplusplus
extern "C" {
#endif


static double   sumtime = 0;
static int      correct = 0;
static int      no = 0;
int     FHRecTestDebug = 1;
int     FHRecTestCorrect = 0;


void FHRecTestStart(){ FHTimeStart(); }
void FHRecTestEnd(int ans, int label){
    sumtime += FHTimeEnd();
    no++;
    if ( ans==label ){ FHRecTestCorrect = 1; correct++; }
    else{ FHRecTestCorrect = 0; }

    if ( FHRecTestDebug ){
        puts("------------------------------------");
        printf("No     : %d\n", no);
        printf("Answer : %d Correct : %d\n", ans, label);
        if ( ans==label ){
            printf("True\n");
        }
        else{
            printf("False\n");
        }
        printf("RR     : %4.2f %%\n", FHRecTestAccuracy());
        printf("AT     : %4.2f msec\n", FHRecTestTime());
    }
    return;
}


void FHRecTestReset(void){
    sumtime = 0;
    correct = 0;
    no = 0;
}


double FHRecTestAccuracy(void){ return(100*(double)correct/no); }
double FHRecTestTime(void){ return(1000*sumtime/no); }


#ifdef  __cplusplus
}
#endif
