/* --------------------------------------------------------
 * FHOCR.h
 *      Written by F.Horie.
 * -------------------------------------------------------- */


#ifndef     FHOCR_H
#define     FHOCR_H


#include    <stdio.h>
#include    "FHStrBox.h"
#include    "FHIDic.h"
#include    "FHSet.h"
#include    "FHVDic.h"
#include    "FHSVM.h"
#include    "FHMLP.h"
#include    "FHTree.h"
#include    "FHTree.h"
#include    "FHImgFeat.h"
#include    "FHImgFeat.h"
//#include    "mojo.h"


namespace mojo{
    class network;
}


/* --------------------------------------------------------
 * FHOCR (Ensemble with SSCD and RI-Feature method)
 * -------------------------------------------------------- */

typedef struct FHOCR {
    int         K;
    int         T;
    int         use_nns;
    int         use_svm;
    int         use_mlp;
    int         RIFeat[32];
    int         SSCD[32];
    int         feature;    /* FHOCR_HOG, FHOCR_CSLBP, FHOCR_ROWFEAT or FHOCR_MRF */
    int         voting;     /* FHOCR_HARD_VOTING, FHOCR_SOFT_VOTING or FHOCR_MAX_VOTING */
    int         hidden;
    int         max_epoch;
    char        traindata_name[64];
    char        traindata_path[128];
    int         num_kernel;
    float       (*kernel)[3][3];
    float       (*mat)[2][2];
    float       rs[1024][2];
    float       *rn_alpha;
    float       (*rfhog)[3][3];
    FHVDic      *vdic;
    FHSVM       *svm;
    FHMLP       *mlp;
    FHSet       class_table;
    float       SVM_C;
    FHImg       *mask;
    int         *ans;       /* length = T */
    float       *prob;      /* length = num_class */
    float       max_prob;   /* Max output probability of the Neural Network */
    char        FHDIR[128];
    char        CNN_FILE[128];
    mojo::network   *cnn;
    FHOCR(void);
    virtual ~FHOCR(void);
} FHOCR;


enum FHOCRFeature {
    FHOCR_HOG,
    FHOCR_CSLBP,
    FHOCR_ROWFEAT,
    FHOCR_MRF,
    FHOCR_HOG16,
    FHOCR_RFHOG,
    FHOCR_CNNFEAT,
};


enum FHOCRVoting {
    FHOCR_HARD_VOTING,
    FHOCR_SOFT_VOTING,
    FHOCR_MAX_VOTING
};


extern char     FHOCRRIFeat[][16];

/** 
 * [Memo]
 * RIFeat[id]:
 *      id = 0:     MSR0 (16)
 *      id = 1:     MSR1 (32)
 *      id = 2:     MSR2 (64)
 *      id = 3:     MSR3 (16, 32)
 *      id = 4:     MSR4 (32, 64)
 *      id = 5:     MSR5 (16, 64)
 *      id = 6:     MSR6 (16, 32, 64)
 *      id = 7:     RA (Random Affine)
 *      id = 8:     RF (Random Filter)
 *      id = 9:     MF (Mean Filter)
 *      id = 10:    GCN (Gravitational Center Normalizing)
 *      id = 11:    NLN (NonLinear Normalizing)
 *      id = 12:    RGCN (Random Gravitational Center Normalization)
 *      id = 13:    RNLN (Random NonLinear Normalization)
 *      id = 14:    RN (Random Normalizing)
 *      id = 15:    LRN (Linear Resize Normalizing)
 *      id = 16:    INN (Inertia Normalizing)
 *      id = 17:    RM (Random Masking)
 *      id = 18:    RS (Random Scaling)
 *      id = 19:    AVP (Average Pooling)
 *      id = 20:    SF (Sobel Filter)
 *      id = -1:    Finish
 */


enum FHRIFeature {
    FHOCR_MSR_16,
    FHOCR_MSR_32,
    FHOCR_MSR_64,
    FHOCR_MSR_16_32,
    FHOCR_MSR_32_64,
    FHOCR_MSR_16_64,
    FHOCR_MSR_16_32_64,
    FHOCR_RA,
    FHOCR_RF,
    FHOCR_MF,
    FHOCR_GCN,
    FHOCR_NLN,
    FHOCR_RGCN,
    FHOCR_RNLN,
    FHOCR_RN,
    FHOCR_LRN,
    FHOCR_INN,
    FHOCR_RM,
    FHOCR_RS,
    FHOCR_AVP,
    FHOCR_SF,
    FHOCR_NONE
};


extern float    FHOCRVar;
extern float    FHOCREta;


/* ---- Functions ---- */

extern void     FHOCRInit(FHOCR *FHOCR);
extern int      FHOCROpen(FHOCR *FHOCR);
extern void     FHOCRClose(FHOCR *FHOCR);
extern void     FHOCRFinish(FHOCR *FHOCR);
extern int      FHOCRTrain(FHOCR *FHOCR);
extern void     FHOCRRecogOne(FHOCR *FHOCR, FHImg *query, int i, \
        int *nns_ans, int *svm_ans, int *mlp_ans);
extern int      FHOCRVoting(FHOCR *FHOCR, FHImg *query);
extern int      FHOCRRecog(FHOCR *FHOCR, FHImg *query);
extern int      FHOCRRecogThresh(FHOCR *FHOCR, FHImg *query, float thresh);
extern int      FHOCRMNRecog(FHOCR *FHOCR, FHImg *query, int num_norm);
extern int      FHOCRSNRecog(FHOCR *FHOCR, FHImg *query, float alpha);
extern int      FHOCRWrite(FHOCR *FHOCR, FILE *fp);
extern int      FHOCRSave(FHOCR *FHOCR, char *fname);
extern int      FHOCRRead(FHOCR *FHOCR, FILE *fp);
extern int      FHOCRLoad(FHOCR *FHOCR, char *fname);
extern int      FHOCRLoadParam(FHOCR *FHOCR, const char *fname);
extern void     FHOCRPrint(FHOCR *FHOCR, FILE *fp);

static inline int      FHOCRDim(FHOCR *FHOCR){
    int     dim;
    if ( FHOCR_HOG==FHOCR->feature ){ dim = FHHOGDim(); }
    else if ( FHOCR_CSLBP==FHOCR->feature ){ dim = FHCSLBPDim(); }
    else if ( FHOCR_ROWFEAT==FHOCR->feature ){ dim = 16*16; }
    else if ( FHOCR_HOG16==FHOCR->feature ){ dim = FHHOG16Dim(); }
    else if ( FHOCR_RFHOG==FHOCR->feature ){ dim = 320; }
    else if ( FHOCR_CNNFEAT==FHOCR->feature ){ dim = 512; }
    else{ return(-1); }
    return(dim);
}
static inline int     FHOCRNumClass(FHOCR *FHOCR){ return(FHOCR->class_table.num); }


inline FHOCR :: FHOCR(void){ FHOCRInit(this); }
inline FHOCR :: ~FHOCR(void){ FHOCRFinish(this); }


#endif      /* FHOCR_H */
