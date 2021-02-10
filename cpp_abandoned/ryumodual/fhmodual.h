#pragma once
#ifndef _FHMODUAL_H
#define _FHMODUAL_H

namespace ryu
{
    class fhocr
    {
    private:
        int K;
        int T;

        int use_nns;
        int use_svm;
        int use_mlp;

        int RIFeat[32];
        int SSCD[32];

        int feature; /* FHOCR_HOG, FHOCR_CSLBP, FHOCR_ROWFEAT or FHOCR_MRF */
        int voting;  /* FHOCR_HARD_VOTING, FHOCR_SOFT_VOTING or FHOCR_MAX_VOTING */
        int hidden;
        int max_epoch;
        char traindata_name[64];
        char traindata_path[128];
        int num_kernel;
        float (*kernel)[3][3];
        float (*mat)[2][2];
        float rs[1024][2];
        float *rn_alpha;
        float (*rfhog)[3][3];
        FHVDic *vdic;
        FHSVM *svm;
        FHMLP *mlp;
        FHSet class_table;
        float SVM_C;
        FHImg *mask;
        int *ans;       /* length = T */
        float *prob;    /* length = num_class */
        float max_prob; /* Max output probability of the Neural Network */
        char FHDIR[128];
        char CNN_FILE[128];
        
    public:
        fhocr();
        ~fhocr();
    };

} // namespace ryu

#endif