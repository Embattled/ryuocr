/* --------------------------------------------------------
 * FHOCR.cpp
 *      Written by F.Horie.
 * -------------------------------------------------------- */


#include    <math.h>
#include    <stdlib.h>
#include    <unistd.h>
#include    <immintrin.h>
#include    "FHImgIO.h"
#include    "FHRGB.h"
#include    "FHOCR.h"
#include    "FHStr.h"
#include    "FHKNN.h"
#include    "FHRnd.h"
#include    "FHDef.h"
#include    "FHMath.h"
#include    "FHSSCD.h"
#include    "FHImgBlend.h"
#include    "FHImgFeat.h"
#include    "FHResize.h"
#include    "FHAffine.h"
#include    "FHSort.h"
#include    "FHFilter.h"
#include    "FHDeform.h"
#include    "FHMorphology.h"
#include    "FHNormChar.h"
#include    "FHVoting.h"
#include    "FHOCRData.h"
#include    "mojo.h"


#define     SHOW_IMAGE(x)   do{ \
    FHImgSave(x, "/home/fuma/Pictures/tmp.bmp"); \
    getchar(); \
}while(0)


static int img_size[3] = {16, 32, 64};


static char FeatureStr[][8] = {
    "HOG",
    "CSLBP",
    "ROWFEAT",
    "MRF",
    "HOG16",
    "RFHOG",
    "CNN",
};


float   FHOCRVar = 0;
float   FHOCREta = 0;

char    FHOCRRIFeat[][16] = {
    "MSR(16)",
    "MSR(32)",
    "MSR(64)",
    "MSR(16, 32)",
    "MSR(32, 64)",
    "MSR(16, 64)",
    "MSR(16, 32, 64)",
    "RA",
    "RF",
    "MF",
    "GCN",
    "NLN",
    "RGCN",
    "RNLN",
    "RN",
    "LRN",
    "INN",
    "RM",
    "RS",
    "AVP",
    "SF",
    "NONE"
};


/* --------------------------------------------------------
 * Initialization / Finalization
 * -------------------------------------------------------- */

void FHOCRInit(FHOCR *FHOCR){
    const char *homepass = getenv("HOME");

    FHOCR->vdic = 0; 
    FHOCR->svm = 0; 
    FHOCR->mlp = 0;
    FHOCR->K = 0;
    FHOCR->T = 0;
    FHOCR->use_nns = 0;
    FHOCR->use_svm = 0;
    FHOCR->use_mlp = 0;
    FHOCR->kernel = 0;
    FHOCR->mat = 0;
    FHOCR->rn_alpha = 0;
    FHOCR->num_kernel = 0;
    FHOCR->SVM_C = 1.0f;
    FHOCR->feature = FHOCR_HOG;
    FHOCR->voting = FHOCR_HARD_VOTING;
    FHOCR->RIFeat[0] = FHOCR_MSR_16_32_64;
    FHOCR->RIFeat[1] = FHOCR_RF;
    FHOCR->RIFeat[2] = FHOCR_MF;
    FHOCR->RIFeat[3] = FHOCR_RF;
    FHOCR->RIFeat[4] = FHOCR_MF;
    FHOCR->RIFeat[5] = FHOCR_RF;
    FHOCR->RIFeat[6] = -1;
    FHOCR->SSCD[0] = FHSSCD_RA;
    FHOCR->SSCD[1] = FHSSCD_DE;
    FHOCR->SSCD[2] = FHSSCD_AC;
    FHOCR->SSCD[3] = FHSSCD_GF;
    FHOCR->SSCD[4] = FHSSCD_RF;
    FHOCR->SSCD[5]= -1;
    FHOCR->mask = 0;
    FHOCR->hidden = 320;
    FHOCR->max_epoch = 10;
    FHOCR->ans = 0;
    FHOCR->prob = 0;
    FHOCR->rfhog = 0;
    FHOCR->cnn = 0;
    sprintf(FHOCR->FHDIR, "%s/.FH", homepass);
    sprintf(FHOCR->CNN_FILE, "%s/prog/cnn_feature.mojo", FHOCR->FHDIR);
    // sprintf(FHOCR->CNN_FILE, "../mojo-cnn/models/mnist_deepcnet.mojo");
    sprintf(FHOCR->traindata_name, "%s", "7FontJP");
    strcpy(FHOCR->traindata_path, "\0");
    FHSetInit(&FHOCR->class_table);
    return;
}


void FHOCRFinish(FHOCR *fhocr){
    FHOCRClose(fhocr);
    FHOCRInit(fhocr);
    return;
}


/* --------------------------------------------------------
 * Open / Close
 * -------------------------------------------------------- */

int FHOCROpen(FHOCR *fhocr, int K, int T, int use_nns, int use_svm, int use_mlp){
    int     i;
    FHOCRFinish(fhocr);
    fhocr->K = K;
    fhocr->T = T;
    fhocr->use_nns = use_nns;
    fhocr->use_svm = use_svm;
    fhocr->use_mlp = use_mlp;

    if ( use_nns ){
        fhocr->vdic = FHMalloc(FHVDic, T);
        for ( i=0 ; i<T ; i++ ){
            FHVDicInit(&fhocr->vdic[i]);
        }
    }
    if ( use_svm ){
        fhocr->svm = FHMalloc(FHSVM, T);
        for ( i=0 ; i<T ; i++ ){
            FHSVMInit(&fhocr->svm[i]);
        }
    }
    if ( use_mlp ){
        fhocr->mlp = FHMalloc(FHMLP, T);
        for ( i=0 ; i<T ; i++ ){
            FHMLPInit(&fhocr->mlp[i]);
        }
    }
    return(0);
}


void FHOCRClose(FHOCR *FHOCR){
    int     i;

    /* ---- Classifier ---- */

    if ( FHOCR->vdic ){ 
        for ( i=0 ; i<FHOCR->T ; i++ ){
            FHVDicFinish(&FHOCR->vdic[i]);
        }
        delete []FHOCR->vdic; FHOCR->vdic = 0; 
    }
    if ( FHOCR->svm ){ 
        for ( i=0 ; i<FHOCR->T ; i++ ){
            FHSVMFinish(&FHOCR->svm[i]);
        }
        delete []FHOCR->svm; FHOCR->svm = 0; 
    }
    if ( FHOCR->mlp ){ 
        for ( i=0 ; i<FHOCR->T ; i++ ){
            FHMLPFinish(&FHOCR->mlp[i]);
        }
        delete []FHOCR->mlp; FHOCR->mlp = 0; 
    }
    
    /* ---- Table ---- */

    FHSetClose(&FHOCR->class_table);

    /* ---- RI-Feature ---- */

    if ( FHOCR->kernel ){ delete []FHOCR->kernel; FHOCR->kernel = 0; }
    if ( FHOCR->mat ){ delete []FHOCR->mat; FHOCR->mat = 0; }
    if ( FHOCR->rn_alpha ){ delete []FHOCR->rn_alpha; FHOCR->rn_alpha = 0; }
    if ( FHOCR->mask ){ 
        for ( i=0 ; i<FHOCR->T ; i++ ){
            FHImgFinish(&FHOCR->mask[i]);
        }
        delete []FHOCR->mask; 
        FHOCR->mask = 0; 
    }

    if ( FHOCR->ans ){ FHFree(FHOCR->ans); FHOCR->ans = 0; }
    if ( FHOCR->prob ){ FHFree(FHOCR->prob); FHOCR->prob = 0; }

    /* ---- RF HOG feature ---- */

    if ( FHOCR->rfhog ){ delete []FHOCR->rfhog; FHOCR->rfhog = 0; }

    /* ---- CNN ---- */

    if ( FHOCR->cnn ){ delete []FHOCR->cnn; FHOCR->cnn = 0; }

    return;
}


int FHRFHOG(FHOCR *FHOCR, FHImg *src, float *feat, int no){
    int     i;
    FHImg   imgtmp;

    FHImgInit(&imgtmp);
    for ( i=0 ; i<16 ; i++ ){
        FHImgCp(src, &imgtmp);
        FHFilter3(&imgtmp, &imgtmp, FHOCR->rfhog[no*48+i*3+0]);
        FHMeanFilter3(&imgtmp, &imgtmp);
        // FHFilter3(&imgtmp, &imgtmp, FHOCR->rfhog[no*48+i*3+1]);
        if ( imgtmp.w==64 || imgtmp.w==32 ){
            FHMeanPooling(&imgtmp, &imgtmp);
        }
        FHFilter3(&imgtmp, &imgtmp, FHOCR->rfhog[no*48+i*3+1]);
        FHMeanFilter3(&imgtmp, &imgtmp);
        // FHFilter3(&imgtmp, &imgtmp, FHOCR->rfhog[no*48+i*3+2]);
        if ( imgtmp.w==32 ){
            FHMeanPooling(&imgtmp, &imgtmp);
        }
        // FHFilter3(&imgtmp, &imgtmp, FHOCR->rfhog[no*48+i*3+2]);
        FHHOG16(&imgtmp, feat+i*20);
    }
    FHImgFinish(&imgtmp);
    return(0);
}



/* --------------------------------------------------------
 * Extract CNN Feature
 * -------------------------------------------------------- */

int FHCNNFeat(FHOCR *FHOCR, FHImg *src, float *feat){
    int     i;
    float   *in = new float[src->size];
    for ( i=0 ; i<src->size ; i++ ){ in[i] = (float)src->d[i]/255; }
    // memcpy(FHOCR->cnn->forward(in), feat, sizeof(float)*FHOCR->cnn->out_size());
    feat = FHOCR->cnn->forward(in);
    delete []in;
    return(0);
}


/* --------------------------------------------------------
 * Multi-Random Fitler Feature
 * -------------------------------------------------------- */

int FHMRFFeat(FHOCR *FHOCR, FHImg *src, float *feat){
    return(0);
}

/* --------------------------------------------------------
 * Random Image Feature (RI-Feature)
 * -------------------------------------------------------- */

int FHRIFeat(FHOCR *FHOCR, FHImg *src, float *feat, int no){
    int     i;
    int     no_kernel;
    int     ret = 0;
    int     cnt = 0;
    FHImg   imgtmp, masktmp;

    FHImgInit(&imgtmp);

    ret = FHImgCp(src, &imgtmp);
    if ( ret ){ FHImgFinish(&imgtmp); return(-1); }

    no_kernel = no * FHOCR->num_kernel;
    cnt = 0;
    for ( i=0 ; ; i++ ){
        if ( -1==FHOCR->RIFeat[i] ){ break; }
        switch ( FHOCR->RIFeat[i] ){
            case    FHOCR_MSR_16:
                ret = FHResizeSquare(&imgtmp, &imgtmp, 16);
                break;
            case    FHOCR_MSR_32:
                ret = FHResizeSquare(&imgtmp, &imgtmp, 32);
                break;
            case    FHOCR_MSR_64:
                ret = FHResizeSquare(&imgtmp, &imgtmp, 64);
                break;
            case    FHOCR_MSR_16_32:
                ret = FHResizeSquare(&imgtmp, &imgtmp, 16+(no%2)*16);
                break;
            case    FHOCR_MSR_32_64:
                ret = FHResizeSquare(&imgtmp, &imgtmp, 32+(no%2)*32);
                break;
            case    FHOCR_MSR_16_64:
                ret = FHResizeSquare(&imgtmp, &imgtmp, 16+(no%2)*48);
                break;
            case    FHOCR_MSR_16_32_64:
                ret = FHResizeSquare(&imgtmp, &imgtmp, img_size[no%3]);
                break;
            case    FHOCR_RA:
                ret = FHCenterAffine(&imgtmp, &imgtmp, FHOCR->mat[no]);
                break;
            case    FHOCR_RF:
                ret = FHFilter3(&imgtmp, &imgtmp, FHOCR->kernel[no_kernel+cnt]);
                cnt++;
                break;
            case    FHOCR_MF:
                ret = FHMeanFilter3(&imgtmp, &imgtmp);
                break;
            case    FHOCR_GCN:
                ret = FHGCNorm(&imgtmp, &imgtmp, 1, 1.0f);
                break;
            case    FHOCR_NLN:
                ret = FHNLNorm(&imgtmp, &imgtmp, 1, 0.5f);
                break;
            case    FHOCR_INN:
                ret = FHInertiaNorm(&imgtmp, &imgtmp, 1);
                break;
            case    FHOCR_LRN:
                ret = FHLRNorm(&imgtmp, &imgtmp, 1);
                break;
            case    FHOCR_RGCN:
                if ( no%6>=3 ){
                    ret = FHGCNorm(&imgtmp, &imgtmp, 1, 1.0f);
                }
                break;
            case    FHOCR_RNLN:
                if ( no%6>=3 ){
                    ret = FHNLNorm(&imgtmp, &imgtmp, 1, 1.0f);
                }
                break;
            case    FHOCR_RN:
                ret = FHGCNorm(&imgtmp, &imgtmp, 1, FHOCR->rn_alpha[no*2+0]);
                ret += FHNLNorm(&imgtmp, &imgtmp, 1, FHOCR->rn_alpha[no*2+1]);
                break;
            case    FHOCR_RM:
                FHImgInit(&masktmp);
                ret = FHResizeSquare(&FHOCR->mask[no], &masktmp, imgtmp.h);
                ret += FHImgBlend(&imgtmp, &masktmp, &imgtmp, 0.8);
                FHImgFinish(&masktmp);
                break;
            case    FHOCR_RS:
                ret = FHAffineResize(&imgtmp, &imgtmp, \
                        FHOCR->rs[no][0], FHOCR->rs[no][1]);
                break;
            case    FHOCR_NONE:
                ret = 0;
                break;
            case    FHOCR_AVP:
                ret = FHMeanPooling(&imgtmp, &imgtmp);
                break;
            case    FHOCR_SF:
                ret = FHSobel(&imgtmp, &imgtmp);
                ret += FHToGray(&imgtmp, &imgtmp);
                break;
            default:
                ret = -1;
                break;
        }
        if ( ret ){ FHImgFinish(&imgtmp); return(-1); }
    }

    if ( FHOCR_HOG==FHOCR->feature ){
        ret = FHHOG(&imgtmp, feat);
    }
    else if ( FHOCR_CSLBP==FHOCR->feature ){
        ret = FHCSLBP(&imgtmp, feat);
    }
    else if ( FHOCR_ROWFEAT==FHOCR->feature ){
        ret = FHImgRowFeat(&imgtmp, feat);
    }
    else if ( FHOCR_MRF==FHOCR->feature ){
        ret = FHMRFFeat(FHOCR, &imgtmp, feat);    
    }
    else if ( FHOCR_HOG16==FHOCR->feature ){
        ret = FHHOG16(&imgtmp, feat);    
    }
    else if ( FHOCR_RFHOG==FHOCR->feature ){
        ret = FHRFHOG(FHOCR, &imgtmp, feat, no);
    }
    else if ( FHOCR_CNNFEAT==FHOCR->feature ){
        FHResizeSquare(&imgtmp, &imgtmp, 32);
        ret = FHCNNFeat(FHOCR, &imgtmp, feat);
    }
    if ( ret ){ FHImgFinish(&imgtmp); return(-1); }
    FHImgFinish(&imgtmp);
    return(0);
}


/* --------------------------------------------------------
 * Print Parameter
 * -------------------------------------------------------- */

void FHOCRPrint(FHOCR *FHOCR, FILE *fp){
    int     i;
    int     dim;

    fprintf(fp, "Training data: %s\n", FHOCR->traindata_name);
    fprintf(fp, "K=%d\n", FHOCR->K);
    fprintf(fp, "T=%d\n", FHOCR->T);
    fprintf(fp, "NNS: %d\n", FHOCR->use_nns);
    fprintf(fp, "SVM: %d\n", FHOCR->use_svm);
    fprintf(fp, "MLP: %d\n", FHOCR->use_mlp);
    fprintf(fp, "Feature: %s\n", FeatureStr[FHOCR->feature]);
    dim = FHOCRDim(FHOCR);
    fprintf(fp, "dim = %d\n", dim);
    fprintf(fp, "RI-Feature: ");
    for ( i=0 ; ; i++ ){
        if ( -1==FHOCR->RIFeat[i] ){ break; }
        fprintf(fp, "%s", FHOCRRIFeat[FHOCR->RIFeat[i]]);
        if ( -1!=FHOCR->RIFeat[i+1] ){ fprintf(fp, "-"); }
    }
    fprintf(fp, "\n");
    fprintf(fp, "SSCD: ");
    for ( i=0 ; ; i++ ){
        if ( -1==FHOCR->SSCD[i] ){ break; }
        fprintf(fp, "%s", FHSSCDStr[FHOCR->SSCD[i]]);
        if ( -1!=FHOCR->SSCD[i+1] ){ fprintf(fp, "-"); }
    }
    fprintf(fp, "\n");
    if ( FHOCR->use_svm ){ fprintf(fp, "SVM_C = %f\n", FHOCR->SVM_C); }
    if ( FHOCR->use_mlp ){ fprintf(fp, "Hidden: %d\n", FHOCR->hidden); }
    return;
}


/* --------------------------------------------------------
 * Training
 * -------------------------------------------------------- */

int FHOCRTrain(FHOCR *FHOCR){
    int     i, j;
    int     T = FHOCR->T, K = FHOCR->K;
    int     dim = FHOCRDim(FHOCR);
    int     id;
    int     judge = 0;
    int     selected_no;
    FHIDic  idic;
    FHVDic  vdictmp;
    FHImg   imgtmp;

    if ( strcmp(FHOCR->traindata_path, "")==0 ){ 
        fprintf(stderr, "[FHOCR] error: Training data path is not set.\n");
        return(-1); 
    }
    if ( FHOCR->vdic!=0 || FHOCR->svm!=0 || FHOCR->mlp!=0 ){ return(-1); }
    if ( FHOCR->kernel!=0 || FHOCR->mat!=0 || FHOCR->rn_alpha!=0 || FHOCR->mask!=0 ){ return(-1); }

    if ( FHOCR->use_nns ){ 
        FHOCR->vdic = new FHVDic[T]; 
        for ( i=0 ; i<T ; i++ ){ FHVDicInit(&FHOCR->vdic[i]); }
    }
    if ( FHOCR->use_svm ){ 
        FHOCR->svm = new FHSVM[T]; 
        for ( i=0 ; i<T ; i++ ){ FHSVMInit(&FHOCR->svm[i]); }
    }
    if ( FHOCR->use_mlp ){ 
        FHOCR->mlp = new FHMLP[T]; 
        for ( i=0 ; i<T ; i++ ){ FHMLPInit(&FHOCR->mlp[i]); }
    }

    FHIDicInit(&idic);
    if ( FHIDicLoad(&idic, FHOCR->traindata_path) ){ FHIDicFinish(&idic); return(-1); }

    FHSegFile(FHOCR->traindata_path, FHOCR->traindata_name, (int)strlen(FHOCR->traindata_path));

    FHSetArrImport(&FHOCR->class_table, idic.label, idic.num);

    FHOCR->num_kernel = 0;
    for ( i=0 ; ; i++ ){
        if ( -1==FHOCR->RIFeat[i] ){ break; }
        if ( FHOCR_RF==FHOCR->RIFeat[i] ){ FHOCR->num_kernel++; }
    }

    FHOCR->kernel = new float[FHOCR->T*FHOCR->num_kernel][3][3];
    FHOCR->mat = new float[FHOCR->T][2][2];
    FHOCR->rn_alpha = new float[FHOCR->T*2];
    FHOCR->mask = new FHImg[FHOCR->T];

    FHOCRPrint(FHOCR, stdout);

    /* RF HOG feature */
    if ( FHOCR->feature==FHOCR_RFHOG ){
        FHOCR->rfhog = new float[FHOCR->T*48][3][3];
        for ( i=0 ; i<T*48 ; i++ ){
            FHMakeKernel3(FHOCR->rfhog[i]);
        }
    }

    /* CNN feature */

    if ( FHOCR->feature==FHOCR_CNNFEAT ){
        printf("Loading CNN model ... \n");
        FHOCR->cnn = new mojo::network;
        FHOCR->cnn->read(FHOCR->CNN_FILE);
        printf("Loading CNN model ... done\n");
    }

    puts("Training start");

    for ( i=0 ; i<T ; i++ ){
        printf("training no. %d/%d\n", i+1, T);
        FHImgInit(&FHOCR->mask[i]);
        FHImgOpen(&FHOCR->mask[i], 64, 64, 1);
        FHVDicInit(&vdictmp);

        for ( j=0 ; j<FHOCR->num_kernel ; j++ ){
            id = i*FHOCR->num_kernel+j;
            FHMakeKernel3(FHOCR->kernel[id]);
        }

        FHMakeMat2(FHOCR->mat[i], 0.2f);
        FHOCR->rn_alpha[i*2+0] = FHRndF(0.0, 1.0);
        FHOCR->rn_alpha[i*2+1] = FHRndF(0.5, 1.0);

        FHOCR->rn_alpha[i*2+0] = 1.0;
        // FHOCR->rn_alpha[i*2+1] = 0.5;

        FHMakeGoghMask(&FHOCR->mask[i]);
        FHToRGB(&FHOCR->mask[i], &FHOCR->mask[i]);

        FHOCR->rs[i][0] = FHRndF(1.0, 1.1);
        FHOCR->rs[i][1] = FHRndF(1.0, 1.1);

        if ( FHVDicOpen(&vdictmp, K, dim) ){ FHIDicFinish(&idic); return(-1); }

        judge = 0;

#ifdef  _OPENMP
#pragma omp parallel for private(selected_no, imgtmp, judge)
#endif  /* _OPENMP */
        for ( j=0 ; j<K ; j++ ){
            // printf("%d/%d\n", j+1, K);
            judge = 1;
            while ( judge ){
                judge = 0;
                FHImgInit(&imgtmp);
                selected_no = FHRndI(0, idic.num-1);
                FHImgCp(&idic.img[selected_no], &imgtmp);
                FHGenSSCD(&imgtmp, &imgtmp, FHOCR->SSCD);

                // char    filetmp[256];
                // sprintf(filetmp, "/home/fuma/Pictures/SSCDCH/%d.bmp", j);
                // FHImgSave(&imgtmp, filetmp);
                // getchar();

                judge = FHRIFeat(FHOCR, &imgtmp, vdictmp.d[j], i);
                vdictmp.label[j] = \
                    FHSetSearch(&FHOCR->class_table, idic.label[selected_no]);
                FHImgFinish(&imgtmp);
            }
        }

        if ( judge ){
            i--;
            FHVDicFinish(&vdictmp);
            continue;
        }
        else{
            if ( FHOCR->use_svm ){
                FHOCR->svm[i].debug = 0;
                FHSVMSetC(&FHOCR->svm[i], FHOCR->SVM_C);
                FHSVMTrain(&FHOCR->svm[i], &vdictmp);
            }
            if ( FHOCR->use_mlp ){
                FHMLPTrain(&FHOCR->mlp[i], &vdictmp, FHOCR->hidden, FHOCR->max_epoch);
            }
            if ( FHOCR->use_nns ){ 
                FHVDicCp(&vdictmp, &FHOCR->vdic[i]); 
            }

            FHVDicFinish(&vdictmp);
        }
    }

    puts("training done.");

    FHIDicFinish(&idic);
    return(0);
}


/* --------------------------------------------------------
 * Recognition
 * -------------------------------------------------------- */

void FHOCRRecogOne(FHOCR *FHOCR, FHImg *query, int no, \
        int *nns_ans, int *svm_ans, int *mlp_ans){
    int     dim = FHOCRDim(FHOCR);
    float   *feat = FHMalloc(float, dim);
    float   probtmp;

    FHRIFeat(FHOCR, query, feat, no);

    *nns_ans = *svm_ans = *mlp_ans = -1;
    if ( FHOCR->use_nns ){ *nns_ans = FHKNN(&FHOCR->vdic[no], feat, 1); }
    if ( FHOCR->use_svm ){ *svm_ans = FHSVMPredict(&FHOCR->svm[no], feat); }
    if ( FHOCR->use_mlp ){ 
        *mlp_ans = FHMLPPredictProb(&FHOCR->mlp[no], feat, &probtmp); 
        FHOCR->max_prob += probtmp;
    }

    FHFree(feat);
    return;
}


int FHOCRHardVoting(FHOCR *FHOCR, FHImg *query){
    int     i;
    int     flag = 0;
    int     *ansarr = FHMalloc(int, FHOCR->T*3);
    int     num_class = FHOCRNumClass(FHOCR);
    int     rifeat_org[32];
    int     kind_size = 1;
    FHImg   img[3];

    FHImgInit(&img[0]);

    FHImgCp(query, &img[0]);

    for ( i=0 ; ; i++ ){
        rifeat_org[i] = FHOCR->RIFeat[i];
        if ( FHOCR->RIFeat[i]==-1 ){ break; }
    }

    /* ---- Adopting fast algorithm for the recognition stage ---- */

    while ( 1 ){

        switch ( FHOCR->RIFeat[0] ){
            case    FHOCR_MSR_16:
                kind_size = 1;
                FHResizeSquare(query, &img[0], img_size[0]);
                break;
            case    FHOCR_MSR_32:
                kind_size = 1;
                FHResizeSquare(query, &img[0], img_size[1]);
                break;
            case    FHOCR_MSR_64:
                kind_size = 1;
                FHResizeSquare(query, &img[0], img_size[2]);
                break;
            case    FHOCR_MSR_16_32:
                kind_size = 2;
                FHResizeSquare(query, &img[0], img_size[0]);
                FHResizeSquare(query, &img[1], img_size[1]);
                break;
            case    FHOCR_MSR_16_64:
                kind_size = 2;
                FHResizeSquare(query, &img[0], img_size[0]);
                FHResizeSquare(query, &img[1], img_size[2]);
                break;
            case    FHOCR_MSR_32_64:
                kind_size = 2;
                FHResizeSquare(query, &img[0], img_size[1]);
                FHResizeSquare(query, &img[1], img_size[2]);
                break;
            case    FHOCR_MSR_16_32_64:
                kind_size = 3;
                FHResizeSquare(query, &img[0], img_size[0]);
                FHResizeSquare(query, &img[1], img_size[1]);
                FHResizeSquare(query, &img[2], img_size[2]);
                break;
            case    FHOCR_GCN:
                for ( i=0 ; i<kind_size ; i++ ){
                    FHGCNorm(&img[i], &img[i], 1, 1.0);
                }
                break;
            case    FHOCR_NLN:
                for ( i=0 ; i<kind_size ; i++ ){
                    FHNLNorm(&img[i], &img[i], 1, 0.5);
                }
                break;
            default:
                flag = 1;
                break;
        }
        if ( flag ){ break; }
        for ( i=1 ; ; i++ ){
            FHOCR->RIFeat[i-1] = FHOCR->RIFeat[i];
            if ( FHOCR->RIFeat[i-1]==-1 ){ break; }
        }
    }

#ifdef  _OPENMP
#pragma omp parallel for
#endif
    for ( i=0 ; i<FHOCR->T ; i++ ){
        FHOCRRecogOne(FHOCR, &img[i%kind_size], i, \
                &ansarr[i*3], &ansarr[i*3+1], &ansarr[i*3+2]);
    }

    for ( i=0 ; ; i++ ){
        FHOCR->RIFeat[i] = rifeat_org[i];
        if ( FHOCR->RIFeat[i]==-1 ){ break; }
    }

    FHHardVoting(ansarr, FHOCR->prob, FHOCR->T*3, num_class);
    for ( i=0 ; i<num_class ; i++ ){ 
        FHOCR->prob[i] /= (FHOCR->T*(FHOCR->use_nns+FHOCR->use_svm+FHOCR->use_mlp)); 
    }

    FHImgFinish(&img[0]);
    FHImgFinish(&img[1]);
    FHImgFinish(&img[2]);
    FHFree(ansarr);
    return(0);
}


int FHOCRSoftVoting(FHOCR *FHOCR, FHImg *query){
    int         i;
    int         dim = FHOCRDim(FHOCR);
    int         num_class = FHOCRNumClass(FHOCR);
    float       *feat, *hidden, **prob;
  
    if ( FHOCR->use_mlp!=1 ){ return(-1); }
    prob = new float*[FHOCR->T];

#ifdef  _OPENMP
#pragma omp parallel for private(hidden, feat)
#endif
    for ( i=0 ; i<FHOCR->T ; i++ ){
        prob[i] = new float[num_class];
        feat = new float[dim];
        hidden = new float[FHOCR->mlp[i].hidden_dim];
        for ( int j=0 ; j<FHOCR->mlp[i].hidden_dim ; j++ ){
            // printf("%f ", hidden[j]);
        }
        // printf("\n");
        FHRIFeat(FHOCR, query, feat, i);
        FHMLPForward(&FHOCR->mlp[i], feat, hidden, prob[i]);
        FHMLPSoftmax(prob[i], num_class);
        delete []feat;
        delete []hidden;
    }

    FHSoftVoting(prob, FHOCR->prob, FHOCR->T, num_class);

    for ( i=0 ; i<num_class ; i++ ){
        if ( FHOCR->prob[i]!=0 ){
            // printf("%e ", FHOCR->prob[i]);
        }
    }
    // printf("\n");

    for ( i=0 ; i<num_class ; i++ ){ FHOCR->prob[i] /= FHOCR->T; }
    for ( i=0 ; i<FHOCR->T ; i++ ){ delete []prob[i]; }
    delete []prob;
    return(0);
}


int FHOCRMaxVoting(FHOCR *FHOCR, FHImg *query){
    int         i;
    int         dim = FHOCRDim(FHOCR);
    int         num_class = FHOCRNumClass(FHOCR);
    float       *feat, *hidden, **prob;
  
    if ( FHOCR->use_mlp!=1 ){ return(-1); }
    prob = new float*[FHOCR->T];

#ifdef  _OPENMP
#pragma omp parallel for private(hidden, feat)
#endif
    for ( i=0 ; i<FHOCR->T ; i++ ){
        prob[i] = new float[num_class];
        feat = new float[dim];
        hidden = new float[FHOCR->mlp[i].hidden_dim];
        FHRIFeat(FHOCR, query, feat, i);
        FHMLPForward(&FHOCR->mlp[i], feat, hidden, prob[i]);
        FHMLPSoftmax(prob[i], num_class);
        delete []feat;
        delete []hidden;
    }

    FHMaxVoting(prob, FHOCR->prob, FHOCR->T, num_class);
    for ( i=0 ; i<FHOCR->T ; i++ ){ delete []prob[i]; }
    delete []prob;
    return(0);
}


int FHOCRVoting(FHOCR *FHOCR, FHImg *query){
    if ( FHOCR->voting==FHOCR_HARD_VOTING ){
        return(FHOCRHardVoting(FHOCR, query));
    }
    else if ( FHOCR->voting==FHOCR_SOFT_VOTING ){
        return(FHOCRSoftVoting(FHOCR, query));
    }
    else if ( FHOCR->voting==FHOCR_MAX_VOTING ){
        return(FHOCRMaxVoting(FHOCR, query));
    }
    else{ return(-1); }
}


int FHOCRRecog(FHOCR *FHOCR, FHImg *query){
    int     i;
    int     ans;
    int     num_class = FHOCRNumClass(FHOCR);
    int     cnt = 0;
    float   *prob_tmp;

    FHOCR->max_prob = 0;

    if ( 0==FHOCR->prob ){
        FHOCR->prob = FHMalloc(float, FHOCR->class_table.num);
    }
    if ( 0==FHOCR->ans ){
        FHOCR->ans = FHMalloc(int, FHOCR->class_table.num);
    }

    for ( i=0 ; i<num_class ; i++ ){ FHOCR->ans[i] = i; }

    if ( FHOCRVoting(FHOCR, query) ){ return(-1); }

    prob_tmp = FHMalloc(float, num_class);
    FHCp(FHOCR->prob, prob_tmp, num_class);

    cnt = 0;
    for ( i=0 ; i<num_class ; i++ ){
        if ( prob_tmp[i] ){ cnt++; }
    }
    cnt = FHMin(100, cnt);

    for ( i=0 ; i<cnt ; i++ ){
        ans = FHArgmaxF(prob_tmp, num_class);
        FHOCR->prob[i] = prob_tmp[ans];
        prob_tmp[ans] = 0;
        FHOCR->ans[i] = FHOCR->class_table.e[ans];
    }
    for ( i=cnt ; i<num_class ; i++ ){
        FHOCR->prob[i] = 0;
        FHOCR->ans[i] = -1;
    }
    // FHSortFI(FHOCR->prob, FHOCR->ans, num_class);

    // FHOCRVar = FHVarF(FHOCR->prob, FHOCR->class_table.num);

    FHFree(prob_tmp);
    return(FHOCR->ans[0]);
}


int FHOCRRecogThresh(FHOCR *FHOCR, FHImg *query, float thresh){
    int     i;
    int     p = 1;
    int     q = 3;
    int     times = p * q;
    int     ret;
    int     argmaxno = 0;
    int     *ans = FHMalloc(int, times);
    float   *ans_prob = FHMalloc(float, times);
    int     *second_ans = FHMalloc(int, times);
    int     *third_ans = FHMalloc(int, times);
    int     id_ans = 0;
    int     final_ans;
    int     pre_ans1, pre_ans2, pre_ans3;
    float   pre_prob;
    FHImg   imgtmp, imgbuf[3], imgtmp0;

    FHImgInit(&imgtmp);
    FHImgInit(&imgtmp0);
    FHImgInit(&imgbuf[0]);
    FHImgInit(&imgbuf[1]);
    FHImgInit(&imgbuf[2]);

    /* ---- Stage 0 ---- */

    pre_ans1 = ans[0] = FHOCRRecog(FHOCR, query);
    pre_prob = ans_prob[0] = FHOCR->prob[0];

    pre_ans2 = second_ans[0] = FHOCR->ans[1];
    pre_ans3 = third_ans[0] = FHOCR->ans[2];

    FHOCREta = ans_prob[0];
    if ( thresh<ans_prob[0] ){
        final_ans = ans[0];
        FHFree(ans_prob);
        FHFree(ans);
        FHFree(second_ans);
        FHFree(third_ans);
        return(final_ans);
    }

    /* ---- To next stage ---- */

    FHResizeSquare(query, &imgtmp0, 64);

    for ( i=0 ; i<times ; i++ ){
        ret = 0;
        if ( i%3==0 ){
            FHImgCp(&imgtmp0, &imgbuf[0]);
            FHImgCp(&imgtmp0, &imgtmp);
        }
        else if ( 1<=i && i<=2 ){
            // ret = FHGaussianFilter3(&imgtmp0, &imgbuf[i], FHOCR_REC_GF[i-1]);
            ret = FHNLNorm(&imgtmp0, &imgbuf[i], 1, FHOCR_REC_NLN[i-1]);
            FHImgCp(&imgbuf[i], &imgtmp);
        }
        else if ( 4<=i && i<=5 ){
            ret = FHRotate(&imgtmp0, &imgbuf[i-3], FHOCR_REC_ROTATE[i-4]);
            FHImgCp(&imgbuf[i-3], &imgtmp);
        }
        else if ( 7<=i && i<=8 ){
            // ret = FHNLNorm(&imgtmp0, &imgbuf[i-6], 1, FHOCR_REC_NLN[i-7]);
            ret = FHAffineResize(&imgtmp0, &imgbuf[i-6], FHOCR_REC_SCALE[i-7][0], \
                        FHOCR_REC_SCALE[i-7][1]);
            FHImgCp(&imgbuf[i-6], &imgtmp);
        }
        else if ( 10<=i && i<=11 ){
            if ( 10==i ){ ret = FHDilation3(&imgtmp0, &imgbuf[i-9]); }
            if ( 10==i ){ ret = FHErosion3(&imgtmp0, &imgbuf[i-9]); }
            FHImgCp(&imgbuf[i-9], &imgtmp);
        }
        else if ( 13<=i && i<=14 ){
            ret = FHRotate(&imgtmp0, &imgbuf[i-12], FHOCR_REC_ROTATE[i-13]);
            FHImgCp(&imgbuf[i-12], &imgtmp);
        }
        else if ( 16<=i && i<=17 ){
            ret = FHAffineResize(&imgtmp0, &imgbuf[i-15], FHOCR_REC_SCALE[i-16][0], \
                        FHOCR_REC_SCALE[i-16][1]);
            FHImgCp(&imgbuf[i-15], &imgtmp);
        }
        // else if ( 10<=i && i<=11 ){
        //     ret = FHRandomFilter3(&imgtmp0, &imgbuf[i-9]);
        //     FHImgCp(&imgbuf[i-9], &imgtmp);
        // }
        // if ( 8<=i && i<=10 ){
        //     ret = FHSkewH(&imgtmp0, &imgbuf[i-8], FHOCR_REC_SKEW[i-8]);
        //     FHImgCp(&imgbuf[i-8], &imgtmp);
        // }
        // if ( 11<=i && i<=13 ){
        //     ret = FHSkewV(&imgtmp0, &imgbuf[i-11], FHOCR_REC_SKEW[i-11]);
        //     FHImgCp(&imgbuf[i-11], &imgtmp);
        // }
        // if ( 14<=i && i<=15 ){
        //     if ( i==14 ){ FHImgCp(&imgtmp0, &imgbuf[i-14]); }
        //     else if ( i==15 ){ FHNLNorm(&imgtmp0, &imgbuf[i-14], 1, 0.5); }
        // }

        // printf("%d\n", i);
        assert(ret==0);
        if ( ret ){
            FHImgFinish(&imgtmp);
            FHImgFinish(&imgtmp0);
            FHImgFinish(&imgbuf[0]);
            FHImgFinish(&imgbuf[1]);
            FHImgFinish(&imgbuf[2]);
            FHFree(ans);
            FHFree(second_ans);
            FHFree(third_ans);
            FHFree(ans_prob);
            return(-1);
        }

        if ( i%3==0 ){
            ans[i] = pre_ans1;
            ans_prob[i] = pre_prob;
            second_ans[i] = pre_ans2;
            third_ans[i] = pre_ans3;
        }
        else{
            ans[i] = FHOCRRecog(FHOCR, &imgtmp);
            ans_prob[i] = FHOCR->prob[0];

            /* Calculation of the diff between first and second probability (option)  */
            // ans_prob[i] -= FHOCR->prob[1];

            second_ans[i] = FHOCR->ans[1];
            third_ans[i] = FHOCR->ans[2];
        }
        if ( i==2 ){
            argmaxno = FHArgmaxF(ans_prob, 3);
            FHImgCp(&imgbuf[argmaxno], &imgtmp0); 
            pre_ans1 = ans[argmaxno];
            pre_ans2 = second_ans[argmaxno];
            pre_ans3 = third_ans[argmaxno];
            pre_prob = ans_prob[argmaxno];
        }
        else if ( i==5 ){
            argmaxno = FHArgmaxF(ans_prob+3, 3);
            FHImgCp(&imgbuf[argmaxno], &imgtmp0);
            argmaxno += 3;
            pre_ans1 = ans[argmaxno];
            pre_ans2 = second_ans[argmaxno];
            pre_ans3 = third_ans[argmaxno];
            pre_prob = ans_prob[argmaxno];
        }
        else if ( i==8 ){
            argmaxno = FHArgmaxF(ans_prob+6, 3);
            FHImgCp(&imgbuf[argmaxno], &imgtmp0);
            argmaxno += 6;
            pre_ans1 = ans[argmaxno];
            pre_ans2 = second_ans[argmaxno];
            pre_ans3 = third_ans[argmaxno];
            pre_prob = ans_prob[argmaxno];
        }
        else if ( i==11 ){
            argmaxno = FHArgmaxF(ans_prob+9, 3);
            FHImgCp(&imgbuf[argmaxno], &imgtmp0);
            argmaxno += 9;
            pre_ans1 = ans[argmaxno];
            pre_ans2 = second_ans[argmaxno];
            pre_ans3 = third_ans[argmaxno];
            pre_prob = ans_prob[argmaxno];
        }
        assert(pre_prob<=ans_prob[argmaxno]);
        // else if ( i==10 ){
        //     argmaxno = FHArgmaxF(ans_prob+8, 3);
        //     FHImgCp(&imgbuf[argmaxno], &imgtmp0);
        // }
        // else if ( i==13 ){
        //     argmaxno = FHArgmaxF(ans_prob+11, 3);
        //     FHImgCp(&imgbuf[argmaxno], &imgtmp0);
        // }
        if ( (pre_prob>thresh && (i==2 || i==5 || i==8 || i==11)) \
                || i==times-1 ){ 
            FHImgFinish(&imgtmp);
            FHImgFinish(&imgtmp0);
            FHImgFinish(&imgbuf[0]);
            FHImgFinish(&imgbuf[1]);
            FHImgFinish(&imgbuf[2]);

            FHOCR->ans[0] = final_ans = pre_ans1;
            assert(final_ans>=0);
            FHOCR->ans[1] = pre_ans2;
            FHOCR->ans[2] = pre_ans3;
            FHOCR->prob[0] = pre_prob;

            FHFree(ans);
            FHFree(second_ans);
            FHFree(third_ans);
            FHFree(ans_prob);
            return(final_ans); 
        }
    }
    id_ans = 0;
    for ( i=1 ; i<times ; i++ ){
        if ( ans_prob[id_ans] < ans_prob[i] ){ id_ans = i; }
    }
    final_ans = ans[id_ans];
    FHOCR->ans[1] = second_ans[id_ans];
    FHOCR->ans[2] = third_ans[id_ans];
    FHFree(ans);
    FHFree(ans_prob);
    FHFree(second_ans);
    FHFree(third_ans);
    FHImgFinish(&imgtmp);
    FHImgFinish(&imgtmp0);
    FHImgFinish(&imgbuf[0]);
    FHImgFinish(&imgbuf[1]);
    FHImgFinish(&imgbuf[2]);
    return(final_ans);
}


/* --------------------------------------------------------
 * Multi-Normalization Recognizer
 * -------------------------------------------------------- */

int FHOCRMNRecog(FHOCR *FHOCR, FHImg *query, int num_norm){
    int     i;
    int     *ans = 0;
    int     final_ans;
    int     num_class = FHOCRNumClass(FHOCR);
    float   maxalpha = 4.0;
    // float   d_alpha = maxalpha / num_norm;
    float   d_alpha = 1.0;
    float   alpha;
    float   *hist;
    float   *prob;
    FHImg   imgtmp, img64;

    FHImgInit(&img64);

    ans = FHMalloc(int, num_norm);
    hist = FHMalloc(float, num_class);
    prob = FHMalloc(float, num_norm);
    FHSubAll(hist, num_class, 0);
    FHSubAll(prob, num_norm, 0);

    FHResizeSquare(query, &img64, 64);

    for ( i=0 ; FHOCR->RIFeat[i]!=-1 && i<50 ; i++ ){
        if ( FHOCR->RIFeat[i]==FHOCR_NLN ){ FHOCR->RIFeat[i] = FHOCR_NONE; }
    }

    for ( i=0 ; i<num_norm ; i++ ){
        alpha = d_alpha * i;
        FHImgInit(&imgtmp);
        if ( 0==i ){
            FHImgCp(&img64, &imgtmp);
        }
        else{
            FHNLNorm(&img64, &imgtmp, 1, alpha);
        }
        ans[i] = FHOCRRecog(FHOCR, &imgtmp);
        prob[i] = FHOCR->max_prob;
        FHImgFinish(&imgtmp);
    }

    for ( i=0 ; FHOCR->RIFeat[i]!=-1 && i<50 ; i++ ){
        if ( FHOCR->RIFeat[i]==FHOCR_NONE ){ FHOCR->RIFeat[i] = FHOCR_NLN; }
    }

    // for ( i=0 ; i<num_norm ; i++ ){
    //     labelid = FHSetSearch(&FHOCR->class_table, ans[i]);
    //     hist[labelid]++;
    // }

    // final_ans = FHArgmaxF(hist, num_class);
    // final_ans = FHOCR->class_table.e[final_ans];

    final_ans = ans[FHArgmaxF(prob, num_norm)];

    FHFree(ans);
    FHFree(hist);
    FHFree(prob);

    FHImgFinish(&img64);
    return(final_ans);
}


/* --------------------------------------------------------
 * Single-Normalization Recognizer
 * -------------------------------------------------------- */

int FHOCRSNRecog(FHOCR *FHOCR, FHImg *query, float alpha){
    int     final_ans;
    FHImg   imgtmp;

    FHImgInit(&imgtmp);
    FHNLNorm(query, &imgtmp, 1, alpha);
    final_ans =  FHOCRRecog(FHOCR, &imgtmp);
    FHImgFinish(&imgtmp);

    return(final_ans);
}


/* --------------------------------------------------------
 * IO
 * -------------------------------------------------------- */

int FHOCRWrite(FHOCR *FHOCR, FILE *fp){
    int     i, j;
    int     use_rf = 0, use_ra = 0, use_rn = 0, use_rm = 0, use_rs = 0;

    /* ---- MAGIC WORD ---- */

    if ( 0>=fprintf(fp, "FHOCR\n") ){ return(-1); }

    /* ---- Dataset ---- */

    if ( 0>=fprintf(fp, "%s\n", FHOCR->traindata_name) ){ return(-1); }

    /* ---- Parameters ---- */

    if ( 1!=fwrite(&FHOCR->K, sizeof(int), 1, fp) ){ return(-1); }
    if ( 1!=fwrite(&FHOCR->T, sizeof(int), 1, fp) ){ return(-1); }
    if ( 1!=fwrite(&FHOCR->use_nns, sizeof(int), 1, fp) ){ return(-1); }
    if ( 1!=fwrite(&FHOCR->use_svm, sizeof(int), 1, fp) ){ return(-1); }
    if ( 1!=fwrite(&FHOCR->use_mlp, sizeof(int), 1, fp) ){ return(-1); }
    if ( 1!=fwrite(&FHOCR->feature, sizeof(int), 1, fp) ){ return(-1); }

    /* ---- SSCD ---- */

    for ( i=0 ; ; i++ ){
        if ( 1!=fwrite(&FHOCR->SSCD[i], sizeof(int), 1, fp) ){ return(-1); }
        if ( -1==FHOCR->SSCD[i] ){ break; }
    }

    /* ---- RI-Feature ---- */

    FHOCR->num_kernel = 0;
    for ( i=0 ; ; i++ ){
        if ( 1!=fwrite(&FHOCR->RIFeat[i], sizeof(int), 1, fp) ){ return(-1); }
        if ( -1==FHOCR->RIFeat[i] ){ break; }
        if ( FHOCR_RF==FHOCR->RIFeat[i] ){ use_rf = 1; FHOCR->num_kernel++; }
        if ( FHOCR_RA==FHOCR->RIFeat[i] ){ use_ra = 1; }
        if ( FHOCR_RN==FHOCR->RIFeat[i] ){ use_rn = 1; }
        if ( FHOCR_RM==FHOCR->RIFeat[i] ){ use_rm = 1; }
        if ( FHOCR_RS==FHOCR->RIFeat[i] ){ use_rs = 1; }
    }

    /* ---- Classifiers ---- */

    for ( i=0 ; i<FHOCR->T && FHOCR->use_nns ; i++ ){
        if ( FHVDicWrite(&FHOCR->vdic[i], fp) ){ return(-1); }
    }
    for ( i=0 ; i<FHOCR->T && FHOCR->use_svm ; i++ ){
        if ( FHSVMWrite(&FHOCR->svm[i], fp) ){ return(-1); }
    }
    for ( i=0 ; i<FHOCR->T && FHOCR->use_mlp ; i++ ){
        if ( FHMLPWrite(&FHOCR->mlp[i], fp) ){ return(-1); }
    }

    /* ---- Class table ---- */

    if ( FHSetWrite(&FHOCR->class_table, fp) ){ return(-1); }

    /* ---- Preprocessing ---- */

    if ( use_rf ){
        for ( i=0 ; i<FHOCR->num_kernel*FHOCR->T ; i++ ){
            for ( j=0 ; j<3 ; j++ ){
                if ( 3!=fwrite(FHOCR->kernel[i][j], sizeof(float), 3, fp) ){ return(-1); }
            }
        }
    }

    if ( use_ra ){
        for ( i=0 ; i<FHOCR->T ; i++ ){
            if ( 2!=fwrite(FHOCR->mat[i][0], sizeof(float), 2, fp) ){ return(-1); }
            if ( 2!=fwrite(FHOCR->mat[i][1], sizeof(float), 2, fp) ){ return(-1); }
        }
    }

    if ( use_rn ){
        if ( (unsigned)FHOCR->T*2!=fwrite(FHOCR->rn_alpha, \
                    sizeof(float), FHOCR->T*2, fp) ){ return(-1); }
    }

    if ( use_rm ){
        for ( i=0 ; i<FHOCR->T ; i++ ){
            FHImgWrite(&FHOCR->mask[i], fp);
        }
    }

    if ( use_rs ){
        for ( i=0 ; i<FHOCR->T ; i++ ){
            if ( 2!=fwrite(FHOCR->rs[i], sizeof(float), 2, fp) ){ return(-1); }
        }
    }

    /* ---- RF HOG feature ---- */

    if ( FHOCR->feature==FHOCR_RFHOG ){
        for ( i=0 ; i<FHOCR->T*48 ; i++ ){
            for ( j=0 ; j<3 ; j++ ){
                if ( 3!=fwrite(FHOCR->rfhog[i][j], sizeof(float), 3, fp) ){ return(-1); }
            }
        }
    }

    return(0);
}


int FHOCRSave(FHOCR *FHOCR, char *fname){
    FILE    *fp;

    fp = fopen(fname, "wb");
    if ( NULL==fp ){ return(-1); }
    if ( FHOCRWrite(FHOCR, fp) ){ fclose(fp); return(-1); }
    if ( EOF==fclose(fp) ){ return(-1); }
    return(0);
}


int FHOCRRead(FHOCR *FHOCR, FILE *fp){
    int     i, j;
    int     use_rf = 0, use_ra = 0, use_rn = 0, use_rm = 0, use_rs = 0;
    char    bufstr[256];

    if ( NULL==FHRead1Line(fp, bufstr, 256) ){ return(-1); }
    if ( strcmp(bufstr, "FHOCR")!=0 ){ return(-1); }
    if ( NULL==FHRead1Line(fp, FHOCR->traindata_name, 64) ){ return(-1); }

    /* ---- Parameters ---- */

    if ( 1!=fread(&FHOCR->K, sizeof(int), 1, fp) ){ return(-1); }
    if ( FHOCR->K<=0 ){ return(-1); }
    if ( 1!=fread(&FHOCR->T, sizeof(int), 1, fp) ){ return(-1); }
    if ( FHOCR->T<=0 ){ return(-1); }
    if ( 1!=fread(&FHOCR->use_nns, sizeof(int), 1, fp) ){ return(-1); }
    if ( FHOCR->use_nns!=0 && FHOCR->use_nns!=1 ){ return(-1); }
    if ( 1!=fread(&FHOCR->use_svm, sizeof(int), 1, fp) ){ return(-1); }
    if ( FHOCR->use_svm!=0 && FHOCR->use_svm!=1 ){ return(-1); }
    if ( 1!=fread(&FHOCR->use_mlp, sizeof(int), 1, fp) ){ return(-1); }
    if ( FHOCR->use_mlp!=0 && FHOCR->use_mlp!=1 ){ return(-1); }
    if ( 1!=fread(&FHOCR->feature, sizeof(int), 1, fp) ){ return(-1); }

    /* ---- SSCD ---- */

    for ( i=0 ; ; i++ ){
        if ( 1!=fread(&FHOCR->SSCD[i], sizeof(int), 1, fp) ){ return(-1); }
        if ( -1==FHOCR->SSCD[i] ){ break; }
    }

    /* ---- RI-Feature ---- */

    FHOCR->num_kernel = 0;
    for ( i=0 ; ; i++ ){
        if ( 1!=fread(&FHOCR->RIFeat[i], sizeof(int), 1, fp) ){ return(-1); }
        if ( -1==FHOCR->RIFeat[i] ){ break; }
        if ( FHOCR->RIFeat[i]<0 ){ return(-1); }
        if ( FHOCR_RF==FHOCR->RIFeat[i] ){ use_rf = 1; FHOCR->num_kernel++; }
        if ( FHOCR_RA==FHOCR->RIFeat[i] ){ use_ra = 1; }
        if ( FHOCR_RN==FHOCR->RIFeat[i] ){ use_rn = 1; }
        if ( FHOCR_RM==FHOCR->RIFeat[i] ){ use_rm = 1; }
        if ( FHOCR_RS==FHOCR->RIFeat[i] ){ use_rs = 1; }
    }

    /* ---- Classifiers ---- */

    if ( FHOCR->use_nns ){ FHOCR->vdic = new FHVDic[FHOCR->T]; }
    if ( FHOCR->use_svm ){ FHOCR->svm = new FHSVM[FHOCR->T]; }
    if ( FHOCR->use_mlp ){ FHOCR->mlp = new FHMLP[FHOCR->T]; }

    for ( i=0 ; i<FHOCR->T && FHOCR->use_nns ; i++ ){
        if ( FHVDicRead(&FHOCR->vdic[i], fp) ){ return(-1); }
    }
    for ( i=0 ; i<FHOCR->T && FHOCR->use_svm ; i++ ){
        if ( FHSVMRead(&FHOCR->svm[i], fp) ){ return(-1); }
    }
    for ( i=0 ; i<FHOCR->T && FHOCR->use_mlp ; i++ ){
        if ( FHMLPRead(&FHOCR->mlp[i], fp) ){ return(-1); }
    }

    if ( FHOCR->use_mlp ){ FHOCR->hidden = FHOCR->mlp[0].hidden_dim; }

    /* ---- Class table ---- */

    if ( FHSetRead(&FHOCR->class_table, fp) ){ return(-1); }

    /* ---- Random obj ---- */

    if ( use_rf ){
        FHOCR->kernel = new float[FHOCR->num_kernel*FHOCR->T][3][3];
        for ( i=0 ; i<FHOCR->num_kernel*FHOCR->T ; i++ ){
            for ( j=0 ; j<3 ; j++ ){
                if ( 3!=fread(FHOCR->kernel[i][j], sizeof(float), 3, fp) ){ return(-1); }
            }
        }
    }

    if ( use_ra ){
        FHOCR->mat = new float[FHOCR->T][2][2];
        for ( i=0 ; i<FHOCR->T ; i++ ){
            if ( 2!=fread(FHOCR->mat[i][0], sizeof(float), 2, fp) ){ return(-1); }
            if ( 2!=fread(FHOCR->mat[i][1], sizeof(float), 2, fp) ){ return(-1); }
        }
    }

    if ( use_rn ){
        FHOCR->rn_alpha = new float[FHOCR->T*2];
        if ( (unsigned)FHOCR->T*2!=fread(FHOCR->rn_alpha, \
                    sizeof(float), FHOCR->T*2, fp) ){ return(-1); }
    }

    if ( use_rm ){
        FHOCR->mask = new FHImg[FHOCR->T];
        for ( i=0 ; i<FHOCR->T ; i++ ){
            FHImgRead(&FHOCR->mask[i], fp);
        }
    }

    if ( use_rs ){
        for ( i=0 ; i<FHOCR->T ; i++ ){
            if ( 2!=fread(FHOCR->rs[i], sizeof(float), 2, fp) ){ return(-1); }
        }
    }

    /* ---- RF HOG feature ---- */

    if ( FHOCR->feature==FHOCR_RFHOG ){
        FHOCR->rfhog = new float[FHOCR->T*48][3][3];
        for ( i=0 ; i<FHOCR->T*48 ; i++ ){
            for ( j=0 ; j<3 ; j++ ){
                if ( 3!=fread(FHOCR->rfhog[i][j], sizeof(float), 3, fp) ){ return(-1); }
            }
        }
    }

    /* ---- CNN feature ---- */

    if ( FHOCR->feature==FHOCR_CNNFEAT ){
        FHOCR->cnn = new mojo::network;
        FHOCR->cnn->read(FHOCR->CNN_FILE);
    }

    return(0);
}


int FHOCRLoad(FHOCR *FHOCR, char *fname){
    FILE    *fp;

    fp = fopen(fname, "rb");
    if ( NULL==fp ){ return(-1); }
    if ( FHOCRRead(FHOCR, fp) ){ fclose(fp); return(-1); }
    if ( EOF==fclose(fp) ){ return(-1); }
    return(0);
}


/* --------------------------------------------------------
 * Parameter loading
 * -------------------------------------------------------- */

static int SetParam(FHOCR *FHOCR, int n, char bufstr[]){
    int     i;
    int     errflg = 0;
    int     num_ri;
    char    ristr[256];
    char    *strlist[256];
    switch ( n ){
        case 0:
            break;
        case    1:
            errflg = sscanf(bufstr, "K=%d", &FHOCR->K);
            break;
        case    2:
            errflg = sscanf(bufstr, "T=%d", &FHOCR->T);
            break;
        case    3:
            errflg = sscanf(bufstr, "USE_NNS=%d", &FHOCR->use_nns);
            break;
        case    4:
            errflg = sscanf(bufstr, "USE_SVM=%d", &FHOCR->use_svm);
            break;
        case    5:
            errflg = sscanf(bufstr, "USE_MLP=%d", &FHOCR->use_mlp);
            break;
        case    6:
            errflg = sscanf(bufstr, "SVM_C=%f", &FHOCR->SVM_C);
            break;
        case    7:
            errflg = sscanf(bufstr, "RI-Feature=%s", ristr);
            num_ri = FHStrSplit(ristr, "-", strlist, 20);

            for ( i=0 ; i<num_ri ; i++ ){
                if ( strcmp(strlist[i], "MSR_16")==0 ){ 
                    FHOCR->RIFeat[i] = FHOCR_MSR_16; 
                }
                else if ( strcmp(strlist[i], "MSR_32")==0 ){ 
                    FHOCR->RIFeat[i] = FHOCR_MSR_32; 
                }
                else if ( strcmp(strlist[i], "MSR_64")==0 ){ 
                    FHOCR->RIFeat[i] = FHOCR_MSR_64; 
                }
                else if ( strcmp(strlist[i], "MSR_16_32")==0 ){ 
                    FHOCR->RIFeat[i] = FHOCR_MSR_16_32; 
                }
                else if ( strcmp(strlist[i], "MSR_32_64")==0 ){ 
                    FHOCR->RIFeat[i] = FHOCR_MSR_32_64; 
                }
                else if ( strcmp(strlist[i], "MSR_16_64")==0 ){ 
                    FHOCR->RIFeat[i] = FHOCR_MSR_16_64; 
                }
                else if ( strcmp(strlist[i], "MSR_16_32_64")==0 ){ 
                    FHOCR->RIFeat[i] = FHOCR_MSR_16_32_64; 
                }
                else if ( strcmp(strlist[i], "RA")==0 ){ 
                    FHOCR->RIFeat[i] = FHOCR_RA; 
                }
                else if ( strcmp(strlist[i], "RF")==0 ){ 
                    FHOCR->RIFeat[i] = FHOCR_RF; 
                }
                else if ( strcmp(strlist[i], "MF")==0 ){ 
                    FHOCR->RIFeat[i] = FHOCR_MF; 
                }
                else if ( strcmp(strlist[i], "GCN")==0 ){ 
                    FHOCR->RIFeat[i] = FHOCR_GCN; 
                }
                else if ( strcmp(strlist[i], "NLN")==0 ){ 
                    FHOCR->RIFeat[i] = FHOCR_NLN; 
                }
                else if ( strcmp(strlist[i], "INN")==0 ){ 
                    FHOCR->RIFeat[i] = FHOCR_INN; 
                }
                else if ( strcmp(strlist[i], "LRN")==0 ){ 
                    FHOCR->RIFeat[i] = FHOCR_LRN; 
                }
                else if ( strcmp(strlist[i], "RGCN")==0 ){ 
                    FHOCR->RIFeat[i] = FHOCR_RGCN; 
                }
                else if ( strcmp(strlist[i], "RNLN")==0 ){ 
                    FHOCR->RIFeat[i] = FHOCR_RNLN; 
                }
                else if ( strcmp(strlist[i], "RN")==0 ){ 
                    FHOCR->RIFeat[i] = FHOCR_RN; 
                }
                else if ( strcmp(strlist[i], "RM")==0 ){ 
                    FHOCR->RIFeat[i] = FHOCR_RM; 
                }
                else if ( strcmp(strlist[i], "RS")==0 ){ 
                    FHOCR->RIFeat[i] = FHOCR_RS; 
                }
                else if ( strcmp(strlist[i], "AVP")==0 ){ 
                    FHOCR->RIFeat[i] = FHOCR_AVP; 
                }
                else if ( strcmp(strlist[i], "SF")==0 ){
                    FHOCR->RIFeat[i] = FHOCR_SF; 
                }
                else{ errflg = -1; break; }
            }
            FHOCR->RIFeat[num_ri] = -1;
            break;
        case    8:
            errflg = sscanf(bufstr, "SSCD=%s", ristr);
            num_ri = FHStrSplit(ristr, "-", strlist, 20);

            for ( i=0 ; i<num_ri ; i++ ){
                if ( strcmp(strlist[i], "RA")==0 ){ 
                    FHOCR->SSCD[i] = FHSSCD_RA; 
                }
                else if ( strcmp(strlist[i], "RS")==0 ){ 
                    FHOCR->SSCD[i] = FHSSCD_RS; 
                }
                else if ( strcmp(strlist[i], "RM")==0 ){ 
                    FHOCR->SSCD[i] = FHSSCD_RM; 
                }
                else if ( strcmp(strlist[i], "DE")==0 ){ 
                    FHOCR->SSCD[i] = FHSSCD_DE; 
                }
                else if ( strcmp(strlist[i], "AC")==0 ){ 
                    FHOCR->SSCD[i] = FHSSCD_AC; 
                }
                else if ( strcmp(strlist[i], "GN")==0 ){ 
                    FHOCR->SSCD[i] = FHSSCD_GN; 
                }
                else if ( strcmp(strlist[i], "GF")==0 ){ 
                    FHOCR->SSCD[i] = FHSSCD_GF; 
                }
                else if ( strcmp(strlist[i], "RF")==0 ){ 
                    FHOCR->SSCD[i] = FHSSCD_RF; 
                }
                else if ( strcmp(strlist[i], "GOGH")==0 ){ 
                    FHOCR->SSCD[i] = FHSSCD_GOGH; 
                }
                else if ( strcmp(strlist[i], "BL")==0 ){ 
                    FHOCR->SSCD[i] = FHSSCD_BL; 
                }
                else if ( strcmp(strlist[i], "NLD")==0 ){ 
                    FHOCR->SSCD[i] = FHSSCD_NLD; 
                }
                else if ( strcmp(strlist[i], "PT")==0 ){ 
                    FHOCR->SSCD[i] = FHSSCD_PT; 
                }
                else if ( strcmp(strlist[i], "RR")==0 ){ 
                    FHOCR->SSCD[i] = FHSSCD_RR; 
                }
                else{ errflg = -1; break; }
            }
            FHOCR->SSCD[num_ri] = -1;
            break;
        case    9:
            errflg = sscanf(bufstr, "Feature=%s", ristr);
            if ( strcmp(ristr, "HOG")==0 ){
                FHOCR->feature = FHOCR_HOG;
            }
            else if ( strcmp(ristr, "CSLBP")==0 ){
                FHOCR->feature = FHOCR_CSLBP;
            }
            else if ( strcmp(ristr, "ROWFEAT")==0 ){
                FHOCR->feature = FHOCR_ROWFEAT;
            }
            else if ( strcmp(ristr, "MRF")==0 ){
                FHOCR->feature = FHOCR_MRF;
            }
            else if ( strcmp(ristr, "HOG16")==0 ){
                FHOCR->feature = FHOCR_HOG16;
            }
            else if ( strcmp(ristr, "RFHOG")==0 ){
                FHOCR->feature = FHOCR_RFHOG;
            }
            else if ( strcmp(ristr, "CNN")==0 ){
                FHOCR->feature = FHOCR_CNNFEAT;
            }
            break;
        case    10:
            errflg = sscanf(bufstr, "Hidden=%d", &FHOCR->hidden);
            break;
        case    11:
            errflg = sscanf(bufstr, "TrainDataPath=%s", FHOCR->traindata_path);
            break;
        case    12:
            errflg = sscanf(bufstr, "MaxEpoch=%d\n", &FHOCR->max_epoch);
            break;
        default:
            errflg = -1;
            break;
    }

    if ( errflg<0 ){ return(-1); }
    return(0);
}


int FHOCRLoadParam(FHOCR *FHOCR, const char *fname){
    int     n;
    FILE    *fp;
    char    bufstr[128];
    fp = fopen(fname, "r");
    if ( NULL==fp ){ return(-1); }
    while ( FHRead1Line(fp, bufstr, 128) ){
        if ( bufstr[0]=='#' || bufstr[0]=='\0' ){ n = 0; }
        else if ( strstr(bufstr, "K=") ){ n = 1; }
        else if ( strstr(bufstr, "T=") ){ n = 2; }
        else if ( strstr(bufstr, "USE_NNS=") ){ n = 3; }
        else if ( strstr(bufstr, "USE_SVM=") ){ n = 4; }
        else if ( strstr(bufstr, "USE_MLP=") ){ n = 5; }
        else if ( strstr(bufstr, "SVM_C=") ){ n = 6; }
        else if ( strstr(bufstr, "RI-Feature=") ){ n = 7; }
        else if ( strstr(bufstr, "SSCD=") ){ n = 8; }
        else if ( strstr(bufstr, "Feature=") ){ n = 9; }
        else if ( strstr(bufstr, "Hidden=") ){ n = 10; }
        else if ( strstr(bufstr, "TrainDataPath=") ){ n = 11; }
        else if ( strstr(bufstr, "MaxEpoch=") ){ n = 12; }
        else{ n = -1; }
        if ( SetParam(FHOCR, n, bufstr) ){ fclose(fp); return(-1); }
    }
    if ( EOF==fclose(fp) ){ return(-1); }
    return(0);
}
