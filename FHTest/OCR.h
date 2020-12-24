#define     SHOW_IMAGE(x)   do{ \
    FHImgSave(x, "/home/fuma/Pictures/tmp.bmp"); \
    getchar(); \
}while(0)


int Train(int argc, char **argv){
    // int     rndtmp = FHRndI(0, 1);
    FHOCR   FHOCR;
    char    fname[128];
    char    ocrdir[64];
    char    datadir[64];
    const char *homepass = getenv("HOME");
    time_t  current_time;

    FHOCRInit(&FHOCR);

    if ( argc!=3 ){
        fprintf(stderr, "Error: Invalid argument\n");
        PrintUsage();
        FHOCRFinish(&FHOCR);
        return(1);
    }

    // srand(2);

    /* ---- Load Parameter ---- */

    if ( FHOCRLoadParam(&FHOCR, argv[2]) ){
        fprintf(stderr, "Error: Cannot load %s\n", argv[2]);
        FHOCRFinish(&FHOCR);
        return(1);
    }

    /* ---- mkdir ---- */

    sprintf(ocrdir, "%s/.FH/", homepass);
    if ( FHMkdir(ocrdir) ){ 
        fprintf(stderr, "Cannot make %s\n", ocrdir);
        FHOCRFinish(&FHOCR);
        return(1); 
    }

    /* ---- Training ---- */

    if ( FHOCRTrain(&FHOCR) ){
        fprintf(stderr, "error: Cannot Train\n");
        FHOCRFinish(&FHOCR);
        return(1);
    }

    sprintf(datadir, "%s/.FH/%s", homepass, FHOCR.traindata_name);
    if ( FHMkdir(datadir) ){ 
        fprintf(stderr, "Cannot make %s\n", datadir);
        FHOCRFinish(&FHOCR);
        return(1); 
    }

    current_time = time(NULL);

    sprintf(fname, "%s/K%d_T%d_NNS%d_SVM%d_MLP%d_%d.fhocr", datadir, FHOCR.K, \
            FHOCR.T, FHOCR.use_nns, FHOCR.use_svm, FHOCR.use_mlp, (int)current_time);

    printf("Saving on %s ... \n", fname);
    
    if ( FHOCRSave(&FHOCR, fname) ){
        fprintf(stderr, "error: cannot save to \"%s\"\n", fname);
        FHOCRFinish(&FHOCR);
        return(1);
    }

    printf("Saving on \"%s\" ... done\n", fname);

    FHOCRFinish(&FHOCR);

    return(0);
}


int LineRecog(int argc, char **argv){
    int         i;
    int         no;
    char        strbuf[64];
    char        correct[64];
    FHOCR       FHOCR;
    FHImg       img, charimg;
    FHStrBox    list;
    float       thresh = 0;
    char    param_str[][64] = {
        "THRESH=",
    };

    FHOCRInit(&FHOCR);
    FHImgInit(&img);
    FHImgInit(&charimg);
    FHStrBoxInit(&list);

    if ( !(4<=argc && argc<=5) ){
        fprintf(stderr, "Error: Invalid arguments\n");
        PrintUsage();
        return(1);
    }

    for ( i=4 ; i<argc ; i++ ){
        if ( NULL!=strstr(argv[i], param_str[0]) ){
            if ( 1!=sscanf(argv[i], "THRESH=%f", &thresh) ){ 
                fprintf(stderr, "Error: Invalid arguments\n");
                PrintUsage();
                return(1); 
            }
        }
    }

    if ( FHOCRLoad(&FHOCR, argv[2]) ){
        fprintf(stderr, "Error: Cannot load %s\n", argv[2]);
        return(1);
    }

    if ( FHImgLoad(&img, argv[3]) ){
        fprintf(stderr, "Error: Cannot load %s\n", argv[3]);
        return(1);
    }

    if ( FHReadUnicode(&list, 16) ){
        fprintf(stderr, "Error: Cannot read unicode_jp\n");
        return(1);
    }

    FHOCR.voting = FHOCR_HARD_VOTING;

    while ( 1 ){
        if ( FHSegmentH(&img, &charimg, &img) ){ break; }
        sprintf(strbuf, "%04x", FHOCRRecogThresh(&FHOCR, &charimg, thresh));
        FHToupper(strbuf, strbuf, 4);
        no = FHStrBoxFind(&list, strbuf);
        if ( no==-1 ){
            fprintf(stderr, "Error: Unicode label error\n");
            return(1);
        }
        sscanf(list.str[no], "%s %s", strbuf, correct);
        printf("%s ", correct);
    }
    puts("");

    FHImgFinish(&img);
    FHImgFinish(&charimg);
    FHStrBoxFinish(&list);
    FHOCRFinish(&FHOCR);

    return(0);
}


int Recog(int argc, char **argv){
    int         i;
    int         no;
    char        strbuf[64];
    char        correct[64];
    FHOCR       FHOCR;
    FHImg       img;
    FHStrBox    list;
    float       thresh = 0;
    char    param_str[][64] = {
        "THRESH=",
    };

    FHOCRInit(&FHOCR);
    FHImgInit(&img);
    FHStrBoxInit(&list);

    if ( !(4<=argc && argc<=5) ){
        fprintf(stderr, "Error: Invalid arguments\n");
        PrintUsage();
        return(1);
    }

    for ( i=4 ; i<argc ; i++ ){
        if ( NULL!=strstr(argv[i], param_str[0]) ){
            if ( 1!=sscanf(argv[i], "THRESH=%f", &thresh) ){ 
                fprintf(stderr, "Error: Invalid arguments\n");
                PrintUsage();
                return(1); 
            }
        }
    }

    if ( FHOCRLoad(&FHOCR, argv[2]) ){
        fprintf(stderr, "Error: Cannot load %s\n", argv[2]);
        return(1);
    }

    if ( FHImgLoad(&img, argv[3]) ){
        fprintf(stderr, "Error: Cannot load %s\n", argv[3]);
        return(1);
    }

    if ( FHReadUnicode(&list, 16) ){
        fprintf(stderr, "Error: Cannot read unicode_jp\n");
        return(1);
    }

    FHOCR.voting = FHOCR_HARD_VOTING;

    sprintf(strbuf, "%04x", FHOCRRecogThresh(&FHOCR, &img, thresh));
    FHToupper(strbuf, strbuf, 4);
    no = FHStrBoxFind(&list, strbuf);
    if ( no==-1 ){
        fprintf(stderr, "Error: Unicode label error\n");
        return(1);
    }
    sscanf(list.str[no], "%s %s", strbuf, correct);
    // printf("%s\n", correct);
    printf("First:\t%s\n", correct);

    sprintf(strbuf, "%04x", FHOCR.ans[1]);
    FHToupper(strbuf, strbuf, 4);
    no = FHStrBoxFind(&list, strbuf);
    if ( no==-1 ){
        fprintf(stderr, "Error: Unicode label error\n");
        return(1);
    }
    sscanf(list.str[no], "%s %s", strbuf, correct);
    printf("Second:\t%s\n", correct);

    sprintf(strbuf, "%04x", FHOCR.ans[2]);
    FHToupper(strbuf, strbuf, 4);
    no = FHStrBoxFind(&list, strbuf);
    if ( no==-1 ){
        fprintf(stderr, "Error: Unicode label error\n");
        return(1);
    }
    sscanf(list.str[no], "%s %s", strbuf, correct);

    printf("Third:\t%s\n", correct);

    FHImgFinish(&img);
    FHStrBoxFinish(&list);
    FHOCRFinish(&FHOCR);

    return(0);
}


int EtaTest(int argc, char **argv){
    int     i;
    int     ret;
    int     num_candidate = 1;
    int     real_T, rec_T = -1;
    int     use_nns = 0, use_svm = 0, use_mlp = 0;
    float   thresh = 0.0;
    FHOCR   FHOCR;
    FHIDic  idic;
    FILE    *fp;
    char    fname[128];
    int     voting = -1;
    char    voting_method[6];
    char    cls_name[8];
    const char *homepass = getenv("HOME");
    char    param_str[][64] = {
        "THRESH=",
        "CANDIDATE=",
        "T=",
        "VOTING=",
        "CLASSIFIER="
    };

    if ( !(4<=argc && argc<=8) ){
        fprintf(stderr, "Error: Invalid arguments\n");
        PrintUsage();
        return(1);
    }

    for ( i=4 ; i<argc ; i++ ){
        if ( NULL!=strstr(argv[i], param_str[0]) ){
            if ( 1!=sscanf(argv[i], "THRESH=%f", &thresh) ){ 
                fprintf(stderr, "Error: Invalid arguments\n");
                PrintUsage();
                return(1); 
            }
        }
        else if ( NULL!=strstr(argv[i], param_str[1]) ){
            if ( 1!=sscanf(argv[i], "CANDIDATE=%d", &num_candidate) ){ 
                fprintf(stderr, "Error: Invalid arguments\n");
                PrintUsage();
                return(1); 
            }
        }
        else if ( NULL!=strstr(argv[i], param_str[2]) ){
            if ( 1!=sscanf(argv[i], "T=%d", &rec_T) ){ 
                fprintf(stderr, "Error: Invalid arguments\n");
                PrintUsage();
                return(1); 
            }
        }
        else if ( NULL!=strstr(argv[i], param_str[3]) ){
            sscanf(argv[i], "VOTING=%s", voting_method);
            if ( strcmp(voting_method, "HARD")==0 ){
                voting = FHOCR_HARD_VOTING;
            }
            else if ( strcmp(voting_method, "SOFT")==0 ){
                voting = FHOCR_SOFT_VOTING;
            }
            else if ( strcmp(voting_method, "MAX")==0 ){
                voting = FHOCR_MAX_VOTING;
            }
            else{
                fprintf(stderr, "Error: Invalid arguments\n");
                PrintUsage();
                return(1); 
            }
        }
        else if ( NULL!=strstr(argv[i], param_str[4]) ){
            sscanf(argv[i], "CLASSIFIER=%s", cls_name);
            if ( strcmp(cls_name, "NNS")==0 ){ use_nns = 1; }
            else if ( strcmp(cls_name, "SVM")==0 ){ use_svm = 1; }
            else if ( strcmp(cls_name, "MLP")==0 ){ use_mlp = 1; }
            else{
                fprintf(stderr, "Error: Invalid arguments\n");
                PrintUsage();
                return(1); 
            }
        }
        else{
            fprintf(stderr, "Error: Invalid arguments\n");
            PrintUsage();
            return(1); 
        }
    }

    FHOCRInit(&FHOCR);
    FHIDicInit(&idic);

    if ( FHOCRLoad(&FHOCR, argv[2]) ){
        fprintf(stderr, "Error: Cannot load %s\n", argv[2]);
        return(1);
    }

    if ( FHIDicLoad(&idic, argv[3]) ){
        fprintf(stderr, "Error: Cannot load %s\n", argv[3]);
        return(1);
    }

    real_T = FHOCR.T;
    if ( rec_T!=-1 ){ FHOCR.T = rec_T; }

    if ( use_nns+use_svm+use_mlp ){ 
        FHOCR.use_nns = use_nns; 
        FHOCR.use_svm = use_svm; 
        FHOCR.use_mlp = use_mlp;
    }

    if ( voting==-1 || FHOCR.use_nns==1 || FHOCR.use_svm==1 ){ FHOCR.voting = FHOCR_HARD_VOTING; }
    else{ FHOCR.voting = voting; }

    FHRecTestDebug = 0;
    
    for ( i=0 ; i<idic.num ; i++ ){
        FHRecTestStart();
        ret = FHOCRRecogThresh(&FHOCR, &idic.img[i], thresh);
        // for ( int j=0 ; j<FHOCRNumClass(&FHOCR) ; j++ ){
        //     if ( FHOCR.prob[j]<0.000001 ){ continue; }
        //     printf("%f ", FHOCR.prob[j]);
        // }
        // puts("");
        // ret = FHOCRRecog(&FHOCR, &idic.img[i]);
        assert(ret!=-1);
        if ( ret==-1 ){ continue; }

        /* ---- Consider the candidates by adding arguments ---- */
        if ( ret==idic.label[i] ){
            FHRecTestEnd(ret, idic.label[i]);
            printf("1 %.3f\n", FHOCREta);
            // getchar();

        }
        else{
            printf("0 %.3f\n", FHOCREta);
            FHRecTestEnd(ret, idic.label[i]);
        }

        // getchar();
    }

    FHIDicInit(&idic);

    sprintf(fname, "%s/.FH/result.txt", homepass);
    fp = fopen(fname, "a");

    fprintf(fp, "---------------------------------------------\n");
    FHOCRPrint(&FHOCR, fp);
    fprintf(fp, "Test dataset: %s\n", argv[3]);
    fprintf(fp, "Training dataset: %s\n", argv[2]);
    fprintf(fp, "num_candidate = %d\n", num_candidate);
    fprintf(fp, "Accuracy: %f %%\n", FHRecTestAccuracy());
    fprintf(fp, "Time: %f [msec]\n", FHRecTestTime());
    fprintf(fp, "Voting thresh: %f\n", thresh);
    if ( FHOCR.voting==FHOCR_HARD_VOTING ){
        fprintf(fp, "Voting method: %s\n", "HARD");
    }
    else if ( FHOCR.voting==FHOCR_SOFT_VOTING ){
        fprintf(fp, "Voting method: %s\n", "SOFT");
    }

    if ( EOF==fclose(fp) ){ return(-1); }

    FHOCR.T = real_T;

    FHOCRFinish(&FHOCR);
    return(0);
}


int Test(int argc, char **argv){
    int     i;
    int     ret;
    int     num_candidate = 1;
    int     real_T, rec_T = -1;
    int     use_nns = 0, use_svm = 0, use_mlp = 0;
    float   thresh = 0.0;
    FHOCR   FHOCR;
    FHIDic  idic;
    FILE    *fp;
    char    fname[128];
    int     voting = -1;
    char    voting_method[6];
    char    cls_name[8];
    const char *homepass = getenv("HOME");
    char    param_str[][64] = {
        "THRESH=",
        "CANDIDATE=",
        "T=",
        "VOTING=",
        "CLASSIFIER="
    };

    if ( !(4<=argc && argc<=8) ){
        fprintf(stderr, "Error: Invalid arguments\n");
        PrintUsage();
        return(1);
    }

    for ( i=4 ; i<argc ; i++ ){
        if ( NULL!=strstr(argv[i], param_str[0]) ){
            if ( 1!=sscanf(argv[i], "THRESH=%f", &thresh) ){ 
                fprintf(stderr, "Error: Invalid arguments\n");
                PrintUsage();
                return(1); 
            }
        }
        else if ( NULL!=strstr(argv[i], param_str[1]) ){
            if ( 1!=sscanf(argv[i], "CANDIDATE=%d", &num_candidate) ){ 
                fprintf(stderr, "Error: Invalid arguments\n");
                PrintUsage();
                return(1); 
            }
        }
        else if ( NULL!=strstr(argv[i], param_str[2]) ){
            if ( 1!=sscanf(argv[i], "T=%d", &rec_T) ){ 
                fprintf(stderr, "Error: Invalid arguments\n");
                PrintUsage();
                return(1); 
            }
        }
        else if ( NULL!=strstr(argv[i], param_str[3]) ){
            sscanf(argv[i], "VOTING=%s", voting_method);
            if ( strcmp(voting_method, "HARD")==0 ){
                voting = FHOCR_HARD_VOTING;
            }
            else if ( strcmp(voting_method, "SOFT")==0 ){
                voting = FHOCR_SOFT_VOTING;
            }
            else if ( strcmp(voting_method, "MAX")==0 ){
                voting = FHOCR_MAX_VOTING;
            }
            else{
                fprintf(stderr, "Error: Invalid arguments\n");
                PrintUsage();
                return(1); 
            }
        }
        else if ( NULL!=strstr(argv[i], param_str[4]) ){
            sscanf(argv[i], "CLASSIFIER=%s", cls_name);
            if ( strcmp(cls_name, "NNS")==0 ){ use_nns = 1; }
            else if ( strcmp(cls_name, "SVM")==0 ){ use_svm = 1; }
            else if ( strcmp(cls_name, "MLP")==0 ){ use_mlp = 1; }
            else{
                fprintf(stderr, "Error: Invalid arguments\n");
                PrintUsage();
                return(1); 
            }
        }
        else{
            fprintf(stderr, "Error: Invalid arguments\n");
            PrintUsage();
            return(1); 
        }
    }

    FHOCRInit(&FHOCR);
    FHIDicInit(&idic);

    if ( FHOCRLoad(&FHOCR, argv[2]) ){
        fprintf(stderr, "Error: Cannot load %s\n", argv[2]);
        return(1);
    }

    if ( FHIDicLoad(&idic, argv[3]) ){
        fprintf(stderr, "Error: Cannot load %s\n", argv[3]);
        return(1);
    }

    real_T = FHOCR.T;
    if ( rec_T!=-1 ){ FHOCR.T = rec_T; }

    if ( use_nns+use_svm+use_mlp ){ 
        FHOCR.use_nns = use_nns; 
        FHOCR.use_svm = use_svm; 
        FHOCR.use_mlp = use_mlp;
    }

    if ( voting==-1 || FHOCR.use_nns==1 || FHOCR.use_svm==1 ){ FHOCR.voting = FHOCR_HARD_VOTING; }
    else{ FHOCR.voting = voting; }
    
    for ( i=0 ; i<idic.num ; i++ ){
        FHRecTestStart();
        ret = FHOCRRecogThresh(&FHOCR, &idic.img[i], thresh);
        // ret = FHOCRMNRecog(&FHOCR, &idic.img[i], 3);
        // ret = FHOCRSNRecog(&FHOCR, &idic.img[i], 3);
        // for ( int j=0 ; j<FHOCRNumClass(&FHOCR) ; j++ ){
        //     if ( FHOCR.prob[j]<0.000001 ){ continue; }
        //     printf("%f ", FHOCR.prob[j]);
        // }
        // puts("");
        // ret = FHOCRRecog(&FHOCR, &idic.img[i]);
        assert(ret!=-1);
        if ( ret==-1 ){ continue; }

        if ( 1 ){
            char    strtmp[128];
            if ( i<300 ){
                if ( idic.label[i]==ret ){
                    sprintf(strtmp, "/home/fuma/Pictures/PanCor/%d.bmp", i);
                }
                else{
                    sprintf(strtmp, "/home/fuma/Pictures/PanInc/%d.bmp", i);
                }
                // FHImgSave(&idic.img[i], strtmp);
            }
        }

        /* ---- Consider the candidates by adding arguments ---- */
        if ( ret==idic.label[i] ){
            FHRecTestEnd(ret, idic.label[i]);
            // getchar();
            continue;

        }
        if ( num_candidate>=2 ){
            ret = FHOCR.ans[1];
            if ( ret==idic.label[i] ){
                FHRecTestEnd(ret, idic.label[i]);
                continue;
            }
        }
        if ( num_candidate==3 ){
            ret = FHOCR.ans[2];
            if ( ret==idic.label[i] ){
                FHRecTestEnd(ret, idic.label[i]);
                continue;
            }
        }
        FHRecTestEnd(ret, idic.label[i]);

        if ( ret!=idic.label[i] ){
            // getchar();
        }

        // getchar();
    }

    FHIDicInit(&idic);

    sprintf(fname, "%s/.FH/result.txt", homepass);
    fp = fopen(fname, "a");

    fprintf(fp, "---------------------------------------------\n");
    FHOCRPrint(&FHOCR, fp);
    fprintf(fp, "Test dataset: %s\n", argv[3]);
    fprintf(fp, "Training dataset: %s\n", argv[2]);
    fprintf(fp, "num_candidate = %d\n", num_candidate);
    fprintf(fp, "Accuracy: %f %%\n", FHRecTestAccuracy());
    fprintf(fp, "Time: %f [msec]\n", FHRecTestTime());
    fprintf(fp, "Voting thresh: %f\n", thresh);
    if ( FHOCR.voting==FHOCR_HARD_VOTING ){
        fprintf(fp, "Voting method: %s\n", "HARD");
    }
    else if ( FHOCR.voting==FHOCR_SOFT_VOTING ){
        fprintf(fp, "Voting method: %s\n", "SOFT");
    }

    if ( EOF==fclose(fp) ){ return(-1); }

    FHOCR.T = real_T;

    FHOCRFinish(&FHOCR);
    return(0);
}

int Testseg(int argc, char **argv)
{
    int i;
    int ret;
    int num_candidate = 1;
    int real_T, rec_T = -1;
    int use_nns = 0, use_svm = 0, use_mlp = 0;
    float thresh = 0.0;
    FHOCR FHOCR;
    FHIDic idic;
    FILE *fp;
    char fname[128];
    int voting = -1;
    char voting_method[6];
    char cls_name[8];
    const char *homepass = getenv("HOME");
    char param_str[][64] = {
        "THRESH=",
        "CANDIDATE=",
        "T=",
        "VOTING=",
        "CLASSIFIER="};

    if (!(4 <= argc && argc <= 8))
    {
        fprintf(stderr, "Error: Invalid arguments\n");
        PrintUsage();
        return (1);
    }

    for (i = 4; i < argc; i++)
    {
        if (NULL != strstr(argv[i], param_str[0]))
        {
            if (1 != sscanf(argv[i], "THRESH=%f", &thresh))
            {
                fprintf(stderr, "Error: Invalid arguments\n");
                PrintUsage();
                return (1);
            }
        }
        else if (NULL != strstr(argv[i], param_str[1]))
        {
            if (1 != sscanf(argv[i], "CANDIDATE=%d", &num_candidate))
            {
                fprintf(stderr, "Error: Invalid arguments\n");
                PrintUsage();
                return (1);
            }
        }
        else if (NULL != strstr(argv[i], param_str[2]))
        {
            if (1 != sscanf(argv[i], "T=%d", &rec_T))
            {
                fprintf(stderr, "Error: Invalid arguments\n");
                PrintUsage();
                return (1);
            }
        }
        else if (NULL != strstr(argv[i], param_str[3]))
        {
            sscanf(argv[i], "VOTING=%s", voting_method);
            if (strcmp(voting_method, "HARD") == 0)
            {
                voting = FHOCR_HARD_VOTING;
            }
            else if (strcmp(voting_method, "SOFT") == 0)
            {
                voting = FHOCR_SOFT_VOTING;
            }
            else if (strcmp(voting_method, "MAX") == 0)
            {
                voting = FHOCR_MAX_VOTING;
            }
            else
            {
                fprintf(stderr, "Error: Invalid arguments\n");
                PrintUsage();
                return (1);
            }
        }
        else if (NULL != strstr(argv[i], param_str[4]))
        {
            sscanf(argv[i], "CLASSIFIER=%s", cls_name);
            if (strcmp(cls_name, "NNS") == 0)
            {
                use_nns = 1;
            }
            else if (strcmp(cls_name, "SVM") == 0)
            {
                use_svm = 1;
            }
            else if (strcmp(cls_name, "MLP") == 0)
            {
                use_mlp = 1;
            }
            else
            {
                fprintf(stderr, "Error: Invalid arguments\n");
                PrintUsage();
                return (1);
            }
        }
        else
        {
            fprintf(stderr, "Error: Invalid arguments\n");
            PrintUsage();
            return (1);
        }
    }

    FHOCRInit(&FHOCR);
    FHIDicInit(&idic);

    if (FHOCRLoad(&FHOCR, argv[2]))
    {
        fprintf(stderr, "Error: Cannot load %s\n", argv[2]);
        return (1);
    }

    if (FHIDicLoad(&idic, argv[3]))
    {
        fprintf(stderr, "Error: Cannot load %s\n", argv[3]);
        return (1);
    }

    real_T = FHOCR.T;
    if (rec_T != -1)
    {
        FHOCR.T = rec_T;
    }

    if (use_nns + use_svm + use_mlp)
    {
        FHOCR.use_nns = use_nns;
        FHOCR.use_svm = use_svm;
        FHOCR.use_mlp = use_mlp;
    }

    if (voting == -1 || FHOCR.use_nns == 1 || FHOCR.use_svm == 1)
    {
        FHOCR.voting = FHOCR_HARD_VOTING;
    }
    else
    {
        FHOCR.voting = voting;
    }

    for (i = 0; i < idic.num; i++)
    {
        FHRecTestStart();
        ret = FHOCRRecogThresh(&FHOCR, &idic.img[i], thresh);
        // ret = FHOCRMNRecog(&FHOCR, &idic.img[i], 3);
        // ret = FHOCRSNRecog(&FHOCR, &idic.img[i], 3);
        // for ( int j=0 ; j<FHOCRNumClass(&FHOCR) ; j++ ){
        //     if ( FHOCR.prob[j]<0.000001 ){ continue; }
        //     printf("%f ", FHOCR.prob[j]);
        // }
        // puts("");
        // ret = FHOCRRecog(&FHOCR, &idic.img[i]);
        assert(ret != -1);
        if (ret == -1)
        {
            continue;
        }

        if (1)
        {
            char strtmp[128];
            if (i < 300)
            {
                if (idic.label[i] == ret)
                {
                    sprintf(strtmp, "/home/fuma/Pictures/PanCor/%d.bmp", i);
                }
                else
                {
                    sprintf(strtmp, "/home/fuma/Pictures/PanInc/%d.bmp", i);
                }
                // FHImgSave(&idic.img[i], strtmp);
            }
        }

        /* ---- Consider the candidates by adding arguments ---- */
        if (ret == idic.label[i])
        {
            FHRecTestEnd(ret, idic.label[i]);
            // getchar();
            continue;
        }
        if (num_candidate >= 2)
        {
            ret = FHOCR.ans[1];
            if (ret == idic.label[i])
            {
                FHRecTestEnd(ret, idic.label[i]);
                continue;
            }
        }
        if (num_candidate == 3)
        {
            ret = FHOCR.ans[2];
            if (ret == idic.label[i])
            {
                FHRecTestEnd(ret, idic.label[i]);
                continue;
            }
        }
        FHRecTestEnd(ret, idic.label[i]);

        if (ret != idic.label[i])
        {
            // getchar();
        }

        // getchar();
    }

    FHIDicInit(&idic);

    sprintf(fname, "%s/.FH/result.txt", homepass);
    fp = fopen(fname, "a");

    fprintf(fp, "---------------------------------------------\n");
    FHOCRPrint(&FHOCR, fp);
    fprintf(fp, "Test dataset: %s\n", argv[3]);
    fprintf(fp, "Training dataset: %s\n", argv[2]);
    fprintf(fp, "num_candidate = %d\n", num_candidate);
    fprintf(fp, "Accuracy: %f %%\n", FHRecTestAccuracy());
    fprintf(fp, "Time: %f [msec]\n", FHRecTestTime());
    fprintf(fp, "Voting thresh: %f\n", thresh);
    if (FHOCR.voting == FHOCR_HARD_VOTING)
    {
        fprintf(fp, "Voting method: %s\n", "HARD");
    }
    else if (FHOCR.voting == FHOCR_SOFT_VOTING)
    {
        fprintf(fp, "Voting method: %s\n", "SOFT");
    }

    if (EOF == fclose(fp))
    {
        return (-1);
    }

    FHOCR.T = real_T;

    FHOCRFinish(&FHOCR);
    return (0);
}

int DiffTest(int argc, char **argv){
    int     i;
    int     ret1, ret2;
    int     cnt1, cnt2;
    FHOCR   FHOCR1, FHOCR2;
    FHIDic  idic;
    FILE    *fp;
    char    fname[128];
    const char *homepass = getenv("HOME");

    FHOCRInit(&FHOCR1);
    FHOCRInit(&FHOCR2);
    FHIDicInit(&idic);

    if ( argc!=5 ){
        fprintf(stderr, "Error: Invalid arguments\n");
        PrintUsage();
        return(1);
    }

    if ( FHOCRLoad(&FHOCR1, argv[2]) ){
        fprintf(stderr, "Error: Cannot load %s\n", argv[2]);
        return(1);
    }

    if ( FHOCRLoad(&FHOCR2, argv[3]) ){
        fprintf(stderr, "Error: Cannot load %s\n", argv[3]);
        return(1);
    }

    if ( FHIDicLoad(&idic, argv[4]) ){
        fprintf(stderr, "Error: Cannot load %s\n", argv[4]);
        return(1);
    }

    cnt1 = cnt2 = 0;
    for ( i=0 ; i<idic.num ; i++ ){
        ret1 = FHOCRRecog(&FHOCR1, &idic.img[i]);

        /* For ACPR2019 */
#if     1
        ret2 = FHOCRRecogThresh(&FHOCR2, &idic.img[i], 0.5);
#else
        ret2 = FHOCRRecog(&FHOCR2, &idic.img[i]);
#endif
        if ( ret1==-1 || ret2==-1 ){
            FHOCRFinish(&FHOCR1);
            FHOCRFinish(&FHOCR2);
            FHIDicInit(&idic);
            return(1);
        }
        if ( ret1==ret2 ){ continue; }
        else if ( ret1==idic.label[i] && ret2!=idic.label[i] ){ 
            printf("No. %d OK, NG\n", i);
            cnt1++; 
        }
        else if ( ret2==idic.label[i] && ret1!=idic.label[i] ){ 
            printf("No. %d NG, OK\n", i);
            cnt2++; 
        }
    }

    FHIDicInit(&idic);

    sprintf(fname, "%s/.FH/result.txt", homepass);
    fp = fopen(fname, "a");

    fprintf(fp, "---------------------------------------------\n");
    fprintf(fp, "[diff test]\n");
    fprintf(fp, "FHOCR1:\n");
    FHOCRPrint(&FHOCR1, fp);
    fprintf(fp, "FHOCR2:\n");
    FHOCRPrint(&FHOCR2, fp);
    fprintf(fp, "Test dataset: %s\n", argv[3]);
    fprintf(fp, "OK NG: %d\n", cnt1);
    fprintf(fp, "NG OK: %d\n", cnt2);

    if ( EOF==fclose(fp) ){ return(-1); }

    fp = stdout;
    fprintf(fp, "---------------------------------------------\n");
    fprintf(fp, "[diff test]\n");
    fprintf(fp, "FHOCR1:\n");
    FHOCRPrint(&FHOCR1, fp);
    fprintf(fp, "FHOCR2:\n");
    FHOCRPrint(&FHOCR2, fp);
    fprintf(fp, "Test dataset: %s\n", argv[3]);
    fprintf(fp, "OK NG: %d\n", cnt1);
    fprintf(fp, "NG OK: %d\n", cnt2);

    FHOCRFinish(&FHOCR1);
    FHOCRFinish(&FHOCR2);
    return(0);
}


int ConvTest(int argc, char **argv){
    int     i, j;
    int     ret;
    FHOCR   FHOCR1, FHOCR2;
    FHIDic  idic;
    FHSet   class_table;
    FILE    *fp;
    char    fname[128];
    int     num_class;
    float   *hist;
    const char *homepass = getenv("HOME");

    FHOCRInit(&FHOCR1);
    FHOCRInit(&FHOCR2);
    FHIDicInit(&idic);
    FHSetInit(&class_table);

    if ( argc!=5 ){
        fprintf(stderr, "Error: Invalid arguments\n");
        PrintUsage();
        return(1);
    }

    if ( FHOCRLoad(&FHOCR1, argv[2]) ){
        fprintf(stderr, "Error: Cannot load %s\n", argv[2]);
        return(1);
    }

    if ( FHOCRLoad(&FHOCR2, argv[3]) ){
        fprintf(stderr, "Error: Cannot load %s\n", argv[3]);
        return(1);
    }

    if ( FHIDicLoad(&idic, argv[4]) ){
        fprintf(stderr, "Error: Cannot load %s\n", argv[4]);
        return(1);
    }

    FHOCR1.T /= 2;
    FHOCR2.T /= 2;

    // if ( FHOCRNumClass(&FHOCR1)!=FHOCRNumClass(&FHOCR2) ){ return(1); }
    // if ( FHOCRNumClass(&FHOCR1)>FHOCRNumClass(&FHOCR2) ){
    //     FHSetCp(&FHOCR1.class_table, &class_table);
    // }
    // else{
    //     FHSetCp(&FHOCR2.class_table, &class_table);
    // }
    // num_class = FHOCRNumClass(&FHOCR1);
    FHSetUnion(&FHOCR1.class_table, &FHOCR2.class_table, &class_table);
    num_class = class_table.num;
    printf("num_class: %d\n", num_class);

    FHOCR1.voting = FHOCR_HARD_VOTING;
    FHOCR2.voting = FHOCR_HARD_VOTING;

    FHOCR1.prob = FHMalloc(float, FHOCRNumClass(&FHOCR1));
    FHOCR2.prob = FHMalloc(float, FHOCRNumClass(&FHOCR2));

    hist = new float[num_class];
    FHSubAll(hist, num_class, 0);

    for ( i=0 ; i<idic.num ; i++ ){
        FHRecTestStart();
        FHSubAll(hist, num_class, 0);
        FHOCRVoting(&FHOCR1, &idic.img[i]);
        FHOCRVoting(&FHOCR2, &idic.img[i]);
        for ( j=0 ; j<num_class ; j++ ){
            int     label = class_table.e[j];
            int     index1 = FHSetSearch(&FHOCR1.class_table, label);
            int     index2 = FHSetSearch(&FHOCR2.class_table, label);
            if ( index1>=0 ){ hist[j] += FHOCR1.prob[index1]; }
            if ( index2>=0 ){ hist[j] += FHOCR2.prob[index2]; }
        }
        ret = FHArgmaxF(hist, num_class);
        ret = class_table.e[ret];
        FHRecTestEnd(ret, idic.label[i]);
    }

    FHIDicFinish(&idic);

    sprintf(fname, "%s/.FH/result.txt", homepass);
    fp = fopen(fname, "a");

    fprintf(fp, "---------------------------------------------\n");
    fprintf(fp, "[conv test]\n");
    fprintf(fp, "FHOCR1:\n");
    FHOCRPrint(&FHOCR1, fp);
    fprintf(fp, "FHOCR2:\n");
    FHOCRPrint(&FHOCR2, fp);
    fprintf(fp, "Test dataset: %s\n", argv[3]);
    fprintf(fp, "Accuracy: %f\n", FHRecTestAccuracy());

    if ( EOF==fclose(fp) ){ return(-1); }

    fp = stdout;
    FHOCR1.T *= 2;
    FHOCR2.T *= 2;

    FHOCRFinish(&FHOCR1);
    FHOCRFinish(&FHOCR2);
    FHSetFinish(&class_table);
    delete []hist;
    return(0);
}


int EachTest(int argc, char **argv){
    int     i, j, n;
    int     ret;
    int     nns_ans, svm_ans, mlp_ans;
    int     msr = 0;
    int     use16 = 0, use32 = 0, use64 = 0;
    int     cnt = 0;
    int     true_T = 0, rec_T = 0;
    float   accuracy = 0;
    FHOCR   FHOCR;
    FHIDic  idic;
    int img_size[3] = {16, 32, 64};

    FHOCRInit(&FHOCR);
    FHIDicInit(&idic);

    if ( argc>5 ){
        fprintf(stderr, "Error: Invalid arguments\n");
        PrintUsage();
        return(1);
    }

    for ( i=4 ; i<argc ; i++ ){
        if ( NULL!=strstr("T=", argv[i]) ){
            if ( 1!=sscanf(argv[i], "T=%d", &rec_T) ){
                fprintf(stderr, "Error: Invalid arguments\n");
                PrintUsage();
                return(1);
            }
        }
        else{
            fprintf(stderr, "Error: Invalid arguments\n");
            PrintUsage();
            return(1);
        }
    }

    if ( FHOCRLoad(&FHOCR, argv[2]) ){
        fprintf(stderr, "Error: Cannot load %s\n", argv[2]);
        return(1);
    }

    if ( rec_T==0 ){ rec_T = FHOCR.T; }

    true_T = FHOCR.T;
    FHOCR.T = rec_T;

    if ( FHIDicLoad(&idic, argv[3]) ){
        fprintf(stderr, "Error: Cannot load %s\n", argv[3]);
        return(1);
    }

    for ( i=0 ; ; i++ ){
        if ( FHOCR.RIFeat[i]==FHOCR_MSR_16 || \
                FHOCR.RIFeat[i]==FHOCR_MSR_16_32 || \
                FHOCR.RIFeat[i]==FHOCR_MSR_16_64 || \
                FHOCR.RIFeat[i]==FHOCR_MSR_16_32_64 ){ use16 = 1; }
        if ( FHOCR.RIFeat[i]==FHOCR_MSR_32 || \
                FHOCR.RIFeat[i]==FHOCR_MSR_16_32 || \
                FHOCR.RIFeat[i]==FHOCR_MSR_32_64 || \
                FHOCR.RIFeat[i]==FHOCR_MSR_16_32_64 ){ use32 = 1; }
        if ( FHOCR.RIFeat[i]==FHOCR_MSR_64 || \
                FHOCR.RIFeat[i]==FHOCR_MSR_16_64 || \
                FHOCR.RIFeat[i]==FHOCR_MSR_32_64 || \
                FHOCR.RIFeat[i]==FHOCR_MSR_16_32_64 ){ use64 = 1; }
        if ( FHOCR.RIFeat[i]==-1 ){ break; }
    }

    msr = use16 + use32 + use64;

    assert(msr!=0);

    FHRecTestDebug = 0;
    for ( n=0 ; n<3 && FHOCR.use_nns ; n++ ){
        if ( !(n==0 && use16) ){ continue; }
        if ( !(n==1 && use32) ){ continue; }
        if ( !(n==2 && use64) ){ continue; }
        cnt = 0;
        accuracy = 0;
        printf("NNS(%d)\n", img_size[n]);
        for ( i=n ; i<FHOCR.T ; i+=msr ){
            FHRecTestReset();
            for ( j=0 ; j<idic.num ; j++ ){
                FHRecTestStart();
                FHOCRRecogOne(&FHOCR, &idic.img[j], i, &nns_ans, \
                        &svm_ans, &mlp_ans);
                ret = FHOCR.class_table.e[nns_ans];
                FHRecTestEnd(ret, idic.label[j]);
            }
            printf("%3d %3.2f\n", cnt+1, FHRecTestAccuracy());
            accuracy += (float)FHRecTestAccuracy();
            cnt++;
        }

        accuracy /= ((float)FHOCR.T/msr);
        printf("Average Accuracy: %f %%\n", accuracy);
    }

    for ( n=0 ; n<3 && FHOCR.use_svm ; n++ ){
        if ( !(n==0 && use16) ){ continue; }
        if ( !(n==1 && use32) ){ continue; }
        if ( !(n==2 && use64) ){ continue; }
        cnt = 0;
        accuracy = 0;
        printf("SVM(%d)\n", img_size[n]);
        for ( i=n ; i<FHOCR.T ; i+=msr ){
            FHRecTestReset();
            for ( j=0 ; j<idic.num ; j++ ){
                FHRecTestStart();
                FHOCRRecogOne(&FHOCR, &idic.img[j], i, &nns_ans, \
                        &svm_ans, &mlp_ans);
                ret = FHOCR.class_table.e[svm_ans];
                FHRecTestEnd(ret, idic.label[j]);
            }
            printf("%3d %3.2f\n", cnt+1, FHRecTestAccuracy());
            accuracy += (float)FHRecTestAccuracy();
            cnt++;
        }

        accuracy /= ((float)FHOCR.T/msr);
        printf("Average Accuracy: %f %%\n", accuracy);
    }

    for ( n=0 ; n<msr && FHOCR.use_mlp ; n++ ){
        if ( !(n==0 && use16) ){ continue; }
        if ( !(n==1 && use32) ){ continue; }
        if ( !(n==2 && use64) ){ continue; }
        cnt = 0;
        accuracy = 0;
        printf("MLP(%d)\n", img_size[n]);
        for ( i=n ; i<FHOCR.T ; i+=msr ){
            FHRecTestReset();
            for ( j=0 ; j<idic.num ; j++ ){
                FHRecTestStart();
                FHOCRRecogOne(&FHOCR, &idic.img[j], i, &nns_ans, \
                        &svm_ans, &mlp_ans);
                ret = FHOCR.class_table.e[mlp_ans];
                FHRecTestEnd(ret, idic.label[j]);
            }
            printf("%3d %3.2f\n", cnt+1, FHRecTestAccuracy());
            accuracy += (float)FHRecTestAccuracy();
            cnt++;
        }

        accuracy /= ((float)FHOCR.T/msr);
        printf("Average Accuracy: %f %%\n", accuracy);
    }

    FHOCR.T = true_T;

    FHIDicFinish(&idic);
    FHOCRFinish(&FHOCR);

    return(0);
}


int TTest(int argc, char **argv){
    int     i, j;
    int     ret;
    int     T;
    FHOCR   FHOCR;
    FHIDic  idic;
    int     **ans;

    FHOCRInit(&FHOCR);
    FHIDicInit(&idic);

    if ( argc>5 ){
        fprintf(stderr, "Error: Invalid arguments\n");
        PrintUsage();
        return(1);
    }

    if ( FHOCRLoad(&FHOCR, argv[2]) ){
        fprintf(stderr, "Error: Cannot load %s\n", argv[2]);
        return(1);
    }

    if ( FHIDicLoad(&idic, argv[3]) ){
        fprintf(stderr, "Error: Cannot load %s\n", argv[3]);
        return(1);
    }

    T = FHOCR.T;

    ans = new int*[idic.num];

    for ( i=0 ; i<idic.num ; i++ ){
        ans[i] = new int[T*3];
#ifdef  _OPENMP
#pragma omp parallel for
#endif
        for ( j=0 ; j<T ; j++ ){
            FHOCRRecogOne(&FHOCR, &idic.img[i], j, &ans[i][j], &ans[i][T+j], \
                    &ans[i][T*2+j]);
        }
        fprintf(stderr, "No. %d/%d\n", i+1, idic.num);
    }

    FHOCRPrint(&FHOCR, stdout);

    printf("NNS:\n");
    if ( FHOCR.use_nns ){
        FHRecTestDebug = 0;
        for ( FHOCR.T=1 ; FHOCR.T<=T ; FHOCR.T++ ){
            FHRecTestReset();
            for ( i=0 ; i<idic.num ; i++ ){
                FHRecTestStart();
                ret = FHPlurVoting(ans[i], FHOCR.T);
                ret = FHOCR.class_table.e[ret];
                if ( ret==-1 ){
                    FHOCRFinish(&FHOCR);
                    FHIDicFinish(&idic);
                    return(1);
                }
                FHRecTestEnd(ret, idic.label[i]);
            }
            printf("%d %f\n", FHOCR.T, FHRecTestAccuracy());
        }
    }
    printf("SVM:\n");
    if ( FHOCR.use_svm ){
        FHRecTestDebug = 0;
        for ( FHOCR.T=1 ; FHOCR.T<=T ; FHOCR.T++ ){
            FHRecTestReset();
            for ( i=0 ; i<idic.num ; i++ ){
                FHRecTestStart();
                ret = FHPlurVoting(ans[i]+T, FHOCR.T);
                ret = FHOCR.class_table.e[ret];
                if ( ret==-1 ){
                    FHOCRFinish(&FHOCR);
                    FHIDicFinish(&idic);
                    return(1);
                }
                FHRecTestEnd(ret, idic.label[i]);
            }
            printf("%d %f\n", FHOCR.T, FHRecTestAccuracy());
        }
    }
    printf("MLP:\n");
    if ( FHOCR.use_mlp ){
        FHRecTestDebug = 0;
        for ( FHOCR.T=1 ; FHOCR.T<=T ; FHOCR.T++ ){
            FHRecTestReset();
            for ( i=0 ; i<idic.num ; i++ ){
                FHRecTestStart();
                ret = FHPlurVoting(ans[i]+T*2, FHOCR.T);
                ret = FHOCR.class_table.e[ret];
                if ( ret==-1 ){
                    FHOCRFinish(&FHOCR);
                    FHIDicFinish(&idic);
                    return(1);
                }
                FHRecTestEnd(ret, idic.label[i]);
            }
            printf("%d %f\n", FHOCR.T, FHRecTestAccuracy());
        }
    }

    FHOCR.T = T;

    for ( i=0 ; i<idic.num ; i++ ){ delete []ans[i]; }
    delete []ans;
    FHOCRFinish(&FHOCR);
    FHIDicFinish(&idic);
    return(0);
}


static float CalcTheta(float e, float mean_e, int T){
    return((e*T-mean_e)/(mean_e*(T-1)));
}


int AnalyEnsemble(int argc, char **argv){
    int     i;
    int     ret;
    float   mean_var = 0;
    float   prob = 0;
    float   prob_sum = 0;
    float   e, mean_e;
    float   accuracy_inf;
    float   theta;
    FHOCR   FHOCR;
    FHIDic  idic;
    const char *homepass = getenv("HOME");
    char    fname[64];
    FILE    *fp;

    FHOCRInit(&FHOCR);
    FHIDicInit(&idic);

    if ( argc!=4 && argc!=5 ){
        fprintf(stderr, "Error: Invalid arguments\n");
        PrintUsage();
        return(1);
    }

    if ( FHOCRLoad(&FHOCR, argv[2]) ){
        fprintf(stderr, "Error: Cannot load %s\n", argv[2]);
        return(1);
    }

    if ( FHIDicLoad(&idic, argv[3]) ){
        fprintf(stderr, "Error: Cannot load %s\n", argv[3]);
        return(1);
    }

    // if ( argc==5 ){
    //     num_candidate = atoi(argv[4]);
    // }

    for ( i=0 ; i<idic.num ; i++ ){
        FHRecTestStart();
        ret = FHOCRRecog(&FHOCR, &idic.img[i]);
        prob = FHOCR.prob[0];
        prob_sum += prob;
        if ( ret==-1 ){
            FHOCRFinish(&FHOCR);
            FHIDicInit(&idic);
            return(1);
        }

        mean_var += FHOCRVar;
        FHRecTestEnd(ret, idic.label[i]);
    }

    /* ---- Analysis of the ensemble ---- */

    e = (100 - FHRecTestAccuracy()) / 100;
    mean_e = 1.0 - prob_sum/idic.num;
    theta = CalcTheta(e, mean_e, FHOCR.T);
    accuracy_inf = theta * mean_e;

    sprintf(fname, "%s/.FH/result.txt", homepass);
    fp = fopen(fname, "a");

    fprintf(fp, "---------------------------------------------\n");
    fprintf(fp, "PARAMETERS:\n");
    FHOCRPrint(&FHOCR, fp);
    fprintf(fp, "Test dataset: %s\n", argv[3]);
    fprintf(fp, "Training dataset: %s\n", argv[2]);
    fprintf(fp, "RESULTS:\n");
    fprintf(fp, "\tT = %d\n", FHOCR.T);
    fprintf(fp, "\tK = %d\n", FHOCR.K);
    fprintf(fp, "\tEnsemble accuracy: %f %%\n", FHRecTestAccuracy());
    fprintf(fp, "\tEnsemble error: %f\n", e);
    fprintf(fp, "\tMean accuracy of classifiers: %f %%\n", 100*prob_sum/idic.num);
    fprintf(fp, "\tMean error of classifiers: %f\n", mean_e);
    fprintf(fp, "\tCorrelation: %f\n", theta);
    // printf("\tMean Variance: %f\n", mean_var/idic.num);
    fprintf(fp, "\tAccuracy when T is infinity: %f\n", 100-100*accuracy_inf);
    fprintf(fp, "\tTime: %f\n", FHRecTestTime());

    fclose(fp);

    FHOCRFinish(&FHOCR);
    FHIDicInit(&idic);
    return(0);
}


int CpFHOCR(int argc, char **argv){
    FHOCR   fhocr;

    FHOCRInit(&fhocr);
    if ( argc!=4 ){ return(1); }

    if ( FHOCRLoad(&fhocr, argv[2]) ){ return(1); }
    if ( FHOCRSave(&fhocr, argv[3]) ){ return(1); }

    FHOCRFinish(&fhocr);
    return(0);
}


int FHOCRInfo(int argc, char **argv){
    FHOCR   FHOCR;

    FHOCRInit(&FHOCR);
    if ( argc!=3 ){ FHOCRFinish(&FHOCR); return(1); }

    if ( FHOCRLoad(&FHOCR, argv[2]) ){ FHOCRFinish(&FHOCR); return(1); }
    FHOCRPrint(&FHOCR, stdout);
    FHOCRFinish(&FHOCR);
    return(0);
}


int ETLTest(int argc, char **argv){
    int     i;
    int     ret;
    int     num_candidate = 1;
    int     real_T, rec_T = -1;
    int     use_nns = 0, use_svm = 0, use_mlp = 0;
    float   thresh = 0.0;
    FHOCR   FHOCR;
    FHIDic  idic;
    FILE    *fp;
    char    fname[128];
    int     voting = -1;
    char    voting_method[6];
    char    cls_name[8];
    const char *homepass = getenv("HOME");
    char    param_str[][64] = {
        "THRESH=",
        "CANDIDATE=",
        "T=",
        "VOTING=",
        "CLASSIFIER="
    };

    if ( !(4<=argc && argc<=8) ){
        fprintf(stderr, "Error: Invalid arguments\n");
        PrintUsage();
        return(1);
    }

    for ( i=4 ; i<argc ; i++ ){
        if ( NULL!=strstr(argv[i], param_str[0]) ){
            if ( 1!=sscanf(argv[i], "THRESH=%f", &thresh) ){ 
                fprintf(stderr, "Error: Invalid arguments\n");
                PrintUsage();
                return(1); 
            }
        }
        else if ( NULL!=strstr(argv[i], param_str[1]) ){
            if ( 1!=sscanf(argv[i], "CANDIDATE=%d", &num_candidate) ){ 
                fprintf(stderr, "Error: Invalid arguments\n");
                PrintUsage();
                return(1); 
            }
        }
        else if ( NULL!=strstr(argv[i], param_str[2]) ){
            if ( 1!=sscanf(argv[i], "T=%d", &rec_T) ){ 
                fprintf(stderr, "Error: Invalid arguments\n");
                PrintUsage();
                return(1); 
            }
        }
        else if ( NULL!=strstr(argv[i], param_str[3]) ){
            sscanf(argv[i], "VOTING=%s", voting_method);
            if ( strcmp(voting_method, "HARD")==0 ){
                voting = FHOCR_HARD_VOTING;
            }
            else if ( strcmp(voting_method, "SOFT")==0 ){
                voting = FHOCR_SOFT_VOTING;
            }
            else if ( strcmp(voting_method, "MAX")==0 ){
                voting = FHOCR_MAX_VOTING;
            }
            else{
                fprintf(stderr, "Error: Invalid arguments\n");
                PrintUsage();
                return(1); 
            }
        }
        else if ( NULL!=strstr(argv[i], param_str[4]) ){
            sscanf(argv[i], "CLASSIFIER=%s", cls_name);
            if ( strcmp(cls_name, "NNS")==0 ){ use_nns = 1; }
            else if ( strcmp(cls_name, "SVM")==0 ){ use_svm = 1; }
            else if ( strcmp(cls_name, "MLP")==0 ){ use_mlp = 1; }
            else{
                fprintf(stderr, "Error: Invalid arguments\n");
                PrintUsage();
                return(1); 
            }
        }
        else{
            fprintf(stderr, "Error: Invalid arguments\n");
            PrintUsage();
            return(1); 
        }
    }

    FHOCRInit(&FHOCR);
    FHIDicInit(&idic);

    if ( FHOCRLoad(&FHOCR, argv[2]) ){
        fprintf(stderr, "Error: Cannot load %s\n", argv[2]);
        return(1);
    }

    if ( ReadETL9B(&idic, argv[3]) ){
        fprintf(stderr, "Error: Cannot load %s\n", argv[3]);
        return(1);
    }

    real_T = FHOCR.T;
    if ( rec_T!=-1 ){ FHOCR.T = rec_T; }

    if ( use_nns+use_svm+use_mlp ){ 
        FHOCR.use_nns = use_nns; 
        FHOCR.use_svm = use_svm; 
        FHOCR.use_mlp = use_mlp;
    }

    if ( voting==-1 || FHOCR.use_nns==1 || FHOCR.use_svm==1 ){ FHOCR.voting = FHOCR_HARD_VOTING; }
    else{ FHOCR.voting = voting; }
    
    for ( i=0 ; i<idic.num ; i++ ){
        FHRecTestStart();
        ret = FHOCRRecogThresh(&FHOCR, &idic.img[i], thresh);
        // ret = FHOCRMNRecog(&FHOCR, &idic.img[i], 3);
        // for ( int j=0 ; j<FHOCRNumClass(&FHOCR) ; j++ ){
        //     if ( FHOCR.prob[j]<0.000001 ){ continue; }
        //     printf("%f ", FHOCR.prob[j]);
        // }
        // puts("");
        // ret = FHOCRRecog(&FHOCR, &idic.img[i]);
        assert(ret!=-1);
        if ( ret==-1 ){ continue; }

        /* ---- Consider the candidates by adding arguments ---- */
        if ( ret==idic.label[i] ){
            FHRecTestEnd(ret, idic.label[i]);
            // getchar();
            continue;

        }
        if ( num_candidate>=2 ){
            ret = FHOCR.ans[1];
            if ( ret==idic.label[i] ){
                FHRecTestEnd(ret, idic.label[i]);
                continue;
            }
        }
        if ( num_candidate==3 ){
            ret = FHOCR.ans[2];
            if ( ret==idic.label[i] ){
                FHRecTestEnd(ret, idic.label[i]);
                continue;
            }
        }
        FHRecTestEnd(ret, idic.label[i]);
        // getchar();
    }

    FHIDicInit(&idic);

    sprintf(fname, "%s/.FH/result.txt", homepass);
    fp = fopen(fname, "a");

    fprintf(fp, "---------------------------------------------\n");
    FHOCRPrint(&FHOCR, fp);
    fprintf(fp, "Test dataset: %s\n", argv[3]);
    fprintf(fp, "Training dataset: %s\n", argv[2]);
    fprintf(fp, "num_candidate = %d\n", num_candidate);
    fprintf(fp, "Accuracy: %f %%\n", FHRecTestAccuracy());
    fprintf(fp, "Time: %f [msec]\n", FHRecTestTime());
    fprintf(fp, "Voting thresh: %f\n", thresh);
    if ( FHOCR.voting==FHOCR_HARD_VOTING ){
        fprintf(fp, "Voting method: %s\n", "HARD");
    }
    else if ( FHOCR.voting==FHOCR_SOFT_VOTING ){
        fprintf(fp, "Voting method: %s\n", "SOFT");
    }

    if ( EOF==fclose(fp) ){ return(-1); }

    FHOCR.T = real_T;

    FHOCRFinish(&FHOCR);
    return(0);
    return(0);
}


