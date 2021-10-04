from collections import Counter
import pathlib
import sys

import numpy
import pandas as pd
from torch.utils.data.dataloader import T


from ryuutils import *
import ryutrain
import ryutest

# ------------program global variable -----------
timeMemo = ryutime.TimeMemo()
log = dict()


def run(config_path: str, loop=1,pretrained=False):
    config = ryuyaml.loadyaml(config_path)
    ryuyaml.printyaml(config)
    timestamp=timeMemo.getNowTimeStr()

    # ---------- global parameter -----
    paramGlobal = config["Global"]
    # Model save path
    saveModelDir = paramGlobal["save_model_dir"]
    # Accuracy History Graph Save Path
    saveResultDir = paramGlobal["save_result_dir"]

    # ----------- Test Dataset-----------
    paramTest = config["Test"]
    testSet = ryudataset.getDataset(paramTest["dataset"])
    # testSet.readDataToMem()

    # ------------ Train Dataset------------
    paramTrain = config["Train"]
    sscdSet = ryudataset.getDataset(paramTrain["dataset"])


    # get model
    if pretrained:
        model=ryutest.getModel(path=saveModelDir,name=paramGlobal["pretrained_model"])
    else:
        model = ryutrain.getModel(paramTrain, num_classes=sscdSet.num_cls)
        model.setSSCD(sscdSet)
        model.setValidSet(testSet)

    highest_acc = 0
    # ---------------- One model -------------------
    for lp in range(loop):
            
        # Fit Model
        timeMemo.reset()
        model.train()
        print("Train cost: "+timeMemo.getTimeCostStr())


        # res = model.inference(
        #     trainData, num_label=True, is_path=False)
        # compair = res == trainLabel
        # cnt = Counter(compair)
        # corr = cnt[True]/sum(cnt.values())
        # print("Correct predicted on Traindata: {}".format(corr))


        timeMemo.reset()
        # --------- Deep Model
        if paramTrain["series"]=="torch":

            res = model.inference(
                testSet.dataPath, num_label=False, is_path=True)
            acc=testSet.evaluate(res)
            print("Finally correct predicted: {}".format(acc))
            print("Predict one set cost time: "+timeMemo.getTimeCostStr())
            print(model.highestValidAccStr)

            error = testSet.getErrorCasesImageGrid(last=True)
            error.save("error/%s_lasterror_%02.3f.png" % (timestamp,acc))
            
            error = testSet.getErrorCasesImageGrid(last=False)
            error.save("error/%s_besterror_%02.3f.png" % (timestamp,model.highestValidAcc))
        # ---------- Sci Model
        elif paramTrain["series"]=="scikit":
            res,res_s = model.inference( 
                testSet.dataPath, num_label=False, is_path=True,pure_data=True)
            res_p=numpy.zeros((len(model.feature),*res.shape))
            acc_his=[[],[],[]]

            res=res.argmax(axis=1)
            res=model.getCharLabel(res)
            acc=testSet.evaluate(res)
            print("Correct predicted on entire ens : {}".format(acc))

            t=len(model.cls)//len(model.feature)
            for i in range(len(res_s)):

                f_i=i//t

                res=res_s[i].argmax(axis = 1)
                res=model.getCharLabel(res)
                acc=testSet.evaluate(res)
                acc_his[f_i].append(acc)
                print("Correct predicted on feature{} cls {}: {}".format(f_i+1,i+1,acc))

                res_p[i//t]+= res_s[i]
                res=res_p[i//t].argmax(axis = 1)
                res=model.getCharLabel(res)
                acc=testSet.evaluate(res)
                print("Correct predicted on feature{} ens {}: {}".format(f_i+1,i+1,acc))

            for i,acc in enumerate(acc_his):
                print("Mean accuracy of single model train feature{} : {}".format(i+1,sum(acc)/len(acc)))


            print("Predict one set cost: "+timeMemo.getTimeCostStr())

        # if acc>highest_acc:
        #     highest_acc=acc
        #     error = testSet.getErrorCasesImageGrid()
        #     error.save("error/error_%02.3f_%d.png" % (acc,lp+1))
    ryuyaml.printyaml(config)
    print(timestamp)