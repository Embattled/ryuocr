from collections import Counter
from pathlib import Path

import os
import sys

import numpy
import pandas as pd
from torch.utils.data.dataloader import T


from ryuutils import *
import ryutrain
import ryutest


def run(config_path: str, loop=1, validontestset: bool = True):
    timeMemo = ryutime.TimeMemo()
    config = ryuyaml.loadyaml(config_path)

    ryuyaml.printyaml(config)

    # ---------- global parameter -----
    paramGlobal = config["Global"]
    # save path
    saveResultDir = os.path.abspath(paramGlobal["save_result_dir"])

    # ------------ Train Dataset------------
    paramTrain = config["Train"]
    sscdSet = ryudataset.getDataset(paramTrain["dataset"])

    # ----------- Test Dataset-----------
    paramTest = config["Test"]
    testSet = ryudataset.getDataset(paramTest["dataset"])

    # testSet.readDataToMem()
    print("{}/{} data from {} use for evaluation.".format(len(testSet.dataLabel),
                                                          len(testSet.dataLabelOld), testSet.name))
    # get model
    model = ryutrain.getModel(paramTrain, num_classes=sscdSet.num_cls)
    model.setSSCD(sscdSet)
    if validontestset:
        model.setValidSet(testSet)

    # ---------------- One model -------------------
    for lp in range(loop):

        # prepare timestamp
        timeMemo.reset()
        timestamp = timeMemo.getNowTimeStr()
        savePath = os.path.join(saveResultDir, timestamp+paramTrain["series"])

        # Training Model
        model.train()
        print("Train cost: "+timeMemo.getTimeCostStr())
        timeMemo.reset()

        os.makedirs(savePath)
        # --------- Deep Model
        if paramTrain["series"] == "torch":

            acck, acc1 = ryutest.evaluate(model, testSet, topk=2)
            print("Predict one set cost time: "+timeMemo.getTimeCostStr())

            print("Finally top1 accuracy: {}".format(acc1))
            print("Finally top2 accuracy: {}".format(acck))

            print(model.highestValidAccStr)

            error = testSet.getErrorCasesImageGrid(last=True)
            error.save(os.path.join(savePath, "lasterror_%02.3f.png" % acc1))

            error = testSet.getErrorCasesImageGrid(last=False)
            error.save(os.path.join(savePath, "besterror_%02.3f.png" %
                                    (model.highestValidAcc)))

        # ---------- Sci Model
        elif paramTrain["series"] == "scikit":
            res, res_s = model.inference(
                testSet.dataPath, num_label=False, is_path=True, pure_data=True)
            res_p = numpy.zeros((len(model.feature), *res.shape))
            acc_his = [[], [], []]

            res = res.argmax(axis=1)
            res = model.getCharLabel(res)
            acc1, acck = testSet.evaluate(res)

            # open log file
            f = open(os.path.join(savePath, "accuracy.txt"), "w")
            print("Correct predicted on entire ens : {}".format(acc1))

            f.write("Correct predicted on entire ens : {}\n".format(acc1))

            t = len(model.cls)//len(model.feature)
            for i in range(len(res_s)):

                f_i = i//t

                # single
                res = res_s[i].argmax(axis=1)
                res = model.getCharLabel(res)
                acc1, acck = testSet.evaluate(res)

                acc_his[f_i].append(acc1)
                f.write("Correct predicted on feature{} cls {}: {}\n".format(
                    f_i+1, i+1, acc1))

                # ensemble
                res_p[f_i] += res_s[i]
                res = res_p[f_i].argmax(axis=1)
                res = model.getCharLabel(res)
                acc1, acck = testSet.evaluate(res)
                f.write("Correct predicted on feature{} ens {}: {}\n".format(
                    f_i+1, i+1, acc1))

            for i, acc_list in enumerate(acc_his):
                f.write("Mean accuracy of single model train feature{} : {}\n".format(
                    i+1, sum(acc_list)/len(acc_list)))

            f.write("Predict one set cost: "+timeMemo.getTimeCostStr()+"\n")

            f.close()
        # save model

        ryuyaml.saveyaml(config, os.path.join(savePath, "config.yml"))
        model.save(savePath, log=True)

    ryuyaml.printyaml(config)
    print(timestamp)


if __name__ == "__main__":
    pass
