from collections import Counter

import ryusci
from PIL import Image
import pathlib
import matplotlib.pyplot as plt


from skimage import io as skio
from sklearn import neighbors


import numpy
import pandas as pd

import sys
from inspect import getsource

from ryuutils import *
import ryutrain
# import ryutest

# ------------program global variable -----------
timeMemo = ryutime.TimeMemo()
log = dict()


def run(config_path: str, test_sscd=False, loop=1):
    config = ryuyaml.loadyaml(config_path)

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

    if test_sscd:
        # Create sscd
        timeMemo.reset()

        # trainData, trainLabel = sscdSet.getFontDataBunch()
        trainData, trainLabel = sscdSet.getTransformedDataSample(64)
        print("Get example set cost: "+timeMemo.getTimeCostStr())

        trainLabelChar = [sscdSet.num_char_dict[i] for i in trainLabel]
        grid = example.getExampleImageLabeledGridPIL(
            trainData, trainLabelChar, 8, 4, size=(128, 128), shuffle=True)
        grid.show()
        grid.save("sscd_example.png")

        sys.exit(0)

    oneModel = True
    # oneModel = False
    model = ryutrain.getModel(paramTrain, num_cls=sscdSet.num_cls)
    model.setCharDict(sscdSet.getLabelNum2CharDict())


    # trainData, trainLabel = sscdSet.getTransformedDataBunch(bunch=10)
    # trainData, trainLabel = sscdSet.getTransformedDataSample(100000)
    # ---------------- One model -------------------
    for _ in range(loop):
        if oneModel:
            # Create sscd
            timeMemo.reset()
            trainData, trainLabel = sscdSet.getTransformedDataSample(100000)
            print("Create train set cost: "+timeMemo.getTimeCostStr())
            # Fit Model
            timeMemo.reset()
            model.train(trainData, trainLabel)
            print("Fit one set cost: "+timeMemo.getTimeCostStr())

            timeMemo.reset()
            res = model.inference(
                testSet.dataPath, num_label=False, is_path=True)

            print("Correct predicted: {}".format(testSet.evaluate(res)))
            print("Predict one set cost: "+timeMemo.getTimeCostStr())

            error = testSet.getErrorCasesImageGrid()
            error.save("last_error.png")
            print("Save error prediction success.")
            # model.save(saveModelDir)
        # ------------------- Ensemble ----------------------
        else:
            T = 10

            p_e = numpy.zeros((1400, sscdSet.num_cls))

            timeMemo.reset()
            for cls_i in range(T):
                print("Start train {}".format(cls_i+1))

                trainData, trainLabel = sscdSet.getTransformedDataSample(
                    100000)
                trainData = sscd.transform.pilSet2Numpy(trainData)

                trainFeature = []
                for image in trainData:
                    _feature = hog(image)
                    trainFeature.append(_feature)
                trainFeature = numpy.array(trainFeature)

                model = ryutrain.getModel(paramTrain)
                model.setCharDict(sscdSet.getLabelNum2CharDict())
                model.train(trainFeature, trainLabel)

                res_p = model.inference_proba(testFe)
                p_e += res_p

                res_p = res_p.argmax(axis=1)
                res_p = model.getCharLabel(res_p)

                print("Classifier {}'s accuracy : {}".format(
                    cls_i+1, testSet.evaluate(res_p)))

                res_p_e = p_e.argmax(axis=1)
                res_p_e = model.getCharLabel(res_p_e)
                print("Ensembled classifier {}'s accuracy : {}".format(
                    cls_i+1, testSet.evaluate(res_p_e)))
                print("Now time cost {}".format(timeMemo.getTimeCostStr()))

            error = testSet.getErrorCasesImageGrid()
            error.save("last_error.png")
            print("Save last error success.")
