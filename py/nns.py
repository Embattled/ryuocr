from collections import Counter
from os import terminal_size
from ryuocr.py import ryusci
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
from ryuutils import sscd

# ------------program global mode -----------
timeMemo = ryutime.TimeMemo()
nowTimeStr = timeMemo.nowTimeStr()

# Set print goal
# ----------- training parameter ---------
configPath = "/home/eugene/workspace/ryuocr/py/config/newconfig.yml"
config = ryuyaml.loadyaml(configPath)

# ---------- global parameter -----
paramGlobal = config["Global"]

# Model save path
saveModelPath = paramGlobal["save_model_dir"] + nowTimeStr+'.pt'
# Accuracy History Graph Save Path
accGraphSavePath = paramGlobal["save_result_dir"] + nowTimeStr+"_acc.png"
logPath = paramGlobal["save_result_dir"] + nowTimeStr+".yml"


# ------------ Dataset -------------
# Train dataset parameter
paramTrain = config["Train"]

# Font data parameter
ttfpath = paramTrain["dataset"]["ttfpath"]
dictpath = paramTrain["dataset"]["dict"]
fontsize = paramTrain["dataset"]["fontsize"]

# Read Font Data
char_list, char_dict = sscd.dict.readDict(dictpath)
num_cls = len(char_list)
config["Train"]["dataset"]["dict_len"] = num_cls

fontData, fontLabel = sscd.font.multi_ttfdictget(
    ttfpath, char_dict, size=fontsize)

hog = ryusci.feature.getHogFun((64, 64), orientations=5,
                pixels_per_cell=(8, 8), cells_per_block=(8, 8))

# Test dataset
timeMemo.reset()
paramTest = config["Test"]

testsetName = paramTest["dataset"]["name"]
testsetPath = paramTest["dataset"]["dir"]

testDataPath, testLabelStr = ryudataset.getDataset(testsetPath)
testLabel = numpy.array([char_dict[strLabel] for strLabel in testLabelStr])

testFe = []
for i in range(len(testDataPath)):
    _image = skio.imread(testDataPath[i])
    _feature = hog(_image)
    testFe.append(_feature)
testFe = numpy.array(testFe)
print("Get JPSC1400 feature cost: "+timeMemo.getTimeCostStr())

test_sscd = False
if test_sscd:
    timeMemo.reset()
    transData = sscd.transform.uniformPerspective(
        fontData, scale=(-0.1, 0.2), p=1, inplace=False)
    # transData=sscd.transform.uniformAffine(fontData,rotation=15,shear=(20,20),p=1)
    # transData=sscd.transform.uniformAffineDirect(fontData,range14=(0.9,1.1),range23=(-0.1,0.1),p=1)
    transData = sscd.transform.randomColorizeSet(transData, inplace=True)
    # ryuutils.example.showExamplePIL(transData,8,4,shuffle=True)
    example.showExamplePIL(transData, 8)

    transData = sscd.transform.pilSet2Numpy(transData)
    print(type(transData))
    print(transData.shape)
    print(transData.dtype)
    print("Create one set cost: "+timeMemo.getTimeCostStr())
    sys.exit(0)


oneModel = False
ensemble = True
# ---------------- One model -------------------
if oneModel:
    # Create sscd
    timeMemo.reset()
    transData = sscd.transform.uniformPerspective(
        fontData, scale=0.5, p=0.7, inplace=False)
    # transData=sscd.transform.randomColorizeSet(transData,inplace=True)
    # ryuutils.example.showExamplePIL(transData,8,4,shuffle=True)

    transData = sscd.transform.pilSet2Numpy(transData)
    print(type(transData))
    print(transData.shape)
    print(transData.dtype)
    print("Create one set cost: "+timeMemo.getTimeCostStr())

    # Get feature
    timeMemo.reset()
    sscdFeature = []
    for image in transData:

        _feature = hog(image)
        sscdFeature.append(_feature)
    sscdFeature = numpy.array(sscdFeature)

    print("Get one sscd feature cost: "+timeMemo.getTimeCostStr())
    print(sscdFeature.shape)
    print(sscdFeature.dtype)

    # Fit
    timeMemo.reset()
    # cls = neighbors.KNeighborsClassifier(10,weights='distance')
    cls = neighbors.KNeighborsClassifier(1, weights='distance')
    cls.fit(sscdFeature, fontLabel)
    print("Fit one set cost: "+timeMemo.getTimeCostStr())

    timeMemo.reset()
    print("Correct predicted:" + str(cls.score(testFe, testLabel)))

    res = cls.predict(testFe)
    print(res.shape)
    print((res == testLabel).shape)

    print(Counter(res == testLabel))

    print("--")
    res_p = cls.predict_proba(testFe).argmax(axis=1)
    print(res_p.shape)

    print("Predict one set cost: "+timeMemo.getTimeCostStr())
    sys.exit(0)

# ------------------- Ensemble ----------------------
if ensemble:
    T = 120
    cls = []
    p_e = numpy.zeros((1400, 3107))

    timeMemo.reset()
    for cls_i in range(T):
        print("Start train {}".format(cls_i+1))
        cls.append(neighbors.KNeighborsClassifier(1, weights='distance'))

        transData = sscd.transform.uniformPerspective(
            fontData, scale=(-0.1, 0.3), p=1, inplace=False)
        # transData=sscd.transform.uniformAffine(transData,rotation=15,shear=(5,5))
        transData = sscd.transform.uniformAffineDirect(
            transData, range14=(0.9, 1.1), range23=(-0.1, 0.1), p=1, inplace=True)
        transData = sscd.transform.randomColorizeSet(transData, inplace=True)
        # ryuutils.example.showExamplePIL(transData,8,4,shuffle=True)

        transData = sscd.transform.pilSet2Numpy(transData)

        sscdFeature = []
        for image in transData:
            _feature = hog(image)
            sscdFeature.append(_feature)
        sscdFeature = numpy.array(sscdFeature)

        cls[cls_i].fit(sscdFeature, fontLabel)
        # print("Accuracy classifier("+str(cls_i+1)+") : " + str(cls[cls_i].score(testFe, testLabel)))
        res_p = cls[cls_i].predict_proba(testFe)
        p_e += res_p

        res_p = res_p.argmax(axis=1)
        cnt = Counter(testLabel == res_p)
        print("Classifier {}'s accuracy : {}".format(
            cls_i+1, cnt[True]/sum(cnt.values())))

        res_p_e = p_e.argmax(axis=1)
        cnt = Counter(testLabel == res_p_e)
        print("Ensembled classifier {}'s accuracy : {}".format(
            cls_i+1, cnt[True]/sum(cnt.values())))
        print("Now time cost {}".format(timeMemo.getTimeCostStr()))
    sys.exit(0)
