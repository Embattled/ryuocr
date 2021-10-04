from copy import Error, deepcopy, error
from builtins import Exception, dict

import os.path
import random
from collections import Counter
from urllib.request import DataHandler
import PIL

import pandas as pd
import numpy
from PIL import Image
from torchvision.transforms.transforms import Lambda

import sscd
import example


class RyuDataSet():
    def __init__(self):
        pass


class EvalDataSet(RyuDataSet):
    def __init__(self, dir, name=None, revise=False):
        super().__init__()

        self.name = name

        data = pd.read_csv(dir, sep=',',
                           index_col=None, header=None, names=["path", "label"])
        data["path"] = os.path.dirname(dir)+'/'+data["path"]

        self.dataPath = list(data["path"].values)
        self.dataLabel = numpy.array(data["label"])

        if revise:
            sscd.dict.reviseJapanDict(self.dataLabel)

        # initialize some property
        self.last_res = None
        self.best_res = None
        self.best_acc = 0.0

    def readDataToMem(self):
        self.pilData = []
        for path in self.dataPath:
            _img = Image.open(path)
            _img.load()
            self.pilData.append(_img)

    def evaluate(self, predict):
        """
        Input array of predict label, return accuracy and save result in instant.
        """
        self.last_res = predict
        compair = predict == self.dataLabel
        cnt = Counter(compair)

        acc=cnt[True]/sum(cnt.values())
        if acc>self.best_acc:
            self.best_acc=acc
            self.best_res=predict
        return acc

    def getErrorCasesImageGrid(self, num=0, format="T:{} F:{}", predict=None, size=(128, 128),last=False):
        if predict == None:
            if self.last_res is None:
                raise ValueError("There has been no evaluation yet.")
            elif last==True:
                predict=self.last_res
            else:
                predict=self.best_res
        
        errorSet = []
        errorLabel = []
        for i in range(len(self.dataLabel)):
            if self.dataLabel[i] != predict[i]:
                _img = Image.open(self.dataPath[i])
                _img.load()
                errorSet.append(_img)

                t = self.dataLabel[i]
                f = predict[i]
                errorLabel.append(format.format(t, f))

        if len(errorSet) == 0:
            print("There are no error samples.")
        if num == 0:
            num = len(errorSet)

        return example.getExampleImageLabeledGridPIL(
            errorSet, errorLabel, 8, num//8, size=size)

    def getAllCasesImageGrid(self, num=0, format="T:{} F:{}", predict=None, size=(128, 128),last=False):
        if predict == None:
            if self.last_res is None:
                raise ValueError("There has been no evaluation yet.")
            elif last==True:
                predict=self.last_res
            else:
                predict=self.best_res

        allset = []
        label = []
        for i in range(len(self.dataLabel)):
            _img = Image.open(self.dataPath[i])
            _img.load()
            allset.append(_img)

            if self.dataLabel[i] != predict[i]:

                t = self.dataLabel[i]
                f = predict[i]
                label.append(format.format(t, f))
            else:
                label.append(self.dataLabel[i])

        if num == 0:
            num = len(allset)
        return example.getExampleImageLabeledGridPIL(
            allset, label, 8, num//8, size=size)


class SSCDGenerator():
    def __init__(self, ttfpaths,  fontsize, dictpara, transform, padding=None):

        self.num_cls, self.num_char_dict, self.char_num_dict = sscd.dict.readDict(
            **dictpara)

        self.ttfpaths = ttfpaths
        self.fontDataSize = fontsize
        self.fontDataPadding = padding

        # get font data
        self.fontData, self.fontLabelNum = sscd.font.multiFontCharImageDictget(
            self.ttfpaths, self.char_num_dict, size=self.fontDataSize, padding=self.fontDataPadding)

        # initialize some property
        self.transform_function = lambda images: None

        self.setTransformFunc(sscd.transform.getTransformFunc(transform))

    def setTransformFunc(self, trans_func):
        self.transform_function = trans_func

    def getFontDataBunch(self, bunch=1):
        data = []
        label = []

        for _ in range(bunch):
            data.extend(deepcopy(self.fontData))
            label.extend(self.fontLabelNum)
        return data, label

    def getFontDataSample(self, num: int,shuffle):
        data = []
        label = []
        if shuffle==True:
            smp = random.choices(range(len(self.fontData)), k=num)
        else:
            smp=[]
            while len(smp)<num:
                k = min(len(self.fontData),num-len(smp))
                smp.extend(range(len(self.fontData))[0:k])
        for i in smp:
            data.append(self.fontData[i].copy())
            label.append(self.fontLabelNum[i])
        return data, label

    def getTransformedDataBunch(self, bunch=1):
        data, label = self.getFontDataBunch(bunch)
        self.transform_function(data)
        return data, label

    def getTransformedDataSample(self, num: int,shuffle=True):
        data, label = self.getFontDataSample(num,shuffle)
        self.transform_function(data)
        return data, label

    def getLabelChar2NumDict(self):
        return self.char_num_dict

    def getLabelNum2CharDict(self):
        return self.num_char_dict

    def getTrainData(self, type: str, num: int):
        if type == "bunch":
            trainData, trainLabel = self.getTransformedDataBunch(
                bunch=num)
        elif type == "sample":
            trainData, trainLabel = self.getTransformedDataSample(
                num)
        else:
            raise ValueError("Invalid sscd type: {}".format(type))
        return trainData, trainLabel


def getSSCD(profile: dict):

    sscdSet = SSCDGenerator(**profile)
    return sscdSet


def getRealSet(profile: dict):
    return EvalDataSet(**profile)


def getDataset(setParmer: dict):

    settype = setParmer.get("type")
    if settype == "real":
        return getRealSet(setParmer["profile"])
    elif settype == "sscd":
        return getSSCD(setParmer["profile"])
    else:
        raise Exception("Illegal dataset type.")
