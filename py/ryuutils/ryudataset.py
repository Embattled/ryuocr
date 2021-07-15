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
    def __init__(self, pathOfLabelFile, name=None):
        super().__init__()

        self.name = name

        data = pd.read_csv(pathOfLabelFile, sep=',',
                           index_col=None, header=None, names=["path", "label"])
        data["path"] = os.path.dirname(pathOfLabelFile)+'/'+data["path"]

        self.dataPath = list(data["path"].values)
        self.dataLabel = numpy.array(data["label"])
        sscd.dict.reviseJapanDict(self.dataLabel)

        # initialize some property
        self.last_res = None

    def readDataToMem(self):
        self.pilData=[]
        for path in self.dataPath:
            _img=Image.open(path)
            _img.load()
            self.pilData.append(_img)

    def evaluate(self, predict):
        """
        Input array of predict label, return accuracy and save result in instant.
        """
        self.last_res = predict
        compair = predict == self.dataLabel
        cnt = Counter(compair)
        return cnt[True]/sum(cnt.values())

    def getErrorCasesImageGrid(self, num=0, format="T:{} F:{}", predict=None, size=(128, 128)):
        if self.last_res is None:
            if predict == None:
                raise ValueError("There has been no evaluation yet.")
            else:
                self.last_res = predict

        errorSet = []
        errorLabel = []
        for i in range(len(self.dataLabel)):
            if self.dataLabel[i] != self.last_res[i]:
                _img=Image.open(self.dataPath[i])
                _img.load()
                errorSet.append(_img)

                t = self.dataLabel[i]
                f = self.last_res[i]
                errorLabel.append(format.format(t, f))

        if len(errorSet) == 0:
            print("There are no error samples.")
        if num == 0:
            num = len(errorSet)

        return example.getExampleImageLabeledGridPIL(
            errorSet, errorLabel, 8, num//8, size=size)

class SSCDGenerator():
    def __init__(self, ttfpaths, dictpath, fontsize):

        self.num_cls, self.num_char_dict, self.char_num_dict = sscd.dict.readDict(
            dictpath)

        self.ttfpaths = ttfpaths
        self.fontDataSize = fontsize

        # get font data
        self.fontData, self.fontLabelNum = sscd.font.multiFontCharImageDictget(
            self.ttfpaths, self.char_num_dict, size=self.fontDataSize)

        # initialize some property
        self.transform_function = lambda images: None

    def setTransformFunc(self, trans_func):
        self.transform_function = trans_func

    def getFontDataBunch(self, bunch=1):
        data = []
        label = []

        for _ in range(bunch):
            data.extend(deepcopy(self.fontData))
            label.extend(self.fontLabelNum)
        return data, label

    def getFontDataSample(self, num: int):
        data = []
        label = []

        smp = random.choices(range(len(self.fontData)), k=num)
        for i in smp:
            data.append(self.fontData[i].copy())
            label.append(self.fontLabelNum[i])
        return data, label

    def getTransformedDataBunch(self, bunch=1):
        data, label = self.getFontDataBunch(bunch)
        self.transform_function(data)
        return data, label

    def getTransformedDataSample(self, num: int):
        data, label = self.getFontDataSample(num)
        self.transform_function(data)
        return data, label

    def getLabelChar2NumDict(self):
        return self.char_num_dict

    def getLabelNum2CharDict(self):
        return self.num_char_dict


def getSSCD(profile: dict):
    ttfpath = profile["ttfpath"]
    dictpath = profile["dict"]
    fontsize = profile["fontsize"]

    sscdSet = SSCDGenerator(ttfpath, dictpath, fontsize)
    sscdSet.setTransformFunc(
        sscd.transform.getTransformFunc(profile["transform"]))
    return sscdSet


def getRealSet(profile: dict):
    name = profile.setdefault("name", "test set")
    return EvalDataSet(profile["dir"], name=name)


def getDataset(setParmer: dict):

    settype = setParmer.get("type")
    if settype == "real":
        return getRealSet(setParmer["profile"])
    elif settype == "sscd":
        return getSSCD(setParmer["profile"])
    else:
        raise Exception("Illegal dataset type.")
