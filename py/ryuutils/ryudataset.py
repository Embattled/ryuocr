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


# set seed
random.seed(1)


class RyuDataSet():
    def __init__(self):
        # dataset type
        self.dataInMem = None
        self.dataPil = None
        self.dataPath = None
        self.dataLabel = None

        # initialize some property
        self.lastRes = None
        self.bestRes = None
        self.bestAcc = -1.0

    def __gettop1(self, predict):
        if len(predict.shape) == 1:
            return predict
        elif len(predict.shape) == 2:
            return predict[:, 0]
        else:
            raise ValueError(
                "Illegal shape of inference result {}".format(predict.shape))

    def __readDataToMem(self):
        if self.dataInMem == True and self.dataPil != None:
            return

        if self.dataPath == None:
            raise ValueError("data path is null.")

        self.dataPil = []
        for path in self.dataPath:
            _img = Image.open(path)
            _img.load()
            self.dataPil.append(_img)
        self.dataInMem = True

    def evaluate(self, predict, topk=1,revise=False):
        """
        Input array of predict label, return accuracy and save result in instant.
        """
        self.lastRes = predict

        if len(predict.shape) == 1:
            if revise==True:
                sscd.dict.reviseJapanDict(predict)
            top1compair = self.dataLabel == predict
        elif len(predict.shape) == 2:
            predictTop1=predict[:,0]
            if revise==True:
                sscd.dict.reviseJapanDict(predictTop1)
            top1compair = predictTop1 == self.dataLabel

            if predict.shape[1] < topk:
                raise ValueError(
                    "Can't run top-{} on inference result has shape {}".format(topk, predict.shape))
            if topk > 1:
                topkcompair = [self.dataLabel[i] in predict[i][:topk]
                               for i in range(len(predict))]
        else:
            raise ValueError("Predict must be a 1 or 2 dimension array.")
        # top 1
        top1cnt = Counter(top1compair)
        top1acc = top1cnt[True]/sum(top1cnt.values())
        if top1acc > self.bestAcc:
            self.bestAcc = top1acc
            self.bestRes = predict

        if topk == 1:
            return top1acc, top1acc

        # topk
        topkcnt = Counter(topkcompair)
        topkacc = topkcnt[True]/sum(topkcnt.values())
        return topkacc, top1acc

    def getErrorCasesImageGrid(self, num=0, format="T:{}\nP:{}", predict=None, size=(128, 128), last=False):
        if predict == None:
            if self.lastRes is None:
                raise ValueError("There has been no evaluation yet.")
            elif last == True:
                predict = self.lastRes
            else:
                predict = self.bestRes

        errorSet = []
        errorLabel = []
        self.__readDataToMem()

        if len(predict.shape) == 1:
            for i in range(len(self.dataLabel)):
                if self.dataLabel[i] != predict[i]:
                    errorSet.append(self.dataPil[i])

                    t = self.dataLabel[i]
                    f = predict[i]
                    errorLabel.append(format.format(t, f))
        else:
            predict_top1 = self.__gettop1(predict)
            k = min(3, predict.shape[1])

            for i in range(len(self.dataLabel)):
                if self.dataLabel[i] != predict_top1[i]:
                    errorSet.append(self.dataPil[i])
                    t = self.dataLabel[i]

                    topk = ["["+pred+"]" if pred ==
                            t else pred for pred in predict[i][:k]]

                    f = ''.join(topk)
                    errorLabel.append(format.format(t, f))

        if len(errorSet) == 0:
            print("There are no error samples.")
            return
        if num == 0:
            num = len(errorSet)

        return example.makeImageGridLabeled(errorSet, errorLabel, 10, size=size)

    def getAllCasesImageGrid(self, num=0, format="T:{}\nP:{}", predict=None, size=(128, 128), last=False):
        if predict == None:
            if self.lastRes is None:
                raise ValueError("There has been no evaluation yet.")
            elif last == True:
                predict = self.lastRes
            else:
                predict = self.bestRes

        label = []
        self.__readDataToMem()

        if len(predict.shape) == 1:
            for i in range(len(self.dataLabel)):

                t = self.dataLabel[i]
                f = predict[i]
                label.append(format.format(t, f))
        else:
            predict_top1 = self.__gettop1(predict)
            k = min(3, predict.shape[1])

            for i in range(len(self.dataLabel)):
                if self.dataLabel[i] != predict_top1[i]:

                    t = self.dataLabel[i]

                    topk = ["["+pred+"]" if pred ==
                            t else pred for pred in predict[i][:k]]

                    f = ''.join(topk)
                    label.append(format.format(t, f))

        if num == 0:
            num = len(self.dataPil)

        return example.makeImageGridLabeled(self.dataPil, label, 10, size=size)


class EvalDataSet(RyuDataSet):
    def __init__(self, dir, name=None, dictpath=None):
        """
        dir: Index file of images
        """
        super().__init__()

        self.name = name

        data = pd.read_csv(dir, sep=',',
                           index_col=None, header=None, names=["path", "label"])
        data["path"] = os.path.dirname(dir)+'/'+data["path"]

        self.dataPath = list(data["path"].values)
        self.dataPathOld = list(data["path"].values)

        self.dataLabel = numpy.array(data["label"])
        self.dataLabelOld = numpy.array(data["label"])

        self.dataInMem = False

        # dict and revise
        if dictpath != None:
            self.num_cls, self.num_char_dict, self.char_num_dict = sscd.dict.readDict(
                dictpath)
            self.removeIllegalData(list(self.char_num_dict.keys()))

    def removeIllegalData(self, dictionary: list):

        self.dataPath.clear()
        newLabel = []
        for i in range(len(self.dataPathOld)):
            if self.dataLabelOld[i] in dictionary:
                self.dataPath.append(self.dataPathOld[i])
                newLabel.append(self.dataLabelOld[i])
        self.dataLabel = numpy.array(newLabel)


class SSCDGenerator(RyuDataSet):
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

    def getFontDataSample(self, num: int, shuffle=True):
        data = []
        label = []
        if shuffle == True:
            smp = random.choices(range(len(self.fontData)), k=num)
        else:
            smp = []
            while len(smp) < num:
                k = min(len(self.fontData), num-len(smp))
                smp.extend(range(len(self.fontData))[0:k])
        for i in smp:
            data.append(self.fontData[i].copy())
            label.append(self.fontLabelNum[i])
        return data, label

    def getTransformedDataBunch(self, bunch=1):
        data, label = self.getFontDataBunch(bunch)
        self.transform_function(data)
        return data, label

    def getTransformedDataSample(self, num: int, shuffle=True):
        data, label = self.getFontDataSample(num, shuffle)
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

    def becomeValidSet(self, num: int):
        self.dataPil, self.dataLabel = self.getFontDataSample(num, True)
        self.transform_function(self.dataPil)

        self.dataInMem = True


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
