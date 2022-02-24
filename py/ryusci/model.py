
# My import
from numpy.lib.function_base import append
from sklearn import neighbors, svm
from . import feature
import numpy
from PIL import Image


class SciModel:
    def __init__(self):
        self.cls = None
        self.process = None
        self.ensemble = None
        self.num_classes = None

        # dataset
        self.sscdSet = None
        self.validSet = None
        
    # label operation
    def _getLabel(self, prob):
        return numpy.argmax(prob, axis=1)

    def getCharLabel(self, num_label):
        return numpy.array([self.num_char_dict[i] for i in num_label])

    def setCharDict(self, num_char_dict: dict):
        self.num_char_dict = num_char_dict

    # feature
    def setFeature(self, args: dict):
        self.feature = feature.getFeatureList(args)

    def setSSCD(self, sscd):
        self.sscdSet = sscd
        self.setCharDict(self.sscdSet.getLabelNum2CharDict())

    def setValidSet(self, validset):
        self.validSet = validset

    def _train(self, trainData, label, cls_i=0, f_i=0):
        trainFeature = []
        for image in trainData:
            _feature = self.feature[f_i](image)
            trainFeature.append(_feature)
        trainFeature = numpy.array(trainFeature)
        self.cls[cls_i].fit(trainFeature, label)

    def train(self):
        if self.process == None:
            raise ValueError("Emply train process")

        msr_t = len(self.cls)//len(self.feature)
        for process in self.process:
            for f_i in range(len(self.feature)):
                for cls_i in range(f_i*msr_t, (f_i+1)*msr_t):
                    print("Start train classifier {} using feature {}".format(
                        cls_i+1, f_i+1))
                    trainData, trainLabel = self.sscdSet.getTrainData(
                        **process["sscd"])
                    self._train(trainData, trainLabel, cls_i, f_i)

    def inference(self, testData, is_path: bool, num_label=False, pure_data=False):

        
        p_e, p_s = self.inference_proba(testData, is_path)

        if pure_data == True:
            return p_e, p_s

        res = p_e.argmax(axis=1)
        if not num_label:
            return self.getCharLabel(res)
        return res

    # Need implementation
    def inference_proba(self, testData, is_path: bool):
        # Get feature
        testFe = []
        for featureFunc in self.feature:
            _fes = []
            for image in testData:
                if is_path:
                    image = Image.open(image)
                    image.load()
                _feature = featureFunc(image)
                _fes.append(_feature)
            testFe.append(_fes)
        testFe = numpy.array(testFe)

        p_e = numpy.zeros((len(testData), self.num_classes))
        p_s = []

        msr_t = len(self.cls)//len(self.feature)
        for cls_i in range(len(self.cls)):
            res_p = self.cls[cls_i].predict_proba(
                testFe[cls_i // msr_t])
            p_e += res_p
            p_s.append(res_p)

        p_s = numpy.array(p_s)
        return p_e, p_s
    def save(self,savepath,**kwargs):
        pass


def getClassifier(args: dict):
    classifier = args["name"]
    l = 1
    if args["ensemble"] == True:
        l = args["ensemble_num"]

    cls = []
    for _ in range(l):
        if classifier == "knn":
            cls.append(neighbors.KNeighborsClassifier(**args["knn"]))
        elif classifier == "SVM":
            cls.append(svm.LinearSVC())
        else:
            raise ValueError("Illegal scikit algorithm name.")
    return cls


def getSciModel(args: dict, num_classes: int):

    model = SciModel()
    model.num_classes = num_classes
    model.cls = getClassifier(args["algorithm"])
    model.setFeature(args["feature"])
    model.process = args["process"]
    return model
