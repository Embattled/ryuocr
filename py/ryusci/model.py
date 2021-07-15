
# My import
from sklearn import neighbors, svm
from . import feature
import numpy
from PIL import Image


class SciModel:
    def __init__(self):
        self.cls = None

    # label operation
    def _getLabel(self, prob):
        return numpy.argmax(prob, axis=1)

    def getCharLabel(self, num_label):
        return numpy.array([self.num_char_dict[i] for i in num_label])

    def setCharDict(self, num_char_dict: dict):
        self.num_char_dict = num_char_dict

    # feature
    def setFeature(self, args: dict):
        self.feature = feature.getFeature(args)

    def train(self, trainData, label):
        # Get feature
        trainFeature = []
        for image in trainData:
            _feature = self.feature(image)
            trainFeature.append(_feature)
        trainFeature = numpy.array(trainFeature)
        self.cls.fit(trainFeature, label)

    def inference(self, testData, is_path:bool, num_label=False):
        # Get feature
        testFe = []
        for image in testData:
            if is_path:
                image=Image.open(image)
                image.load()
            _feature = self.feature(image)
            testFe.append(_feature)

        testFe = numpy.array(testFe)
        res = self.cls.predict(testFe)

        if not num_label:
            return self.getCharLabel(res)
        return res

    # Need implementation
    def inference_proba(self):
        pass


class MyKNN(SciModel):
    def __init__(self, k, weights="distance"):
        super().__init__()

        self.cls = neighbors.KNeighborsClassifier(k, weights=weights)

    def inference_proba(self, feature):
        res = self.cls.predict_proba(feature)
        return res


class MySVM(SciModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cls = svm.LinearSVC()


def getClassifier(args: dict):
    classifier = args["name"]

    if classifier == "knn":
        model = MyKNN(**args["knn"])
    elif classifier == "SVM":
        model = MySVM()
    else:
        raise ValueError("Illegal scikit algorithm name.")
    return model


def getSciModel(args: dict):

    model = getClassifier(args["algorithm"])
    model.setFeature(args["feature"])
    return model
