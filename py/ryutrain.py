

from ryusci import model as sciModel
from ryutorch import model as torchModel

from ryuutils import ryuyaml, ryudataset, ryutime, sscd


def getModel(args: dict,**kwargs):

    series = args["series"]

    if series == "scikit":
        return sciModel.getSciModel(args["scikit"])
    elif series == "torch":
        return torchModel.getTorchModel(args["torch"],**kwargs)
    elif series == "tf":
        pass
    else:
        raise ValueError("Illegal model series.")


def train(paramTrain: dict):

    sscdSet = ryudataset.getDataset(paramTrain["dataset"])
    trainData, trainLabel = sscdSet.getTransformedDataSample(100000)
    trainData = sscd.transform.pilSet2Numpy(trainData)

    model = getModel(paramTrain["algorithm"])
    model.setCharDict(sscdSet.getLabelNum2CharDict())

    model.train(trainFeature, trainLabel)

    pass


def run(config_path: str):
    timeMemo = ryutime.TimeMemo()
    config = ryuyaml.loadyaml(config_path)

    # ---------- global parameter -----
    paramGlobal = config["Global"]

    # Model save path
    saveModelDir = paramGlobal["save_model_dir"]

    # Accuracy History Graph Save Path
    saveResultDir = paramGlobal["save_result_dir"]
    paramTrain = config["Train"]
