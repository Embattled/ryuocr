import os.path

from ryusci import model as sciModel
from ryutorch import model as torchModel
from ryuutils import ryuyaml, ryudataset, ryutime, sscd


def getModel(path: str, name: str, **kwargs):
    modelDir = os.path.join(os.path.abspath(path), name)

    # torch model
    if os.path.exists(os.path.join(modelDir, "model.pt")):
        model = torchModel.TorchModel()
        model.load(modelDir, **kwargs)
    return model


def evaluate(model, testSet, topk=1):
    res = model.inference(
        testSet.dataPath, num_label=False, is_path=True)
    acc = testSet.evaluate(res, topk)

    return acc


def run(config_path: str, model: str = "", **kwargs):
    config = ryuyaml.loadyaml(config_path)

    # ---------- global parameter -----
    paramGlobal = config["Global"]

    # Model save path
    saveModelDir = paramGlobal["save_result_dir"]
    modelName = paramGlobal["pretrained_model"] if model == "" else model

    model = getModel(saveModelDir, modelName, **kwargs)

    # ----------- Test Parameter-----------
    paramTest = config["Test"]
    testSet = ryudataset.getDataset(paramTest["dataset"])
    print("{}/{} data from {} use for evaluation.".format(len(testSet.dataLabel),
                                                          len(testSet.dataLabelOld), testSet.name))
    res = model.inference(
        testSet.dataPath, num_label=False, is_path=True)

    for k in range(1, 11):
        acck, acc1 = testSet.evaluate(res, k)
        print("Top-{} accuracy: {}".format(k, acck))

    revise=paramTest.setdefault("revise", False)
    if revise == True:
        sscd.dict.reviseJapanDict(testSet.dataLabel)
        acc1,_=testSet.evaluate(res,1,revise)
        print("Top-1 accuracy with revised Japanese dictionary: {}".format(acc1))