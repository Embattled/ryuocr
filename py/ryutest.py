

from ryusci import model as sciModel
from ryutorch import model as torchModel
from ryuutils import ryuyaml, ryudataset, ryutime, sscd, modelio


def getModel(path:str,name:str):
    model=modelio.loadModel(path=path,name=name)
    return model

def evaluate(model,testSet):
    res = model.inference(
                testSet.dataPath, num_label=False, is_path=True)
    return testSet.evalute(res)

def run(config_path: str):
    config = ryuyaml.loadyaml(config_path)

    # ---------- global parameter -----
    paramGlobal = config["Global"]

    # Model save path
    saveModelDir = paramGlobal["save_model_dir"]
    modelName = paramGlobal["pretrained_model"]

    # Accuracy History Graph Save Path
    saveResultDir = paramGlobal["save_result_dir"]


    model=getModel(saveModelDir,modelName)
    # ----------- Test Dataset-----------
    paramTest = config["Test"]
    testSet = ryudataset.getDataset(paramTest["dataset"])


    print("Correct predicted: {}".format(evaluate(model,testSet)))


