
import ryusci
import ryutorch
from ryuutils import ryuyaml,ryudataset


def getModel():
    pass



def evaluate():
    pass

def run(config_path:str):
    config = ryuyaml.loadyaml(config_path)

    # ---------- global parameter -----
    paramGlobal = config["Global"]

    # Model save path
    saveModelDir = paramGlobal["save_model_dir"]

    # Accuracy History Graph Save Path
    saveResultDir = paramGlobal["save_result_dir"]

    paramTest = config["Test"]
    testSet = ryudataset.getDataset(paramTest["dataset"])