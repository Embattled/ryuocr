
from ryuutils import *


def run(config_path, size, sscdshuffle, withlabel, sscdmargin, sscdrow, sscdcol, sscdpath, **kwargs):
    timeMemo = ryutime.TimeMemo()
    config = ryuyaml.loadyaml(config_path)

    # ------------ Train Dataset------------
    paramTrain = config["Train"]
    sscdSet = ryudataset.getDataset(paramTrain["dataset"])

    # Create sscd
    timeMemo.reset()
    trainData, trainLabel = sscdSet.getTransformedDataSample(sscdrow*sscdcol,sscdshuffle)
    print("Get example set cost: "+timeMemo.getTimeCostStr())

    trainLabelChar = [sscdSet.num_char_dict[i] for i in trainLabel]
    if withlabel == True:
        grid = example.getExampleImageLabeledGridPIL(
            trainData, trainLabelChar, sscdcol, sscdrow, size=(size, size), margin=sscdmargin)
    else:
        grid = example.getExampleImageGridPIL(
            trainData, sscdcol, sscdrow, size=(size, size), margin=sscdmargin)
    grid.save(sscdpath)
