from ryuutils import *
import numpy as np
import sys

def run(config_path,**kwargs):
    timeMemo = ryutime.TimeMemo()
    config = ryuyaml.loadyaml(config_path)

    # ------------ Train Dataset------------
    paramTrain = config["Train"]
    sscdSet = ryudataset.getDataset(paramTrain["dataset"])

    np.set_printoptions(threshold=sys.maxsize)
    # Create sscd
    timeMemo.reset()
    fontData,_=sscdSet.getFontDataSample(1)
    
    trainData, trainLabel = sscdSet.getTransformedDataSample(1)
    print(np.array(fontData[0]))
    print("----------------------------------")
    print(np.array(trainData[0]))
    fontData[0].save("/home/eugene/workspace/ryuocr/font.png")
    trainData[0].save("/home/eugene/workspace/ryuocr/morph.png")
    # grid = example.getExampleImageGridPIL(
    #         trainData, 1, 1, size=(64, 64), margin=0)
    # grid.save("/home/eugene/workspace/ryuocr/morphgrid.png")
    
    # import numpy,sys
    # from PIL import ImageOps
    # numpy.set_printoptions(threshold=sys.maxsize)
    # print(numpy.array(grid))
    # grid=grid.convert('L')

if __name__ == "__main__":
    config_path="/home/eugene/workspace/ryuocr/config/testsscd.yml"
    run(config_path)