from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.tensor
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils, models

from PIL import Image
import pathlib
import matplotlib.pyplot as plt

from skimage import feature
from skimage import transform as sktr
from skimage import io as skio
import numpy
import pandas as pd

import sys
import ryulib
from ryulib import transform as rt

from inspect import getsource

import ryuutils
from ryuutils.ryutime import TimeMemo
from ryuutils import ryuyaml as yaml

timeMemo=TimeMemo()
nowTimeStr = timeMemo.nowTimeStr()
print(nowTimeStr)
# -------------

sscd_profile=dict()

# Size of sscd
sscdepoch=1
sscd_profile["sscdepoch"] = sscdepoch

# Set weather show trainset example
showExample = False
# showExample = True

if showExample:
    sscdepoch=1
else:
    # Data save path
    savePath = '/home/eugene/workspace/ryuocr/py/tensorsscd/'+str(sscdepoch) + \
        '_'+nowTimeStr+"_"
    # Save log to txt files
    outputfilename = "/home/eugene/workspace/ryuocr/py/tensorsscd/" +str(sscdepoch)+ \
        "_"+nowTimeStr+".yml"


# --------------
print("SSCD Epochs: " + str(sscdepoch))

# Origin Font Data
fontLabel = torch.load(
    "/home/eugene/workspace/ryuocr/py/tensordata/7font3107label.pt")
fontData = torch.load(
    "/home/eugene/workspace/ryuocr/py/tensordata/7font3107img.pt")
print("Loding font image success.")

# Define Algorithm
def fpreprocess(trainData, fontData):
    rt.recovery(trainData, fontData)
    rt.normal_affine(trainData, anglerange=20, scalerange=0.2, shearrange=15)
    rt.normal_perspective(trainData, distortion_scale=0.3,nstd=1/3)
    rt.changeColor(trainData)
    pass
# rt.uniform_affine(trainData, anglerange=20, scalerange=0.2, shearrange=20)
# rt.uniform_perspective(trainData, distortion_scale=0.3, p=0.7)



trainData, trainLabel = ryulib.sscd.sscdCreate(
    fontData, fontLabel, fpreprocess, sscdepoch)

sscd_profile["totalSize"]=trainData.size()
sscd_profile["Transform:"]=getsource(fpreprocess)

if showExample:
    trainsetHog = ryutorch.dataset.RyuImageset(
        trainData, trainLabel, loader=ryutorch.dataset.loader.tensor_hogvisual_loader)
    trainloader = DataLoader(trainsetHog, batch_size=64, shuffle=False)
    ryulib.example.showHogExample(
        trainloader, showScreen=True, showHogImage=False)
    sys.exit(0)
else:
    torch.save(trainData, savePath+"train.pt")
    torch.save(trainLabel, savePath+"label.pt")
    sscd_profile["createResult"]=True
    yaml.saveyaml({"sscd":sscd_profile},outputfilename)
