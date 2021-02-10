import torch
import torch.tensor
import torchvision.io as tio
import torchvision.transforms as ttran
import pandas as pd

import sys
import pathlib

import ryulib





sys.exit(0)
# --------------------------------------------------------------------------
# create JPSC1400 tensor data
# Label Dict
labelData = pd.read_csv(
    "/home/eugene/workspace/dataset/3107jp.csv", index_col=0)
labelDict = dict(zip(labelData['utfcode'].values, labelData.index.values))
num_class = len(labelDict)


# JPSC1400 path
JPSC1400path="/home/eugene/workspace/dataset/JPSC1400-20201218/png"
JPSC1400paths=ryulib.dataset.getFilesPath(JPSC1400path)

# JPSC old label
oldLabelPath="/home/eugene/workspace/dataset/JPSC1400-20201218/label.txt"
oldName=["fileName","percode","char"]
oldLabel=pd.read_csv(oldLabelPath,names=oldName,sep=' ')

# File name to character
oldLabelDict=dict(zip(oldLabel['fileName'].values,oldLabel['char'].values))

# To tensor
jpscimgs=[]
jpsclabels=[]
for path in JPSC1400paths:
    fileName=pathlib.Path(path).stem
    c=oldLabelDict[int(fileName)]
    l=labelDict[str(hex(ord(c))).upper()[2:]]
    img=ttran.functional.resize( tio.read_image(str(path)),(64,64))

    jpscimgs.append(img)
    jpsclabels.append(l)
    # print("read file: "+fileName+"  char:"+c)

jpscimgs=torch.stack(jpscimgs)
jpsclabels=torch.tensor(jpsclabels)
savePath1 = '/home/eugene/workspace/ryuocr/py/tensordata/jpsc1400tensor64.pt'
savePath2 = '/home/eugene/workspace/ryuocr/py/tensordata/jpsc1400label3107.pt'

print(jpscimgs.size())
print(jpsclabels.size())

torch.save(jpscimgs, savePath1)
torch.save(jpsclabels, savePath2)

sys.exit(0)
# ---------------------------------------------------------------------------------
# Create 3107  7 font tensor data
# 3107 7font label
data = pd.read_csv("/home/eugene/workspace/dataset/3107jp.csv", index_col=0)

# fontPath
font7Paths = "/home/eugene/workspace/dataset/7font"
font7Paths = ryulib.dataset.getFilesPath(font7Paths)

# code to label dict
labeldict=dict(zip(data['utfcode'].values,data.index.values))

list7=[]
labels=[]

# write font to image files
for fontPath in font7Paths:
    imagePaths = ryulib.dataset.getFilesPath(fontPath)
    listf=[]
    for imagePath in imagePaths:
        fontcode=pathlib.Path(imagePath).stem
        print("add font : "+str(imagePath))
        listf.append(tio.read_image(str(imagePath)))
        labels.append(labeldict[fontcode])

        pass
    
    list7.append(torch.stack(listf))

list7=torch.cat(list7)
labels=torch.tensor(labels)

savePath1 = '/home/eugene/workspace/ryuocr/py/tensordata/7font3107img.pt'
savePath2 = '/home/eugene/workspace/ryuocr/py/tensordata/7font3107label.pt'
torch.save(list7, savePath1)
torch.save(labels, savePath2)