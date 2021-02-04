import torch
import torch.tensor
import torchvision.io as tio
import pandas as pd

import sys
import pathlib

import ryulib


data = pd.read_csv("/home/eugene/workspace/dataset/3107jp.csv", index_col=0)

font7Paths = "/home/eugene/workspace/dataset/7font"
font7Paths = ryulib.dataset.getFilesPath(font7Paths)

labeldict=dict(zip(data['utfcode'].values,data.index.values))

list7=[]
labels=[]

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

list7=imagestensor=torch.cat(list7)
labels=torch.tensor(labels)

savePath1 = '/home/eugene/workspace/ryuocr/py/tensordata/7font3107img.pt'
savePath2 = '/home/eugene/workspace/ryuocr/py/tensordata/7font3107label.pt'
torch.save(list7, savePath1)
torch.save(labels, savePath2)