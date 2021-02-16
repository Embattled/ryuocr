import torch
import numpy
from torchvision import transforms
# ------- Transmit font image ----------
def recovery(trainData,originData):
    
    for i in range(len(trainData)):
        cb = torch.zeros(3,dtype=torch.int)
        cc = torch.tensor([255,255,255],dtype=torch.int)

        orinp = originData[i].numpy().transpose(1, 2, 0)
        back = (orinp == numpy.array([0, 0, 0])).all(axis=2)
        imnp = trainData[i].numpy().transpose(1, 2, 0)

        imnp[back] = cb
        imnp[numpy.logical_not(back)] = cc
    pass
# Define Color Change
def changeColor(trainData):

    for i in range(len(trainData)):
        cb = torch.randint(256, [3])
        cc = torch.randint(256, [3])

        while(abs(cc[0]-cb[0]) < 50 or abs(cc[1]-cb[1]) < 50 or abs(cc[2]-cb[2]) < 50):
            cb = torch.randint(256, [3])
            cc = torch.randint(256, [3])

        
        imnp = trainData[i].numpy().transpose(1, 2, 0)
        back = (imnp == numpy.array([0, 0, 0])).all(axis=2)

        imnp[back] = cb
        imnp[numpy.logical_not(back)] = cc
    pass

def affine(trainData, anglerange=0, shearrange=0, scalerange=0):

    for i in range(len(trainData)):
        r = torch.rand(10,dtype=torch.float32).tolist()

        angle = (-anglerange)+(2*anglerange*r[0])
        xshear = (-shearrange)+(2*shearrange*r[1])
        yshear = (-shearrange)+(2*shearrange*r[2])
        scale = (1-scalerange)+(2*scalerange*r[3])
        trainData[i] = transforms.functional.affine(
            trainData[i], angle=angle, shear=(xshear, yshear), scale=scale,translate=(0,0))
    # return image
    pass
