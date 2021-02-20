import torch
import numpy
from torchvision import transforms
# ------- Transmit font image ----------


def recovery(trainData, originData):

    for i in range(len(trainData)):
        cb = torch.zeros(3, dtype=torch.int)
        cc = torch.tensor([255, 255, 255], dtype=torch.int)

        orinp = originData[i].numpy().transpose(1, 2, 0)
        back = (orinp == numpy.array([0, 0, 0])).all(axis=2)
        imnp = trainData[i].numpy().transpose(1, 2, 0)

        imnp[back] = cb
        imnp[numpy.logical_not(back)] = cc
    pass

# Define Color Set


def colorSet(trainData, cb, cc):

    for i in range(len(trainData)):
        imnp = trainData[i].numpy().transpose(1, 2, 0)
        back = (imnp == numpy.array([0, 0, 0])).all(axis=2)

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


# Define random affine transform
def affine(trainData, anglerange=0, shearrange=0, scalerange=0):

    for i in range(len(trainData)):
        r = torch.rand(4, dtype=torch.float32).tolist()

        angle = (-anglerange)+(2*anglerange*r[0])
        xshear = (-shearrange)+(2*shearrange*r[1])
        yshear = (-shearrange)+(2*shearrange*r[2])
        scale = (1-scalerange)+(2*scalerange*r[3])
        trainData[i] = transforms.functional.affine(
            trainData[i], angle=angle, shear=(xshear, yshear), scale=scale, translate=(0, 0))
    # return image
    pass


# Define random perspective transform
def perspective(trainData, distortion_scale=0.5, p=0.5):

    s = trainData.size()
    width = s[-1]
    height = s[-2]
    half_height = height // 2
    half_width = width // 2

    startpoints = [[0, 0], [width - 1, 0],
                   [width - 1, height - 1], [0, height - 1]]

    for i in range(len(trainData)):

        r=torch.rand(size=(1,),dtype=torch.float32).item()
        if r>p:
            continue

        rw=torch.randint(0,int(distortion_scale*half_width),size=[4]).tolist()
        rh=torch.randint(0,int(distortion_scale*half_height),size=[4]).tolist()

        topleft=[rw[0],rh[0]]
        topright=[width-rw[1],rh[1]]
        botright=[width-rw[2],height-rh[2]]
        botleft=[rw[3],height-rh[3]]

        endpoints = [topleft, topright, botright, botleft]

        trainData[i] = transforms.functional.perspective(
            trainData[i], startpoints=startpoints,endpoints=endpoints)
        pass
