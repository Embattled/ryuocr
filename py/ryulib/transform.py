import torch
import numpy
from torchvision import transforms
from PIL import ImageMorph
from PIL import ImageOps 
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

    # ch=[]

    for i in range(len(trainData)):
        cb = torch.randint(256, [3], dtype=torch.float32)
        cc = torch.randint(256, [3], dtype=torch.float32)

        # while(abs(cc[0]-cb[0]) < 50 or abs(cc[1]-cb[1]) < 50 or abs(cc[2]-cb[2]) < 50):
        while(abs(cb.mean().item()-cc.mean().item()) < 80):
            cb = torch.randint(256, [3], dtype=torch.float32)
            cc = torch.randint(256, [3], dtype=torch.float32)

        imnp = trainData[i].numpy().transpose(1, 2, 0)
        back = (imnp == numpy.array([0, 0, 0])).all(axis=2)

        imnp[back] = cb
        imnp[numpy.logical_not(back)] = cc

        # if trainData[i].mean(dtype=torch.float32).item()<30:
        #     ch.append(trainData[i])
    pass


# Define uniform random affine transform
def uniform_affine(trainData, anglerange=0, shearrange=0, scalerange=0):

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

# Define normal random affine transform


def normal_affine(trainData, anglerange=0, shearrange=0, scalerange=0, nstd=1/3):

    r = torch.normal(0, nstd, (4, len(trainData)))
    r[r > 1] = 1
    r[r < -1] = -1

    r[0] = r[0]*anglerange
    r[1] = r[1]*shearrange
    r[2] = r[2]*shearrange
    r[3] = r[3]*scalerange+1
    r = r.tolist()

    for i in range(len(trainData)):
        # r = torch.rand(4, dtype=torch.float32).tolist()
        # angle = (-anglerange)+(2*anglerange*r[0])
        # xshear = (-shearrange)+(2*shearrange*r[1])
        # yshear = (-shearrange)+(2*shearrange*r[2])
        # scale = (1-scalerange)+(2*scalerange*r[3])
        trainData[i] = transforms.functional.affine(
            trainData[i], angle=r[0][i], shear=(r[1][i], r[2][i]), scale=r[3][i], translate=(0, 0))
    # return image
    pass


# Define uniform random perspective transform
def uniform_perspective(trainData, distortion_scale=0.5, p=0.5):

    s = trainData.size()
    width = s[-1]
    height = s[-2]
    half_height = height // 2
    half_width = width // 2

    startpoints = [[0, 0], [width - 1, 0],
                   [width - 1, height - 1], [0, height - 1]]

    for i in range(len(trainData)):

        r = torch.rand(size=(1,), dtype=torch.float32).item()
        if r > p:
            continue

        rw = torch.randint(
            0, int(distortion_scale*half_width), size=[4]).tolist()
        rh = torch.randint(
            0, int(distortion_scale*half_height), size=[4]).tolist()

        topleft = [rw[0], rh[0]]
        topright = [width-rw[1], rh[1]]
        botright = [width-rw[2], height-rh[2]]
        botleft = [rw[3], height-rh[3]]

        endpoints = [topleft, topright, botright, botleft]

        trainData[i] = transforms.functional.perspective(
            trainData[i], startpoints=startpoints, endpoints=endpoints)
        pass


# Define normal random perspective transform
def normal_perspective(trainData, distortion_scale=0.5, nstd=1/3):

    s = trainData.size()
    width = s[-1]
    height = s[-2]
    half_height = height // 2
    half_width = width // 2

    startpoints = [[0, 0], [width - 1, 0],
                   [width - 1, height - 1], [0, height - 1]]

    r = torch.normal(0, nstd, (len(trainData), 4, 2)).abs()
    r[r > 1] = 1
    r[:, :, 0] *= distortion_scale*half_width
    r[:, :, 1] *= distortion_scale*half_height

    r[:, 1, 0] = width-r[:, 1, 0]
    r[:, 2, 0] = width-r[:, 2, 0]
    r[:, 2, 1] = height-r[:, 2, 1]
    r[:, 3, 1] = height-r[:, 3, 1]

    for i in range(len(trainData)):
        # rw=torch.randint(0,int(distortion_scale*half_width),size=[4]).tolist()
        # rh=torch.randint(0,int(distortion_scale*half_height),size=[4]).tolist()
        # topleft=[rw[0],rh[0]]
        # topright=[width-rw[1],rh[1]]
        # botright=[width-rw[2],height-rh[2]]
        # botleft=[rw[3],height-rh[3]]
        endpoints = r[i].tolist()
        trainData[i] = transforms.functional.perspective(
            trainData[i], startpoints=startpoints, endpoints=endpoints)
        pass


# Define Morphology Transform
known_patterns = {
    "corner": ["1:(... ... ...)->0", "4:(00. 01. ...)->1"],
    "dilation4": ["4:(... .0. .1.)->1"],
    "dilation8": ["4:(... .0. .1.)->1", "4:(... .0. ..1)->1"],
    "erosion4": ["4:(... .1. .0.)->0"],
    "erosion8": ["4:(... .1. .0.)->0", "4:(... .1. ..0)->0"],
    "edge": [
        "1:(... ... ...)->0",
        "4:(.0. .1. ...)->1",
        "4:(01. .1. ...)->1",
    ],
}

def random_morph(trainData,op_name=None):
    # if op_name not in known_patterns:
    #     raise Exception("Unknown pattern " + op_name + "!")
    dil=ImageMorph.MorphOp(op_name="dilation8")
    ero=ImageMorph.MorphOp(op_name="erosion4")
    
    ops=[dil,ero]

    for i in range(len(trainData)):
        r=torch.randint(2,(1,1)).item()
        if r==1 :
            continue
        
        data=transforms.functional.rgb_to_grayscale(trainData[i].clone(),1)

        _,data=ero.apply(transforms.functional.to_pil_image(data,mode="L"))
        data=ImageOps.colorize(data,"black","white")
        trainData[i]=transforms.functional.pil_to_tensor(data)
        


