import torch
import numpy
from torchvision.transforms import functional as F


# Convert list of pil image to torch tensor. (Have Copy)
def pil2Tensor(pilimage: list):
    images = []
    for i in range(len(pilimage)):
        # handle PIL Image
        img = torch.as_tensor(numpy.array(pilimage[i]))
        img = img.view(pilimage[i].size[1], pilimage[i].size[0], len(pilimage[i].getbands()))

        # put it from HWC to CHW format
        img = img.permute((2, 0, 1))
        images.append(img)
    t = torch.stack(images)
    return t.clone().detach()

# None copy
def tensor2Pil(t):
    images = []
    for i in range(len(t)):
        images.append(F.to_pil_image(t[i]))
    return images


# Define uniform random affine transform
def uniform_affine(trainData, anglerange=0, shearrange=0, scalerange=0):

    for i in range(len(trainData)):
        r = torch.rand(4, dtype=torch.float32).tolist()

        angle = (-anglerange)+(2*anglerange*r[0])
        xshear = (-shearrange)+(2*shearrange*r[1])
        yshear = (-shearrange)+(2*shearrange*r[2])
        scale = (1-scalerange)+(2*scalerange*r[3])
        trainData[i] = F.affine(
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
        trainData[i] = F.affine(
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

        trainData[i] = F.perspective(
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
        trainData[i] = F.perspective(
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


def random_morph(trainData, op_name=None):
    # if op_name not in known_patterns:
    #     raise Exception("Unknown pattern " + op_name + "!")
    dil = ImageMorph.MorphOp(op_name="dilation8")
    ero = ImageMorph.MorphOp(op_name="erosion4")

    ops = [dil, ero]

    for i in range(len(trainData)):
        r = torch.randint(2, (1, 1)).item()
        if r == 1:
            continue

        data = F.rgb_to_grayscale(trainData[i].clone(), 1)

        _, data = ero.apply(F.to_pil_image(data, mode="L"))
        data = ImageOps.colorize(data, "black", "white")
        trainData[i] = F.pil_to_tensor(data)
