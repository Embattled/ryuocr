import numpy
import PIL

from math import sin, cos, tan, pi
from PIL import ImageOps, Image

rng = numpy.random.default_rng()
# PILSet to numpy array (Copy)
def pilSet2Numpy(images):
    npimages = []
    if len(images) == 0:
        raise ValueError
    for img in images:
        npimages.append(numpy.array(img))
    return numpy.asarray(npimages)


def numpySet2PilSet(npimages):
    images = []
    npimages = numpy.squeeze(npimages)
    for i in range(len(npimages)):
        images.append(PIL.Image.fromarray(npimages[i]))
    return images

def randomColorizeSet(images, gap=50, inplace=True):
    newImages = []
    if inplace:
        for i in range(len(images)):
            c = rng.integers(256, size=(2, 3))
            while(abs(c[0].mean()-c[1].mean()) < gap):
                c = rng.integers(256, size=(2, 3))

            images[i] = ImageOps.colorize(
                images[i], black=c[0].tolist(), white=c[1].tolist())
        newImages = images
    else:
        for i in range(len(images)):
            c = rng.integers(256, size=(2, 3))
            while(abs(c[0].mean()-c[1].mean()) < gap):
                c = rng.integers(256, size=(2, 3))

            newImages.append(ImageOps.colorize(
                images[i], black=c[0].tolist(), white=c[1].tolist()))
    return newImages


# def affine(image, rotation=0, scale: tuple = (1, 1), shear: tuple = (0, 0), translation: tuple = (0, 0), size: tuple = None):
#     if size == None:
#         size = image.size
#     sinr = sin(rotation*pi/180)
#     cosr = cos(rotation*pi/180)
#     print(sinr)
#     print(cosr)

#     data = numpy.array([[1, 0, translation[0]], [0, 1, translation[1]], [
#                        0, 0, 1]], dtype=numpy.float32)

#     data = numpy.dot(data, numpy.array(
#         [[scale[0], 0, 0], [0, scale[1], 0], [0, 0, 1]]))
#     data = numpy.dot(data, numpy.array(
#         [[1, tan(shear[0]), 0], [tan(shear[1]), 1, 0], [0, 0, 1]]))
#     data = numpy.dot(data, numpy.array(
#         [[cosr, -sinr, 0], [sinr, cosr, 0], [0, 0, 1]]))

#     return image.transform(size, PIL.Image.AFFINE, data.flatten().tolist()[:6])
#

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = numpy.matrix(matrix, dtype=numpy.float)
    B = numpy.array(pb).reshape(8)

    res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
    return numpy.array(res).reshape(8)

# Target_ratio 传入8个点的缩放比例
# 左上, 右上, 右下, 左下, 先x后y
def perspective(image, target_ratio, size=None, resample=PIL.Image.BILINEAR):
    if(len(target_ratio) < 8):
        print("Wrong target ratio.")
        raise ValueError

    target_ratio = numpy.array(target_ratio)
    tr = (target_ratio)/(2*(1-target_ratio))
    w, h = image.size
    if size == None:
        size = image.size

    pa = [[0, 0], [h, 0], [h, w], [0, w]]
    pb = []
    pb.append([-h*tr[0], -w*tr[1]])
    pb.append([h+h*tr[2], -w*tr[3]])
    pb.append([h+h*tr[4], w+w*tr[5]])
    pb.append([-h*tr[6], w+w*tr[7]])

    coeffs = find_coeffs(pa, pb)
    return image.transform(image.size, PIL.Image.PERSPECTIVE, coeffs, resample=resample)



# p means the percentage data will be transform
def uniformPerspective(images, scale=0.5, p=0.5,inplace=True):
    n = len(images)
    target_ratio=rng.uniform(low=0,high=scale,size=(n,8))
    p_trans=rng.uniform(low=0,high=1,size=n)

    new_images=[]
    if inplace:
        for i in range(n):
            if p_trans[i]>=p:
                continue
            images[i]=perspective(images[i],target_ratio[i])
        new_images=images
    else:
        for i in range(n):
            if p_trans[i]>=p:
                new_images.append(images[i])    
                continue
            new_images.append(perspective(images[i],target_ratio[i]))
    return new_images

def getTransform(para:dict):

    def transform():
        pass
    return transform

def sscdCreate(originData,originLabel,transfunction,epoch=1):

    sscd=[]
    labels=[]
    for _ in range(epoch):
        sscd.extend(transfunction(originData))
        labels.extend(originLabel)
    return sscd,labels