import numpy
import PIL

from math import floor, sin, cos, tan, pi
from PIL import ImageOps, Image, ImageMorph

rng = numpy.random.default_rng()
# PILSet to numpy array (Copy)


def pilSet2Numpy(images):
    npimages = []
    if len(images) == 0:
        raise ValueError
    for img in images:
        npimages.append(numpy.asarray(img))
    npimages = numpy.asarray(npimages)
    return npimages


def numpySet2PilSet(npimages):
    images = []
    npimages = numpy.squeeze(npimages)
    for i in range(len(npimages)):
        images.append(PIL.Image.fromarray(npimages[i]))
    return images


# ------------------- Transform ------------- function
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


def affine(image, rotation=0, scale: tuple = (1, 1), shear: tuple = (0, 0), translation: tuple = (0, 0), resample=PIL.Image.BILINEAR, size=None, center=True):
    """
    rotation : degree of counter clockwise.
    sclae : (2,2) means output image will be scale to 0.5 times,
            (0.5,0.5) means output image will be enlarge to 2 times.
    shear : (shearx,sheary) degree of shear
    translation : (x,y) move image to left direction x pixel, up direction y pixel.

    """
    osize = image.size
    cx = 0
    cy = 0
    if center == True:
        cx = floor((osize[1]-1)/2)
        cy = floor((osize[0]-1)/2)

    if size == None:
        size = osize
    tx = floor((size[1]-1)/2)
    ty = floor((size[0]-1)/2)

    sinr = sin(rotation*pi/180)
    cosr = cos(rotation*pi/180)

    shx = sin(shear[0]*pi/180)
    shy = sin(shear[1]*pi/180)

    # move center of origin image to (0,0)
    affine_mat = numpy.array([[1, 0, cx], [0, 1, cy], [
        0, 0, 1]], dtype=numpy.float32)
    # scale
    affine_mat = numpy.dot(affine_mat, numpy.array(
        [[scale[0], 0, 0], [0, scale[1], 0], [0, 0, 1]]))
    # shear
    affine_mat = numpy.dot(affine_mat, numpy.array(
        [[1, shx, 0], [shy, 1, 0], [0, 0, 1]]))

    # roation
    affine_mat = numpy.dot(affine_mat, numpy.array(
        [[cosr, -sinr, 0], [sinr, cosr, 0], [0, 0, 1]]))

    # # move back
    affine_mat = numpy.dot(affine_mat, numpy.array([[1, 0, -tx+translation[0]], [0, 1, -ty+translation[1]], [
        0, 0, 1]], dtype=numpy.float32))
    return image.transform(size, PIL.Image.AFFINE, affine_mat.flatten().tolist()[:6], resample=resample)

# p means the probability of data will be transform.


def affineDirect(image, data: tuple, resample=PIL.Image.BILINEAR, size=None, center=True):
    """
    Affine with matrix data directly, input data=(a1,a2,a3,a4), will apple affine transform with:
        | a1, a2, 0 |
        | a3, a4, 0 |
        |  0,  0, 1 |
    """
    if len(data) < 4:
        raise ValueError(
            "Input data of affineDirect must have 4 value (4-list-like), but receive %d values." % (len(data)))
    osize = image.size
    if size == None:
        size = osize
    cx = 0
    cy = 0
    tx = 0
    ty = 0
    if center == True:
        cx = floor((osize[1]-1)/2)
        cy = floor((osize[0]-1)/2)
        tx = floor((size[1]-1)/2)
        ty = floor((size[0]-1)/2)
    # move center of origin image to (0,0)
    affine_mat = numpy.array([[1, 0, cx], [0, 1, cy], [
        0, 0, 1]], dtype=numpy.float32)
    # apply matrix
    affine_mat = numpy.dot(affine_mat, numpy.array(
        [[data[0], data[1], 0], [data[2], data[3], 0], [0, 0, 1]]))
    # # move back
    affine_mat = numpy.dot(affine_mat, numpy.array([[1, 0, -tx], [0, 1, -ty], [
        0, 0, 1]], dtype=numpy.float32))
    return image.transform(size, PIL.Image.AFFINE, affine_mat.flatten().tolist()[:6], resample=resample)


def uniformAffine(images, rotation=0, shear=(0, 0), scale=(1, 1), p=0.5, inplace=True):
    """
    rotation : int , rotate -x ~ x degree randomly
    shear : (x,y) , shear horizontal -x~x degree, vertical -y~y degree.
    scale : (low,high) scale range for both scalex and scaley.
    """
    n = len(images)
    p_trans = rng.uniform(low=0, high=1, size=n)

    target_rotation = numpy.zeros(n)
    if rotation != 0:
        target_rotation = rng.uniform(
            low=-rotation, high=rotation, size=n)

    target_shear = numpy.zeros((n, 2))
    if shear != (0, 0):
        target_shear = numpy.hstack((rng.uniform(low=-shear[0], high=shear[0], size=(
            n, 1)), rng.uniform(low=-shear[1], high=shear[1], size=(n, 1))))

    target_scale = numpy.ones((n, 2))
    if scale != (1, 1):
        target_scale = rng.uniform(low=scale[0], high=scale[1], size=(n, 2))

    new_images = []
    if inplace:
        for i in range(n):
            if p_trans[i] >= p:
                continue
            images[i] = affine(images[i], target_rotation[i],
                               shear=target_shear[i], scale=target_scale[i])
        new_images = images
    else:
        for i in range(n):
            if p_trans[i] >= p:
                new_images.append(images[i])
                continue
            new_images.append(affine(
                images[i], target_rotation[i], shear=target_shear[i]), scale=target_scale[i])
    return new_images


def uniformAffineDirect(images, range14=(1, 1), range23=(0, 0), p=0.5, inplace=True):
    """
    range14 : range of random value a1,a4
    range23 : range of random value a2,a3
    """
    n = len(images)
    p_trans = rng.uniform(low=0, high=1, size=n)

    target_14 = numpy.ones((n, 2))
    if range14 != (1, 1):
        target_14 = rng.uniform(
            low=range14[0], high=range14[1], size=(n, 2))

    target_23 = numpy.zeros((n, 2))
    if range23 != (0, 0):
        target_23 = rng.uniform(low=range23[0], high=range23[1], size=(n, 2))

    datas = numpy.hstack((target_14[:, 0:1], target_23, target_14[:, 1:2]))
    new_images = []
    if inplace:
        for i in range(n):
            if p_trans[i] >= p:
                continue
            images[i] = affineDirect(images[i], data=datas[i])
        new_images = images
    else:
        for i in range(n):
            if p_trans[i] >= p:
                new_images.append(images[i])
                continue
            new_images.append(affineDirect(images[i], data=datas[i]))
    return new_images


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


# p means the probability of data will be transform.
def uniformPerspective(images, scale=(-0.1, 0.2), p=0.5, inplace=True):
    n = len(images)
    target_ratio = rng.uniform(low=scale[0], high=scale[1], size=(n, 8))
    p_trans = rng.uniform(low=0, high=1, size=n)
    new_images = []
    if inplace:
        for i in range(n):
            if p_trans[i] >= p:
                continue
            images[i] = perspective(images[i], target_ratio[i])
        new_images = images
    else:
        for i in range(n):
            if p_trans[i] >= p:
                new_images.append(images[i])
                continue
            new_images.append(perspective(images[i], target_ratio[i]))
    return new_images


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
    "erosion4l":["N:(.1. .1. .0.)->0"],
    "erosion8l":["4:(.1. .1. .0.)->0","4:(1.. .1. ..0)->0"]
}


def morphology(image, **kwargs):
    morph = ImageMorph.MorphOp(**kwargs)
    _, img = morph.apply(image)
    return img


def randomMorph(images, inplace=True, p=0.3):
    n = len(images)
    patterns = ['dilation4', 'dilation8', 'erosion4l']
    target_pattern = rng.integers(low=0, high=len(patterns), size=(n))
    p_trans = rng.uniform(low=0, high=1, size=n)

    new_images = []
    if inplace:
        for i in range(n):
            if p_trans[i] >= p:
                continue
            pt=known_patterns[patterns[target_pattern[i]]]
            # pt=known_patterns[patterns[2]]
            images[i] = morphology(
                images[i], patterns=pt)
        new_images = images
    else:
        for i in range(n):
            if p_trans[i] >= p:
                new_images.append(image)
                continue
            image = morphology(images[i], op_name=patterns[target_pattern[i]])
            new_images.append(image)
    return new_images


def sscdCreate(originData, originLabel, transfunction, epoch=1):

    sscd = []
    labels = []
    for _ in range(epoch):
        sscd.extend(transfunction(originData))
        labels.extend(originLabel)
    return sscd, labels


def _getTransform(para: dict):
    name = para.setdefault("name", "None")
    if name == "perspective":
        scale = para.setdefault("scale", [0, 0, 1])
        p = para.setdefault("p", 0.5)

        def transform(images):
            uniformPerspective(images, scale=scale, p=p)
            return images
    elif name == "affine":
        rotation = para.setdefault("rotation", 0)
        scale = para.setdefault("scale", (1, 1))
        shear = para.setdefault("shear", (0, 0))
        para.setdefault
        p = para.setdefault("p", 0.5)

        def transform(images):
            uniformAffine(images, rotation=rotation,
                          shear=shear, scale=scale, p=p)
            return images
    elif name == "color":
        gap = para.setdefault("gap", 50)

        def transform(images):
            randomColorizeSet(images, gap=gap)
            return images
    elif name == "morph":
        def transform(images):
            p = para.setdefault("p", 0.5)
            randomMorph(images, p=p)
            return images

    else:
        print("Unsupport transform function: "+"name")

        def transform(images):
            return
    return transform


def getTransformFunc(para: list):
    trans = []
    if para != None:
        for i in range(len(para)):
            trans.append(_getTransform(para[i]))

    def transfunc(images):
        for t in trans:
            t(images)
        return

    return transfunc
