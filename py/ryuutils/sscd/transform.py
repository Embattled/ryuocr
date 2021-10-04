import numpy
import PIL

from math import floor, sin, cos, tan, pi
from PIL import ImageOps, Image, ImageMorph, ImageFilter
from skimage import util as skiutil


rng = numpy.random.default_rng(1)

numpy.random.get_state()

# --------------------
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
# p means the probability of data will be transform.


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

    sinr = sin(rotation*pi/180)
    cosr = cos(rotation*pi/180)

    shx = tan(shear[0]*pi/180)
    shy = tan(shear[1]*pi/180)

    # move center of origin image to (0,0)
    affine_mat = numpy.array([[1, 0, -cx], [0, 1, -cy], [
        0, 0, 1]], dtype=numpy.float32)

    # scale
    affine_mat = numpy.dot(numpy.array(
        [[scale[0], 0, 0], [0, scale[1], 0], [0, 0, 1]]), affine_mat)

    # shear
    affine_mat = numpy.dot(numpy.array(
        [[1, shx, 0], [shy, 1, 0], [0, 0, 1]]), affine_mat)

    # roation
    affine_mat = numpy.dot(numpy.array(
        [[cosr, -sinr, 0], [sinr, cosr, 0], [0, 0, 1]]), affine_mat)

    # # move back
    affine_mat = numpy.dot(numpy.array([[1, 0, cx+translation[0]], [0, 1, cy+translation[1]], [
        0, 0, 1]], dtype=numpy.float32), affine_mat)

    return image.transform(size, PIL.Image.AFFINE, affine_mat.flatten().tolist()[:6], resample=resample)


def affine_autoscale(image, rotation=0, shear: tuple = (0, 0), translation: tuple = (0, 0), resample=PIL.Image.BILINEAR, size=None, center=True):
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

    sinr = sin(rotation*pi/180)
    cosr = cos(rotation*pi/180)

    shx = tan(shear[0]*pi/180)
    shy = tan(shear[1]*pi/180)

    # move center of origin image to (0,0)
    affine_mat = numpy.array([[1, 0, -cx], [0, 1, -cy], [
        0, 0, 1]], dtype=numpy.float32)

    # shear
    affine_mat = numpy.dot(numpy.array(
        [[1, shx, 0], [shy, 1, 0], [0, 0, 1]]), affine_mat)

    # roation
    affine_mat = numpy.dot(numpy.array(
        [[cosr, -sinr, 0], [sinr, cosr, 0], [0, 0, 1]]), affine_mat)

    # scale
    p0 = (affine_mat[0][2], affine_mat[1][2])
    p1 = (affine_mat[0][0]+affine_mat[0][2], affine_mat[1, 0]+affine_mat[1][2])
    p2 = (affine_mat[0][1]+affine_mat[0][2], affine_mat[1, 1]+affine_mat[1][2])
    p3 = (numpy.sum(affine_mat[0]), numpy.sum(affine_mat[1]))

    scalex = max(abs(p3[0]-p0[0]), abs(p2[0]-p1[0]))
    scaley = max(abs(p3[1]-p0[1]), abs(p2[1]-p1[1]))

    affine_mat = numpy.dot(numpy.array(
        [[scalex, 0, 0], [0, scaley, 0], [0, 0, 1]]), affine_mat)

    # affine_mat[0][0] = affine_mat[0][0]*scalex
    # affine_mat[1][1] = affine_mat[1][1]*scaley

    # # move back
    affine_mat = numpy.dot(numpy.array([[1, 0, cx+translation[0]], [0, 1, cy+translation[1]], [
        0, 0, 1]], dtype=numpy.float32), affine_mat)

    return image.transform(size, PIL.Image.AFFINE, affine_mat.flatten().tolist()[:6], resample=resample)


def uniformAffine(images, rotation=0, shear=(0, 0), scale=(1, 1), p=0.5, autoscale=True, inplace=True):
    """
    rotation : int , rotate -x ~ x degree randomly
    shear : (x,y) , shear horizontal -x~x degree, vertical -y~y degree.
    scale : (low,high) scale range for both scalex and scaley.
    """
    n = len(images)
    p_trans = rng.uniform(low=0, high=1, size=n)

    target_rotation = numpy.zeros(n)
    if isinstance(rotation, int):
        target_rotation = rng.uniform(
            low=-rotation, high=rotation, size=n)
    elif isinstance(rotation, list) and len(rotation) == 2:
        target_rotation = rng.uniform(
            low=rotation[0], high=rotation[1], size=n)
    else:
        raise ValueError("rotation of affine must int or list have 2 values")

    target_shear = numpy.zeros((n, 2))
    if isinstance(shear, list):
        if len(shear) == 2:
            target_shear = numpy.hstack((rng.uniform(low=-shear[0], high=shear[0], size=(
                n, 1)), rng.uniform(low=-shear[1], high=shear[1], size=(n, 1))))
        elif len(shear) == 4:
            target_shear = numpy.hstack((rng.uniform(low=shear[0], high=shear[1], size=(
                n, 1)), rng.uniform(low=shear[2], high=shear[3], size=(n, 1))))
        else:
            raise ValueError(
                "shear of affine must is a list have 2 or 4 values")

    target_scale = numpy.ones((n, 2))
    if scale != (1, 1):
        target_scale = rng.uniform(low=scale[0], high=scale[1], size=(n, 2))

    new_images = []
    if inplace:
        for i in range(n):
            if p_trans[i] >= p:
                continue
            if autoscale:
                images[i] = affine_autoscale(images[i], target_rotation[i],
                                             shear=target_shear[i])
            else:
                images[i] = affine(images[i], target_rotation[i],
                                   shear=target_shear[i], scale=target_scale[i])
        new_images = images
    else:
        for i in range(n):
            if p_trans[i] >= p:
                new_images.append(images[i])
                continue
            if autoscale:
                new_images.append(affine_autoscale(images[i], target_rotation[i],
                                                   shear=target_shear[i]))
            else:
                new_images.append(affine(
                    images[i], target_rotation[i], shear=target_shear[i]), scale=target_scale[i])
    return new_images


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
        raise ValueError("Wrong target ratio.")

    tr = numpy.array(target_ratio)
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


def uniformPerspective(images, scale=(-0.1, 0.2), p=0.5, inplace=True):
    n = len(images)

    target_ratio = rng.uniform(low=scale[0], high=scale[1], size=(n, 8))
    target_ratio = (target_ratio)/(2*(1-target_ratio))

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


def perspectiveDirect(image, target_point, size=None, resample=PIL.Image.BILINEAR):
    if(len(target_point) < 8):
        raise ValueError("Wrong target point.")

    tr = numpy.array(target_point)
    w, h = image.size
    if size == None:
        size = image.size

    pa = [[0, 0], [h, 0], [h, w], [0, w]]
    pb = []
    pb.append([tr[0], tr[1]])
    pb.append([h+tr[2], tr[3]])
    pb.append([h+tr[0], w+tr[1]])
    pb.append([tr[0], w+tr[1]])

    coeffs = find_coeffs(pa, pb)
    return image.transform(image.size, PIL.Image.PERSPECTIVE, coeffs, resample=resample)


def uniformPerspectiveDirect(images, scale=(-1, 1), p=0.5, inplace=True):
    n = len(images)
    target_point = rng.interger(low=scale[0], high=scale[1], size=(n, 8))
    p_trans = rng.uniform(low=0, high=1, size=n)

    new_images = []
    if inplace:
        for i in range(n):
            if p_trans[i] >= p:
                continue
            images[i] = perspective(images[i], target_point[i])
        new_images = images
    else:
        for i in range(n):
            if p_trans[i] >= p:
                new_images.append(images[i])
                continue
            new_images.append(perspective(images[i], target_point[i]))
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
    "erosion4l": ["N:(.1. .1. .0.)->0"],
    "erosion8l": ["4:(.1. .1. .0.)->0", "4:(1.. .1. ..0)->0"]
}


def morphology(image, **kwargs):
    morph = ImageMorph.MorphOp(**kwargs)
    _, img = morph.apply(image)
    return img


def randomMorph(images, patterns: list, p=0.3, inplace=True):
    n = len(images)
    target_pattern = rng.integers(low=0, high=len(patterns), size=(n))
    p_trans = rng.uniform(low=0, high=1, size=n)

    new_images = []
    if inplace:
        for i in range(n):
            if p_trans[i] >= p:
                continue
            pt = known_patterns[patterns[target_pattern[i]]]
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


def gaussian_kernel(kernel_size, sigma):
    kernel = numpy.zeros(shape=(kernel_size, kernel_size), dtype=numpy.float)
    radius = kernel_size//2
    for y in range(-radius, radius + 1):  # [-r, r]
        for x in range(-radius, radius + 1):
            # 二维高斯函数
            v = 1.0 / (2 * numpy.pi * sigma ** 2) * \
                numpy.exp(-1.0 / (2 * sigma ** 2) * (x ** 2 + y ** 2))
            kernel[y + radius, x + radius] = v  # 高斯函数的x和y值 vs 高斯核的下标值
    kernel2 = kernel / numpy.sum(kernel)
    return kernel2.flatten()


def random_kernel(kernel_size, scale=(-1, 1)):
    kernel = rng.uniform(
        low=scale[0], high=scale[1], size=kernel_size*kernel_size)
    while kernel.sum() == 0:
        kernel = rng.uniform(low=scale[0], high=scale[1])
    kernel = kernel/kernel.sum()
    return kernel


def applyFilter(image, kernel, kernel_size):
    return image.filter(ImageFilter.Kernel((kernel_size, kernel_size), kernel))


def uniformGaussianFilter(images, sigma=(0, 10), kernel_size=3, inplace=True):
    n = len(images)
    sigma_list = rng.uniform(low=sigma[0], high=sigma[1], size=n)

    new_images = []
    if inplace:
        for i in range(n):
            kernel = gaussian_kernel(kernel_size, sigma_list[i])
            images[i] = applyFilter(images[i], kernel, kernel_size)
        new_images = images
    else:
        for i in range(n):
            kernel = gaussian_kernel(kernel_size, sigma_list[i])
            image = applyFilter(images[i], kernel, kernel_size)
            new_images.append(image)
    return new_images


def randomFilter(images, inplace=True, kernel_size=3, scale=(-1, 1)):
    n = len(images)

    new_images = []
    if inplace:
        for i in range(n):
            kernel = random_kernel(kernel_size=kernel_size, scale=scale)
            images[i] = applyFilter(images[i], kernel, kernel_size)
        new_images = images
    else:
        for i in range(n):
            kernel = random_kernel(kernel_size=kernel_size, scale=scale)
            image = applyFilter(images[i], kernel, kernel_size)
            new_images.append(image)
    return new_images


def applyNoise(image, mode, **kwargs):

    img = skiutil.random_noise(numpy.asarray(image), mode=mode, **kwargs)
    # img= skiutil.random_noise(numpy.asarray(image),mode="salt")
    return Image.fromarray(numpy.uint8(img*255))


def randomNoise(images, modes=("gaussian"), inplace=True, mean=(0, 0), var=(0.01, 0.01)):
    n = len(images)

    m = rng.integers(low=0, high=len(modes), size=(n))
    means = rng.uniform(low=mean[0], high=mean[1], size=(n))
    vars = rng.uniform(low=var[0], high=var[1], size=(n))
    new_images = []
    if inplace:
        for i in range(n):
            images[i] = applyNoise(
                images[i], mode=modes[m[i]], mean=means[i], var=vars[i])
        new_images = images
    else:
        for i in range(n):
            image = applyNoise(
                images[i], mode=modes[m[i]], mean=means[i], var=vars[i])
            new_images.append(image)
    return new_images


def sscdCreate(originData, originLabel, transfunction, epoch=1):

    sscd = []
    labels = []
    for _ in range(epoch):
        sscd.extend(transfunction(originData))
        labels.extend(originLabel)
    return sscd, labels


def _getTransform(name, **para: dict):

    if name == "perspective":
        scale = para.setdefault("scale", [0, 0.1])
        p = para.setdefault("p", 0.5)

        def transform(images):
            uniformPerspective(images, scale=scale, p=p)
            return images
    elif name == "perspective_direct":
        scale = para.setdefault("scale", [-1, 1])
        p = para.setdefault("p", 0.5)

        def transform(images):
            uniformPerspectiveDirect(images, scale=scale, p=p)
            return images
    elif name == "affine":
        def transform(images):
            uniformAffine(images, **para)
            return images
    elif name == "affine_direct":
        a14 = para.setdefault("a14", (0.9, 1.1))
        a23 = para.setdefault("a23", (-0.1, 0.1))
        p = para.setdefault("p", 0.5)

        def transform(images):
            uniformAffineDirect(images, range14=a14, range23=a23, p=p)
            return images
    elif name == "color":
        gap = para.setdefault("gap", 50)

        def transform(images):
            randomColorizeSet(images, gap=gap)
            return images
    elif name == "morph":
        def transform(images):
            p = para.setdefault("p", 0.5)
            randomMorph(images, para["patterns"], p=p)
            return images
    elif name == "gaussian_filter":
        sigma = para.setdefault("sigma", (0, 10))

        def transform(images):
            uniformGaussianFilter(images, sigma)
            return images
    elif name == "random_filter":
        scale = para.setdefault("scale", (-1, 1))

        def transform(images):
            randomFilter(images, scale=scale)
            return images
    elif name == "gaussian_noise":
        modes = ("gaussian",)
        mean = para.setdefault("mean", (0, 0))
        var = para.setdefault("var", (0.01, 0.01))

        def transform(images):
            randomNoise(images, modes=modes, mean=mean, var=var)
            return images
    else:
        ValueError("Unsupport transform function: "+"name")
    return transform


def getTransformFunc(para: list):
    trans = []
    if para != None:
        for i in range(len(para)):
            trans.append(_getTransform(**para[i]))

    def transfunc(images):
        for t in trans:
            t(images)
        return

    return transfunc
