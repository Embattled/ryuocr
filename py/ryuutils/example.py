from matplotlib import pyplot as plt
from math import ceil
import random

from PIL import Image
from PIL.ImageOps import scale
import sscd


def sampleUnique(dataset, num, sampleIndex=None, shuffle=False):
    k = min(len(dataset), num)

    if shuffle == True:
        if sampleIndex == None:
            sampleIndex = random.sample(range(len(dataset)), k)
        data = []
        for i in sampleIndex:
            data.append(dataset[i])
        return data
    return dataset[0:k]


# Use PIL make a grid
def makeImageGrid(images, num_col, margin=0, size=(0, 0), background=(255, 255, 255)):
    """
    Input a list of PIL.images, return a grid PIL.image
    num_col : number of column in grid, row of grid will be calculated automatlly.
    """
    num = len(images)
    if num == 0:
        raise ValueError("Empty image list.")

    if size == (0, 0):
        size = images[0].size
    num_row = ceil(num/num_col)

    grid_width = num_col*size[0]+margin*(num_col+1)
    grid_height = num_row*size[1]+margin*(num_row+1)

    grid = Image.new("RGB", size=(grid_width, grid_height), color=background)

    x = margin
    y = margin

    col_count = 0

    for i in range(num):
        pasted = images[i].resize(size)
        grid.paste(pasted, (x, y))
        col_count += 1
        x += size[0]+margin
        if col_count == num_col:
            col_count = 0
            x = margin
            y += size[1]+margin
    return grid


def makeImageGridLabeled(images, labels, num_col, margin=0, size=(0, 0), background=(255, 255, 255)):
    """
    Input a list of PIL.images and list of labels, return a grid PIL.image.
    num_col : number of column in grid, row of grid will be calculated automatlly.
    """
    num = len(images)
    if num == 0 or num != len(labels):
        raise ValueError("Empty image list or unpairable labels and images.")

    num_row = ceil(num/num_col)

    if size == (0, 0):
        size = images[0].size

    labelchar_size = (min(size[0], size[1])) // 6
    labelheight = (labels[0].count("\n")+1)*labelchar_size

    grid_width = num_col*size[0]+margin*(num_col+1)
    grid_height = num_row*(labelheight+size[1])+margin*(num_row+1)

    grid = Image.new("RGB", size=(grid_width, grid_height), color=background)

    x = margin
    y = margin

    col_count = 0

    for i in range(num):
        pasted = images[i].resize(size)
        pasted_char = sscd.font.fontpathLabelImageGet(labels[i], size=(
            size[0], labelheight), padding=1, background=255, fill=0, fontpoint=labelchar_size)

        grid.paste(pasted, (x, y))
        grid.paste(pasted_char, (x, y+size[1]))
        col_count += 1
        x += size[0]+margin
        if col_count == num_col:
            col_count = 0
            x = margin
            y += size[1]+margin+labelheight
    return grid


def getExampleImageGridPIL(dataset, num_col, num_row=None, shuffle=False, margin=0, size=(0, 0), background="white"):
    """
    Random show some examples about a PIL image dataset.
    Using PIL.Image
    """
    if num_row == None:
        num_row = num_col
    sample = sampleUnique(dataset, num_col*num_row, shuffle=shuffle)

    grid = makeImageGrid(sample, num_col, margin, size, background)

    return grid


def getExampleImageLabeledGridPIL(dataset, labels, num_col, num_row=None, shuffle=False, margin=0, size=(0, 0), background="white"):
    """
    Random show some examples about a PIL image dataset.
    Using PIL.Imageshow
    """
    if len(dataset) == 0 or len(dataset) != len(labels):
        raise ValueError("Empty image list or unpairable labels and images.")

    if num_row == None:
        num_row = num_col

    num = min(len(dataset), num_row*num_col)

    sampleIndex = random.sample(range(len(dataset)), num)
    sampleImages = sampleUnique(
        dataset, num, sampleIndex=sampleIndex, shuffle=shuffle)
    sampleLabels = sampleUnique(
        labels, num, sampleIndex=sampleIndex, shuffle=shuffle)

    grid = makeImageGridLabeled(
        sampleImages, sampleLabels, num_col, margin, size, background)

    return grid


def showExamplePLT(dataset, num_col, num_row=None, shuffle=False):
    """
    Random show some examples about a PIL image dataset.
    Using matplotlib.pyplot
    """

    pass


# Run code
if __name__ == "__main__":

    from sscd import *
    from font import *
    from dict import readDict
    from transform import *
    from ryutime import TimeMemo

    fontpath = "/home/eugene/workspace/dataset/font/font/7/AozoraMinchoRegular.ttf"
    image = ttfimageget(fontpath, '龍', background=120)

    image2 = affine(image, shear=(5, 5), size=(128, 128), scale=(0.5, 0.5))

    image2.show()
