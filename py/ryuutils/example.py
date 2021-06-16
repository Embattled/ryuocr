from PIL.ImageOps import scale
from matplotlib import pyplot as plt
from PIL import Image
from math import ceil
import random

from numpy.core.fromnumeric import size

# Use PIL make a grid
def make_grid(images, num_col, margin=0, size=None, background=(255, 255, 255)):
    if size == None:
        size = images[0].size
    num = len(images)
    if num == 0:
        return
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


def sampleUnique(dataset, num, shuffle=False):
    k = min(len(dataset), num)

    if shuffle:
        return random.sample(dataset, k)
    return dataset[0:k]


def showExamplePIL(dataset, num_col, num_row=None, shuffle=False, margin=0, size=None, background="white"):
    """
    Random show some examples about a PIL image dataset.
    Using PIL.Imageshow
    """
    if num_row == None:
        num_row = num_col
    sample = sampleUnique(dataset, num_col*num_row, shuffle)
    grid = make_grid(sample, num_col, margin, size, background)

    grid.show()


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

    fontpath="/home/eugene/workspace/dataset/font/font/7/AozoraMinchoRegular.ttf"
    image=ttfimageget(fontpath,'龍',background=120)

    image2=affine(image,shear=(5,5),size=(128,128),scale=(0.5,0.5))

    image2.show()