from typing import Sequence

import numpy
from skimage import feature
from skimage import transform

def getHogFun(img_size, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3)):
    """
    Return a single parameter function, 
    which input numpy array image data, 
    return hog feature vector.
    """
    if isinstance(img_size,int):
        img_size=(img_size,img_size)
    elif isinstance(img_size,Sequence) and len(img_size)==1:
        img_size=(img_size[0],img_size[0])
    def _hog(image):
        image = numpy.asarray(image)
        if image.shape[:2] != img_size:
            _image = transform.resize(image, img_size)
            _feature = feature.hog(_image, orientations=orientations,
                           pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
        else:
            _feature = feature.hog(image, orientations=orientations,
                           pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
        return _feature
    return _hog



def getFeature(args:dict):
    name=args["name"]

    if name=="hog":
        return getHogFun(**args["hog"])

