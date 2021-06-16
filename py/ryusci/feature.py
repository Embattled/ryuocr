from skimage import feature
from skimage import transform

def getHogFun(img_size: tuple, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3)):
    """
    Return a single parameter function, 
    which input numpy array image data, 
    return hog feature vector.
    """
    def _hog(image):

        if image.shape[:2] != img_size:
            _image = transform.resize(image, img_size)
            _feature = feature.hog(_image, orientations=orientations,
                           pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
        else:
            _feature = feature.hog(image, orientations=orientations,
                           pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
        return _feature
    return _hog
