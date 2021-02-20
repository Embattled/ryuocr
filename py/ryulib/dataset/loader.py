from skimage import feature
from torchvision import transforms
from torch import from_numpy
import numpy

# def getHogLoader(preprocess):

#     def hog_loader(img):
#         image = transforms.functional.convert_image_dtype(image)
#         image = preprocess(image)

#         hog_vec, hog_img = feature.hog(image.numpy().transpose(
#             (1, 2, 0)), visualize=True)
#         hogimg_tensor = from_numpy(numpy.array([hog_img]))
#         return image, hogimg_tensor


#     return hog_loader

    
