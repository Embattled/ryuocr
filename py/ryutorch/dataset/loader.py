from skimage import feature
from torchvision import transforms, io
from torch import from_numpy
import numpy

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

default_preprocess = transforms.Compose([
    normalize
])

class RyuLoader(object):
    def __init__(self,preprocess=None):
        self.preprocess=default_preprocess
        if preprocess != None:
            self.preprocess=preprocess

    def file_loader(self,path):
        image = io.read_image(path)
        # Convert uint8 to float32 and rescale
        image = transforms.functional.convert_image_dtype(image)
        img_tensor = self.preprocess(image)
        return img_tensor

    def tensor_loader(self,image):
        image=transforms.functional.convert_image_dtype(image)
        img_tensor = self.preprocess(image)
        return img_tensor

    # def tensor_hogvisual_loader(self,image):

    #     image = transforms.functional.convert_image_dtype(image)
    #     image = self.preprocess(image)

    #     _, hog_img = feature.hog(image.numpy().transpose(
    #         (1, 2, 0)), visualize=True)

    #     hogimg_tensor = from_numpy(numpy.array([hog_img]))
    #     return image, hogimg_tensor

    def file_hog_loader(self,path):
        image = io.read_image(path)
        # Convert uint8 to float32 and rescale
        image = transforms.functional.convert_image_dtype(image)
        image = self.preprocess(image)

        img_ski = feature.hog(image.numpy().transpose(
            (1, 2, 0))).astype(numpy.float32)
        hog_tensor = from_numpy(img_ski)
        return hog_tensor

    def tensor_hog_loader(self,image):
        image = transforms.functional.convert_image_dtype(image)
        image = self.preprocess(image)
        img_ski = feature.hog(image.numpy().transpose(
            (1, 2, 0))).astype(numpy.float32)
        hog_tensor = from_numpy(img_ski)
        return hog_tensor