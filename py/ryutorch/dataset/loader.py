from torch import from_numpy

# Import
from torchvision import transforms, io
import numpy

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


def getLoader(size,is_path:bool,preprocess=None):

    if preprocess == None:
        preprocess = transforms.Compose([
            transforms.Resize(size),
            normalize
        ])

    if is_path:
        def _func(path):
            image = io.read_image(path)
            
            # Convert uint8 to float32 and rescale
            image = transforms.functional.convert_image_dtype(image)
            img_tensor = preprocess(image)
            return img_tensor
    else:
        def _func(image):
            image = transforms.functional.convert_image_dtype(image)
            img_tensor = preprocess(image)
            return img_tensor
    return _func

