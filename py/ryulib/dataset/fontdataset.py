from PIL import Image
from torchvision import transforms


from baseset import RyuImageset

# 当然出来的时候已经全都变成了tensor


class FontTrainSet(RyuImageset):
    # Init
    def __init__(self, imagePaths, labels, size=64, loader=None):
        """
        imagesPaths: List of all font images's path
        labels      : List of all font images's unicode
        loader     : function read a path and return images tensor
        """

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.preprocess = transforms.Compose([
            # transforms.Scale(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize
        ])

        def font_loader(path):
            img_pil = Image.open(path)
            img_pil = img_pil.resize((size, size))
            img_tensor = self.preprocess(img_pil)
            return img_tensor

        if(loader == None):
            super().__init__(imagePaths, labels, font_loader)
        else:
            super().__init__(imagePaths, labels, loader)
