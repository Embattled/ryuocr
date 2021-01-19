from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
from PIL import Image, ImageChops


class RyuImageset(Dataset):
    # Init
    def __init__(self, imagePaths, labels, loader=None):
        """
        imagesPaths: List of all font images's path
        labels      : List of all font images's unicode
        loader     : function read a path and return images tensor
        """
        self.imagesPaths = imagePaths
        self.labels = labels

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])

        def default_loader(path):
            img_pil = Image.open(path)
            img_tensor = self.preprocess(img_pil)
            return img_tensor
        if loader==None:
            self.loader = default_loader
        else:
            self.loader=loader

    def __getitem__(self, index):
        path = self.imagesPaths[index]
        img_tensor = self.loader(path)
        label = self.labels[index]
        return img_tensor, label

    def __len__(self):
        return len(self.imagesPaths)


if __name__ == '__main__':
    import pathlib
    import utils

    fontImageDirPath = "/home/eugene/workspace/dataset/fontImage/"
    fontImagesPath = utils.getImagesPath(fontImageDirPath)
    fontlabels = []
    for path in fontImagesPath:
        fontlabels.append(pathlib.Path(path).stem)

    trainset = RyuImageset(fontImagesPath, fontlabels)

    trainLoader = DataLoader(trainset, shuffle=True, batch_size=5)

    for images, labels in trainLoader:
        print(labels)
