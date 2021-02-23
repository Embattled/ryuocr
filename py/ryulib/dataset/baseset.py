from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
from PIL import Image, ImageChops


class RyuImageset(Dataset):
    # Init
    def __init__(self, images, labels, loader):
        """
        images      : List of all font images's path
        labels      : List of all font images's unicode
        loader      : function read a path and return images tensor
        """
        self.images = images
        self.labels = labels
        self.loader= loader

    def __getitem__(self, index):
        image = self.images[index]
        img_tensor = self.loader(image)
        label = self.labels[index]
        return img_tensor, label

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    import pathlib
    import utils

    fontImageDirPath = "/home/eugene/workspace/dataset/fontImage/"
    fontImagesPath = utils.getFilesPath(fontImageDirPath)
    fontlabels = []
    for path in fontImagesPath:
        fontlabels.append(pathlib.Path(path).stem)

    trainset = RyuImageset(fontImagesPath, fontlabels)

    trainLoader = DataLoader(trainset, shuffle=True, batch_size=5)

    for images, labels in trainLoader:
        print(labels)
