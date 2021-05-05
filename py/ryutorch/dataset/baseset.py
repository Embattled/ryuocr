from torch.utils.data import Dataset, DataLoader
import torch

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

    def changeData(self,images,labels=None):
        if labels==None:
            if len(images)!=len(self.images):
                raise ValueError("Length of new data is different with old data.")
            self.images=images
        else:
            if len(images)!=len(self.images):
                raise ValueError("Length of new data is different with new label.")
            self.images = images
            self.labels = labels