from baseset import RyuImageset
from utils import getFilesPath, kanji2unicode
from torchvision import transforms, utils
from PIL import Image, ImageChops
import pandas


class SSCDset(RyuImageset):
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
            transforms.ToTensor(),
            self.normalize
        ])

        def sscd_loader(path):
            img_pil = Image.open(path)
            img_pil = img_pil.resize((size, size))
            img_tensor = self.preprocess(img_pil)
            return img_tensor

        if loader == None:
            super().__init__(imagePaths, labels, sscd_loader)
        else:
            super().__init__(imagePaths, labels, loader)


def getCodeJPSC():
    labelFile = "/home/eugene/workspace/dataset/JPSC1400-20201218/labelryu.txt"
    jpscLabels = list(pandas.read_csv(
        labelFile, index_col=0)["Unicode"])
    return jpscLabels


class JPSC1400(SSCDset):
    def __init__(self, newLabel, size=64, loader=None):
        """
        Read JPSC1400 as pytorch map-key dataset
        """
        self.imagesDir = "/home/eugene/workspace/dataset/JPSC1400-20201218/png"
        self.imagesPaths = getFilesPath(self.imagesDir)
        if loader == None:
            super().__init__(self.imagesPaths, newLabel, size)
        else:
            super().__init__(self.imagesPaths, newLabel, size, loader=loader)

    # labels = pd.read_csv(RJPSC1400Path)

    # for i in range(len(data["character"])):
    #     word=data["character"][i]

    #     # labels[word]=word.encode('unicode-escape').decode()[2:]
    #     labels["code"][i]=int(word.encode('unicode-escape').decode()[2:],16)

    # print(word.encode('unicode-escape').decode())
    # print(word.encode('unicode-escape'))


if __name__ == '__main__':
    import pandas
    # names=['index','percentcode','character']
    # jpscpath="/home/eugene/workspace/dataset/JPSC1400-20201218/label.txt"
    # jpsclabel=pandas.read_csv(jpscpath,names=names,sep=" ",index_col='index')
    # jpsclabel.to_csv("/home/eugene/workspace/dataset/JPSC1400-20201218/labelryu.txt")

    # jpscpath="/home/eugene/workspace/dataset/JPSC1400-20201218/labelryu.txt"
    # jpsclabel=pandas.read_csv(jpscpath,index_col=0)
    # jpsclabel["Unicode"]=kanji2unicode(jpsclabel["character"].values.flatten())
    # print(jpsclabel.head())
    # jpsclabel.to_csv("/home/eugene/workspace/dataset/JPSC1400-20201218/labelryu.txt")

    # from torch.utils.data import Dataset, DataLoader
    # JPSC=JPSC1400()
    # # print(JPSC.jpscLabels)
    # testLoader=DataLoader(JPSC,shuffle=False,batch_size=5)
    # for images,labels in testLoader:
    #     print(images.size())
    #     print(labels)
