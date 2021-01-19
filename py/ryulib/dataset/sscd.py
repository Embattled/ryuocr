from baseset import RyuImageset
from utils import getImagesPath, kanji2unicode
from torchvision import transforms, utils
from PIL import Image, ImageChops
import pandas

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    # transforms.Scale(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


def sscd_loader(path):
    img_pil = Image.open(path)
    img_pil = img_pil.resize((64, 64))
    img_pil = ImageChops.invert(img_pil)
    img_tensor = preprocess(img_pil)
    return img_tensor

# 当然出来的时候已经全都变成了tensor


class SSCDset(RyuImageset):
    # Init
    def __init__(self, imagePaths, labels, size=64):
        """
        imagesPaths: List of all font images's path
        labels      : List of all font images's unicode
        loader     : function read a path and return images tensor
        """
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        preprocess = transforms.Compose([
            # transforms.Scale(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        def sscd_loader(path):
            img_pil = Image.open(path)
            img_pil = img_pil.resize((size, size))
            img_tensor = preprocess(img_pil)
            return img_tensor
        super().__init__(imagePaths, labels, sscd_loader)


class JPSC1400(SSCDset):
    def __init__(self):
        """
        Read JPSC1400 as pytorch map-key dataset
        """
        self.labelPath = "/home/eugene/workspace/dataset/JPSC1400-20201218/labelryu.txt"
        self.jpscLabels = list(pandas.read_csv(
            self.labelPath, index_col=0)["Unicode"])
        self.imagesDir = "/home/eugene/workspace/dataset/JPSC1400-20201218/png"
        self.imagesPaths = getImagesPath(self.imagesDir)
        super().__init__(self.imagesPaths, self.jpscLabels)

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
