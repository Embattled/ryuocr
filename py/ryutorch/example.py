from torchvision import utils
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

# Input a dataset, show some examples.
def showExample(dataset,num,nrow=8,shuffle=True,showScreen=True,outputpath=None):

    trainloader = DataLoader(dataset, batch_size=num, shuffle=shuffle)

    # Image
    iter_data = iter(trainloader)
    images, _ = next(iter_data)

    show_imgs = utils.make_grid(
        images, nrow=nrow).numpy().transpose((1, 2, 0))

    plt.imshow(show_imgs)

    if showScreen:
        plt.show()
    else:
        outputpath = "sscdexample.png" if outputpath==None else outputpath
        plt.savefig(outputpath)