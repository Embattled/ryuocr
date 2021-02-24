from torchvision import utils
from matplotlib import pyplot as plt

def showExample(dataloader,showScreen=True,outputpath=None):
    # Image
    iter_data = iter(dataloader)
    images, _ = next(iter_data)

    show_imgs = utils.make_grid(
        images, nrow=8).numpy().transpose((1, 2, 0))

    plt.imshow(show_imgs)

    if showScreen:
        plt.show()
    else:
        outputpath = "sscdexample.png" if outputpath==None else outputpath
        plt.savefig(outputpath)


# Hog DataLoder
def showHogExample(dataloader,showHogImage=False,showScreen=True,outputpath=None):
    # Image
    if not showHogImage:
        iter_data = iter(dataloader)
        (images, _), _ = next(iter_data)
        show_imgs = utils.make_grid(
            images, nrow=8).numpy().transpose((1, 2, 0))

        plt.imshow(show_imgs)
    
    # Example with hog
    else:
        iter_data = iter(dataloader)
        (images, hogimage), _ = next(iter_data)

        show_imgs = utils.make_grid(
            images, nrow=8).numpy().transpose((1, 2, 0))
        plt.subplot(1, 2, 1)
        plt.imshow(show_imgs)

        show_imgs = utils.make_grid(
            hogimage, nrow=8).numpy().transpose((1, 2, 0))
        plt.subplot(1, 2, 2)
        plt.imshow(show_imgs)

    if showScreen:
        plt.show()
    else:
        outputpath = "sscdexample.png" if outputpath==None else outputpath
        plt.savefig(outputpath)

