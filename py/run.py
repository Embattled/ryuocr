from matplotlib import pyplot as plt
from socket import gethostname
import torch
import torch.nn as nn
import torch.tensor
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils, models

from PIL import Image
import pathlib
import matplotlib.pyplot as plt

from skimage import feature
from skimage import transform as sktr
from skimage import io as skio
import numpy
import pandas as pd

import sys
import ryulib
from ryulib import transform as rt

# ------------program global config -----------
import time
nowTimeStr = time.strftime("%Y%m%d%H%M%S")
myhostname = gethostname()

# Set print goal
# outputScreen = False
outputScreen = True


# Set weather show trainset example
# showExample = False
showExample = True
showHogImage = False


# Network model save path
savePath = '/home/eugene/workspace/ryuocr/py/trained/'+myhostname+'lastTrained.pt'
# Accuracy History Graph Save Path
accGraphSavePath = "/home/eugene/workspace/ryuocr/py/output/output_" + \
    myhostname+"_"+nowTimeStr+"_acc.png"

if not outputScreen:
    outputfilename = "/home/eugene/workspace/ryuocr/py/output/output_" + \
        myhostname+"_"+nowTimeStr+".txt"
    pfile = open(outputfilename, mode='w')
    sys.stdout = pfile

# ----------------- Training parameter ------------------------

epochs = 200


# ------------ Run ------------------

# Label Dict
labelData = pd.read_csv(
    "/home/eugene/workspace/dataset/3107jp.csv", index_col=0)
labelDict = dict(zip(labelData['utfcode'].values, labelData.index.values))
num_class = len(labelDict)


# Train Datas
trainLabel = torch.load(
    "/home/eugene/workspace/ryuocr/py//tensordata/7font3107label.pt")
originData = torch.load(
    "/home/eugene/workspace/ryuocr/py//tensordata/7font3107img.pt")
print("Loding font image success.")

# Copy original data
trainData = originData.clone().detach()

# JPSC1400 Datas
jpsc1400Data = torch.load(
    "/home/eugene/workspace/ryuocr/py/tensordata/jpsc1400tensor64.pt")
jpsc1400Label = torch.load(
    "/home/eugene/workspace/ryuocr/py/tensordata/jpsc1400label3107.pt")
print("Loding jpsc1400 image success.")


# Define Transforms
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    # transforms.RandomPerspective(),
    # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    normalize
])


def fpreprocess(trainData, originData):
    rt.recovery(trainData, originData)
    rt.affine(trainData, anglerange=10, scalerange=0.1, shearrange=10)
    rt.changeColor(trainData)
    pass


# print("Use new transform function, color change transform")
print("Use new transform function, affine: degree=10, shear=10, scale=0.1")
print("Use Gaussian Blur k=3, sigma 0.1 2,0")


# DataLoder Function
# ----define hog loader-----
if showExample:
    def hog_loader(image):

        image = transforms.functional.convert_image_dtype(image)
        image = preprocess(image)

        hog_vec, hog_img = feature.hog(image.numpy().transpose(
            (1, 2, 0)), visualize=True)
        hogimg_tensor = torch.from_numpy(numpy.array([hog_img]))
        return image, hogimg_tensor
else:
    def hog_loader(image):
        image = transforms.functional.convert_image_dtype(image)
        image = preprocess(image)
        img_ski = feature.hog(image.numpy().transpose(
            (1, 2, 0))).astype(numpy.float32)
        hog_tensor = torch.from_numpy(img_ski)
        return hog_tensor


def uint8_loader(image):
    image = transforms.functional.convert_image_dtype(image)
    img_tensor = preprocess(image)
    return img_tensor


# Train Dataset
trainsetHog = ryulib.dataset.FontTrainSet(
    trainData, trainLabel, loader=hog_loader)
trainsetImg = ryulib.dataset.FontTrainSet(
    trainData, trainLabel, loader=uint8_loader)

# Train Dataloader
trainloader = DataLoader(trainsetHog, batch_size=64, shuffle=True)
trainloaderImg = DataLoader(trainsetImg, batch_size=64, shuffle=True)

# Test Dataset
testsetHog = ryulib.dataset.sscd.SSCDset(
    jpsc1400Data, jpsc1400Label, loader=hog_loader)
# Test Dataloader
testloader = DataLoader(testsetHog, batch_size=128, shuffle=False)


# Test Transform
if showExample:

    fpreprocess(trainData, originData)

    # Image
    if not showHogImage:
        iter_data = iter(trainloader)
        (images, hogimage), labels = next(iter_data)
        show_imgs = utils.make_grid(
            images, nrow=8).numpy().transpose((1, 2, 0))

        plt.imshow(show_imgs)
    # Example with hog
    if showHogImage:
        iter_data = iter(trainloader)
        (images, hogimage), labels = next(iter_data)

        show_imgs = utils.make_grid(
            images, nrow=8).numpy().transpose((1, 2, 0))
        plt.subplot(1, 2, 1)
        plt.imshow(show_imgs)

        show_imgs = utils.make_grid(
            hogimage, nrow=8).numpy().transpose((1, 2, 0))
        plt.subplot(1, 2, 2)
        plt.imshow(show_imgs)

    # plt.show()
    plt.savefig("sscdexample.png")
    sys.exit(0)


iter_data = iter(trainloader)
images, labels = next(iter_data)
print("Input tensor's shape", images.size())


ct_num = 0
running_loss = 0.0


#  Define ensemble system
net = ryulib.model.MLP(images.size()[1], 512, num_class).cuda()
print(net)
loss_func = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.0001)

# Train
print("Training Start...\n")
while True:
    sumEpoch = 0
    accuracy_history = []
    meanloss_history = []

    for epoch in range(epochs):

        # Transform every epoch
        fpreprocess(trainData, originData)

        sumEpoch += 1

        for iteration, data in enumerate(trainloader):
            # Take the inputs and the labels for 1 batch.
            images, labels = data

            # bch = images.size(0)
            # inputs = inputs.view(bch, -1) <-- We don't need to reshape inputs here (we are using CNNs).

            # inputs = []
            # for i, image in enumerate(images):
            #     inputs.append(feature.hog(
            #         image.numpy().transpose(1, 2, 0), 8, (8, 8), (2, 2)))
            # inputs = torch.from_numpy(numpy.array(inputs, dtype=numpy.float32))

            inputs = images
            # Move inputs and labels into GPU
            inputs = inputs.cuda()
            labels = labels.cuda()

            # Remove old gradients for the optimizer.
            optimizer.zero_grad()

            # Compute result (Forward)
            outputs = net(inputs)

            # Compute loss
            loss = loss_func(outputs, labels)

            loss.backward()
            optimizer.step()

            # with torch.no_grad():
            running_loss += loss.item()
            ct_num += 1
            # if iteration == 0:
            #print("Iteration: "+str(iteration+1)+", Loss: "+str(running_loss/ct_num)+'.')
            # print("[Epoch: "+str(epoch+1)+"]"" --- Iteration: " +
            #     str(iteration+1)+", Loss: "+str(running_loss/ct_num)+'.')

        meanloss = running_loss/ct_num
        print("[Epoch:"+str(sumEpoch)+"]--MeanLoss:" +
              str(meanloss)+"  TrueLoss:"+str(loss.item()))

        # if sumEpoch % 5 == 0:
        acc, true_label, pred_label = ryulib.evaluate.evaluate_model(
            net, testloader, False)
        print("Accuracy: "+"%.4f" % acc)

        #  Save loss and accuracy
        accuracy_history.append(acc)
        meanloss_history.append(meanloss)

    # Choice continue
    # print("Input 'q' to finish train, other to continue train.")

    # over = input()
    # if over == "q":
    break

torch.save(net, savePath)
print("--------------------Training Finish---------------------------")

# Draw a graph

plt.xlabel("Training Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.plot(range(1, epochs+1), accuracy_history)
plt.savefig(accGraphSavePath)


# unicodeJPSC = ryulib.dataset.sscd.getCodeJPSC()
# labelJPSC = []
# for code in unicodeJPSC:
#     labelJPSC.append(code2label[str(code)])

# JPSC1400 = ryulib.dataset.sscd.JPSC1400(labelJPSC, loader=hog_loader)
# testloader = DataLoader(JPSC1400, batch_size=128, shuffle=False)

# acc, true_label, pred_label = ryulib.evaluate.evaluate_model(
#     net, testloader, False)
# print("Accuracy: "+"%.3f" % acc)
