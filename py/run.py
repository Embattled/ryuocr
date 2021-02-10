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

# Copy original data
trainData = originData.clone().detach()

# JPSC1400 Datas
jpsc1400Data = torch.load(
    "/home/eugene/workspace/ryuocr/py/tensordata/jpsc1400tensor64.pt")
jpsc1400Label = torch.load(
    "/home/eugene/workspace/ryuocr/py/tensordata/jpsc1400label3107.pt")


# DataLoder Function
# ----define hog loader-----

def hog_loader(image):
    img_ski = feature.hog(image.numpy().transpose(
        (1, 2, 0))).astype(numpy.float32)
    img_tensor = torch.from_numpy(img_ski)
    return img_tensor


# ----define uint8 loader
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    normalize
])


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

# ------- Transmit font image ----------
# Define Color Change
def changeColor():
    for i in range(len(trainData)):
        cb = torch.randint(256, [3])
        cc = torch.randint(256, [3])

        while(abs(cc[0]-cb[0]) < 50 or abs(cc[1]-cb[1]) < 50 or abs(cc[2]-cb[2]) < 50):
            cb = torch.randint(256, [3])
            cc = torch.randint(256, [3])

        orinp= originData[i].numpy().transpose(1, 2, 0)
        back = (orinp == numpy.array([0, 0, 0])).all(axis=2)
        imnp = trainData[i].numpy().transpose(1, 2, 0)

        imnp[back] = cb
        imnp[numpy.logical_not(back)] = cc
    pass


# Test Color Change
showExample=False
if showExample:
    changeColor()
    iter_data = iter(trainloaderImg)
    images, labels = next(iter_data)

    show_imgs = utils.make_grid(
        images, nrow=8).numpy().transpose((1, 2, 0))
    plt.imshow(show_imgs)
    plt.show()
    sys.exit(0)


iter_data = iter(trainloader)
images, labels = next(iter_data)
print(images.size())


# Train
net = ryulib.model.MLP(images.size()[1], 512, num_class).cuda()
print(net)


# ------ We define the loss function and the optimizer -------
loss_func = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.0001)
epochs = 100

ct_num = 0
running_loss = 0.0

savePath = '/home/eugene/workspace/ryuocr/py/trained/lastTrained.pt'

pfile=open("result.txt",mode='a')
while True:
    sumEpoch = 0
    for epoch in range(epochs):

        # change color every epoch
        # new dataset
        changeColor()
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

        print("[Epoch:"+str(sumEpoch)+"]--MeanLoss:" +
              str(running_loss/ct_num)+"  TrueLoss:"+str(loss.item()),file=pfile)

        if sumEpoch % 5 == 0:
            acc, true_label, pred_label = ryulib.evaluate.evaluate_model(
                net, testloader, False)
            print("Accuracy: "+"%.4f" % acc,file=pfile)

    # Choice continue
    # print("Input 'q' to finish train, other to continue train.")
    # over = input()
    # if over == "q":
    break

torch.save(net, savePath)

# unicodeJPSC = ryulib.dataset.sscd.getCodeJPSC()
# labelJPSC = []
# for code in unicodeJPSC:
#     labelJPSC.append(code2label[str(code)])

# JPSC1400 = ryulib.dataset.sscd.JPSC1400(labelJPSC, loader=hog_loader)
# testloader = DataLoader(JPSC1400, batch_size=128, shuffle=False)

# acc, true_label, pred_label = ryulib.evaluate.evaluate_model(
#     net, testloader, False)
# print("Accuracy: "+"%.3f" % acc)
