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
outputScreen = False
# outputScreen = True

# Set weather show trainset example
showExample = False
# showExample = True
showHogImage = False


# Network model save path
savePath = '/home/eugene/workspace/ryuocr/py/trained/' + \
    myhostname+'_'+nowTimeStr+'.pt'
# Accuracy History Graph Save Path
accGraphSavePath = "/home/eugene/workspace/ryuocr/py/output/output_" + \
    myhostname+"_"+nowTimeStr+"_acc.png"

if not outputScreen:
    outputfilename = "/home/eugene/workspace/ryuocr/py/output/output_" + \
        myhostname+"_"+nowTimeStr+".txt"
    pfile = open(outputfilename, mode='w')
    sys.stdout = pfile

# ----------------- Training parameter ------------------------

epochs = 150
learning_rate = 0.0001
batch_size = 512
loss_func = nn.CrossEntropyLoss()


ensemble_num = 15

# ------------ Run ------------------

# Label Dict
labelData = pd.read_csv(
    "/home/eugene/workspace/dataset/font/3107jp.csv", index_col=0)
labelDict = dict(zip(labelData['utfcode'].values, labelData.index.values))
num_cls = len(labelDict)


# Train Datas
trainLabel = torch.load(
    "/home/eugene/workspace/ryuocr/py/tensordata/7font3107label.pt")
originData = torch.load(
    "/home/eugene/workspace/ryuocr/py/tensordata/7font3107img.pt")
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
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    normalize
])


def fpreprocess(trainData, originData):
    rt.recovery(trainData, originData)
    rt.perspective(trainData, distortion_scale=0.3, p=0.7)
    rt.affine(trainData, anglerange=10, scalerange=0.1, shearrange=10)
    rt.changeColor(trainData)
    pass


print("Total epochs: " + str(epochs))
print("Train batch size: "+str(batch_size))
if ensemble_num!=1:
    print("Ensembled system, number of models = "+str(ensemble_num))


print("Use new transform function")
print("perspective: scale=0.3, p=0.7")
print("affine: degree=10, shear=10, scale=0.1")
print("Use Gaussian Blur k=3, sigma 0.1 2,0")


# DataLoder Function
# ----define hog loader-----

# Train Dataset
if showExample:
    trainsetHog = ryulib.dataset.RyuImageset(
        trainData, trainLabel, loader=ryulib.dataset.loader.tensor_hogvisual_loader)
else:
    def hog_loader(image):
        image = transforms.functional.convert_image_dtype(image)
        image = preprocess(image)
        img_ski = feature.hog(image.numpy().transpose(
            (1, 2, 0))).astype(numpy.float32)
        hog_tensor = torch.from_numpy(img_ski)
        return hog_tensor
    trainsetHog = ryulib.dataset.RyuImageset(
        trainData, trainLabel, loader=hog_loader)


# trainsetImg = ryulib.dataset.FontTrainSet(
#     trainData, trainLabel, loader=uint8_loader)

# Train Dataloader
trainloader = DataLoader(trainsetHog, batch_size=batch_size, shuffle=True)
# trainloaderImg = DataLoader(trainsetImg, batch_size=64, shuffle=True)

# Test Dataset
testsetHog = ryulib.dataset.RyuImageset(
    jpsc1400Data, jpsc1400Label, loader=ryulib.dataset.loader.tensor_hog_loader)

# Test Dataloader
testloader = DataLoader(testsetHog, batch_size=128, shuffle=False)


# Test Transform
if showExample:
    fpreprocess(trainData, originData)
    ryulib.example.showHogExample(
        trainloader, showScreen=outputScreen, showHogImage=showHogImage)
    sys.exit(0)


iter_data = iter(trainloader)
images, labels = next(iter_data)
print("Input tensor's shape", images.size())


ct_num = 0
running_loss = 0.0


# Train Single
print("Training Start...\n")
if ensemble_num == 1:

    net = ryulib.model.MLP(images.size()[1], 512, num_cls).cuda()
    print(net)

    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

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
            # print("Iteration: "+str(iteration+1)+", Loss: "+str(running_loss/ct_num)+'.')
            # print("[Epoch: "+str(epoch+1)+"]"" --- Iteration: " +
            #     str(iteration+1)+", Loss: "+str(running_loss/ct_num)+'.')

        meanloss = running_loss/ct_num
        print("[Epoch:"+str(sumEpoch)+"]--MeanLoss:" +
              str(meanloss)+"  TrueLoss:"+str(loss.item()))

        # if sumEpoch % 5 == 0:
        acc, true_label, pred_label = ryulib.evaluate.evaluate_model(
            net, testloader)
        print("At epoch:"+str(sumEpoch)+" accuracy: "+"%.4f" % acc)

        #  Save loss and accuracy
        accuracy_history.append(acc)
        meanloss_history.append(meanloss)


#  Define ensemble system
else:
    net = []
    accuracy_history = []
    for netid in range(ensemble_num):
        
        print("Start training net "+str(netid+1)+"/"+str(ensemble_num))
        net.append(ryulib.model.MLP(images.size()[1], 512, num_cls).cuda())
        optimizer = optim.Adam(net[netid].parameters(), lr=learning_rate)

        sumEpoch = 0

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
                outputs = net[netid](inputs)

                # Compute loss
                loss = loss_func(outputs, labels)

                loss.backward()
                optimizer.step()

                # with torch.no_grad():
                running_loss += loss.item()
                ct_num += 1

            meanloss = running_loss/ct_num
            print("[Epoch:"+str(sumEpoch)+"]--MeanLoss:" +
                  str(meanloss)+"  TrueLoss:"+str(loss.item()))

            if sumEpoch % 10 == 0:
                acc, true_label, pred_label = ryulib.evaluate.evaluate_model(
                    net[netid], testloader)
                print("At epoch:"+str(sumEpoch)+" accuracy: "+"%.4f" % acc)


        #  Save accuracy history
        acc, true_label, pred_label=ryulib.evaluate.evaluate_ensemble(
            net, testloader, num_cls)
        accuracy_history.append(acc)
        print("Number of models="+str(netid+1)+" accuracy: "+"%.4f" % acc)



print("--------------------Training Finish---------------------------")
torch.save(net, savePath)
print("Pure Data:")
print(accuracy_history)
# Draw a graph

plt.xlabel("Training Epoch")
plt.ylabel("Accuracy")

plt.plot(range(1, epochs+1), accuracy_history)

maxacc, maxaccepoch=torch.max(torch.tensor(accuracy_history), 0)
maxaccepoch=maxaccepoch.item()+1

coord=(maxaccepoch-150, maxacc.item())
plt.grid()

plt.annotate("Max accuracy: "+"%.4f" % maxacc.item() +
             " epoch: "+str(maxaccepoch), coord, xytext=coord)
plt.savefig(accGraphSavePath)


# JPSC1400 = ryulib.dataset.sscd.JPSC1400(labelJPSC, loader=hog_loader)
# testloader = DataLoader(JPSC1400, batch_size=128, shuffle=False)

# acc, true_label, pred_label = ryulib.evaluate.evaluate_model(
#     net, testloader, False)
# print("Accuracy: "+"%.3f" % acc)
