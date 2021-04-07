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
from inspect import getsource

import ryulib
from ryulib import transform as rt

import ryuutils
from ryuutils.ryutime import TimeMemo
# ------------program global config -----------
timeMemo=TimeMemo()
nowTimeStr = timeMemo.nowTimeStr()
print(nowTimeStr)

myhostname = gethostname()

# Set print goal
# outputScreen = False
outputScreen = True

# Set weather show trainset example
# showExample = False
showExample = True


showHogImage = False


# Network model save path
savePath = '/home/eugene/workspace/ryuocr/py/trained/' + \
    nowTimeStr+'.pt'
# Accuracy History Graph Save Path
accGraphSavePath = "/home/eugene/workspace/ryuocr/py/output/output_" + \
    nowTimeStr+"_acc.png"

if (not showExample) and (not outputScreen):
    outputfilename = "/home/eugene/workspace/ryuocr/py/output/output_" + \
        nowTimeStr+".txt"
    pfile = open(outputfilename, mode='w')
    sys.stdout = pfile

# ----------------- Training parameter ------------------------

epochs = 20
learning_rate = 0.0001
batch_size = 128
loss_func = nn.CrossEntropyLoss()


# ------------ Run ------------------

# Label Dict
labelData = pd.read_csv(
    "/home/eugene/workspace/dataset/font/3107jp.csv", index_col=0)
labelDict = dict(zip(labelData['utfcode'].values, labelData.index.values))
num_cls = len(labelDict)


# Train Datas
# fontLabel = torch.load(
#     "/home/eugene/workspace/ryuocr/py/tensordata/7font3107label.pt")
# fontData = torch.load(
#     "/home/eugene/workspace/ryuocr/py/tensordata/7font3107img.pt")

sscdDataFile="20_20210406184753_train.pt"
sscdLabelFile="20_20210406184753_label.pt"

trainData=torch.load('/home/eugene/workspace/ryuocr/py/tensorsscd/'+sscdDataFile)
trainLabel=torch.load('/home/eugene/workspace/ryuocr/py/tensorsscd/'+sscdLabelFile)
print("Loding SSCD data success.")


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


print("Epochs:\t\t" + str(epochs))
print("Batch size:\t"+str(batch_size))
print("LearningRate:\t"+str(learning_rate))


print("Preprocess:")
print(preprocess)


print("Training SSCD: " + sscdDataFile)
print("Training SSCD size:")
print(trainData.size())
print(trainLabel.size())
# sys.exit(0)

# DataLoder Function
# ----define hog loader-----

# Train Dataset
if showExample:
    batch_size=64
    trainsetHog = ryulib.dataset.RyuImageset(
        trainData, trainLabel, loader=ryulib.dataset.loader.tensor_hogvisual_loader)
else:
    trainsetHog = ryulib.dataset.RyuImageset(
        trainData, trainLabel, loader=ryulib.dataset.loader.tensor_hog_loader)


# Train Dataloader
trainloader = DataLoader(trainsetHog, batch_size=batch_size, shuffle=True)

# Test Dataset
testsetHog = ryulib.dataset.RyuImageset(
    jpsc1400Data, jpsc1400Label, loader=ryulib.dataset.loader.tensor_hog_loader)

# Test Dataloader
testloader = DataLoader(testsetHog, batch_size=128, shuffle=False)


# Test Transform
if showExample:
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
starttime=time.time()

net = ryulib.model.MLP(images.size()[1], 512, num_cls).cuda()
print(net)

# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

sumEpoch = 0
accuracy_history = []
meanloss_history = []

for epoch in range(epochs):

    # Transform every epoch
    # fpreprocess(trainData, fontData)

    sumEpoch += 1

    for iteration, data in enumerate(trainloader):
        # Take the inputs and the labels for 1 batch.
        images, labels = data

        bch = images.size(0)
        # inputs = inputs.view(bch, -1) <-- We don't need to reshape inputs here (we are using CNNs).

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

        if outputScreen and iteration%20==0:
            print("[Epoch: "+str(sumEpoch)+"]"" --- Iteration: " +
                str(iteration+1)+", Loss: "+str(running_loss/ct_num)+'.')

    meanloss = running_loss/ct_num
    print("[Epoch:"+str(sumEpoch)+"]--MeanLoss:" +
            str(meanloss))

    # if sumEpoch % 5 == 0:
    acc, true_label, pred_label = ryulib.evaluate.evaluate_model(
        net, testloader)
    print("At epoch:"+str(sumEpoch)+" accuracy: "+"%.4f" % acc)

    #  Save loss and accuracy
    accuracy_history.append(acc)
    meanloss_history.append(meanloss)



finishtime=time.time()
print("--------------------Training Finish---------------------------")
torch.save(net, savePath)
print("Pure Data accuracy history:")
print(accuracy_history)
print("Pure Data mean loss history:")
print(meanloss_history)
# Draw a graph

plt.xlabel("Training Epoch")
plt.ylabel("Accuracy")

plt.plot(range(1, epochs+1), accuracy_history)

maxacc, maxaccepoch=torch.max(torch.tensor(accuracy_history), 0)
maxaccepoch=maxaccepoch.item()+1

coord=(maxaccepoch-10, maxacc.item()+0.01)
plt.grid()

plt.annotate("%.3f" % maxacc.item() +
             " : "+str(maxaccepoch), coord, xytext=coord)
if outputScreen:
    plt.show()
else:
    plt.savefig(accGraphSavePath)

# JPSC1400 = ryulib.dataset.sscd.JPSC1400(labelJPSC, loader=hog_loader)
# testloader = DataLoader(JPSC1400, batch_size=128, shuffle=False)

# acc, true_label, pred_label = ryulib.evaluate.evaluate_model(
#     net, testloader, False)
# print("Accuracy: "+"%.3f" % acc)
