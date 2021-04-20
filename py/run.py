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


import ryuutils
from ryuutils.ryutime import TimeMemo
from ryuutils import ryuyaml as yaml

import ryutorch
from ryutorch import model,evaluate
import ryutorch.dataset
from ryutorch.dataset.baseset import RyuImageset
from ryutorch.dataset.loader import RyuLoader

# ------------program global mode -----------
timeMemo = TimeMemo()
nowTimeStr = timeMemo.nowTimeStr()
print(nowTimeStr)

myhostname = gethostname()

# Set print goal
# outputScreen = False
outputScreen = True


# ----------- training parameter ---------
configPath = "/home/eugene/workspace/ryuocr/py/config/torchconfig.yml"

config = yaml.loadyaml(configPath)
# ---------- global parameter -----

paramGlobal = config["Global"]
# Network model save path
savePath = paramGlobal["save_model_dir"] + nowTimeStr+'.pt'
# Accuracy History Graph Save Path
accGraphSavePath = paramGlobal["save_result_dir"] + nowTimeStr+"_acc.png"
logpath = paramGlobal["save_result_dir"] + nowTimeStr+".yml"


if not outputScreen:
    outputfilename = paramGlobal["save_result_dir"] + nowTimeStr+".txt"
    pfile = open(outputfilename, mode='w')
    sys.stdout = pfile


epochs = paramGlobal["epoch"]

# ----------  Optimizer ----
paramOpt = config["Optimizer"]

learning_rate = paramOpt["learning_rate"]
loss_func = nn.CrossEntropyLoss()


# ------------ Dataset -------------
# Train dataset
paramTrain = config["Train"]
train_batch_size = paramTrain["loader"]["batch_size"]
train_shuffle = paramTrain["loader"]["shuffle"]

trainsetName = paramTrain["dataset"]["name"]
trainsetPath = paramTrain["dataset"]["data_dir"]+trainsetName+"_train.pt"
trainsetLabel = paramTrain["dataset"]["label_dir"]+trainsetName+"_label.pt"

trainData = torch.load(trainsetPath)
trainLabel = torch.load(trainsetLabel)


# Test dataset
paramTest = config["Test"]
test_batch_size = paramTest["loader"]["batch_size"]
test_shuffle = paramTest["loader"]["shuffle"]

testsetName = paramTest["dataset"]["name"]
testsetPath = paramTest["dataset"]["data_dir"]
testsetLabel = paramTest["dataset"]["label_dir"]

jpsc1400Data = torch.load(testsetPath)
jpsc1400Label = torch.load(testsetLabel)

# Label Dict
labelData = pd.read_csv(
    "/home/eugene/workspace/dataset/font/3107jp.csv", index_col=0)
labelDict = dict(zip(labelData['utfcode'].values, labelData.index.values))
num_cls = len(labelDict)


print("Loding data success.")
# ------------- Preprocess and Dataloader

# Define Transforms
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    normalize
])

MyLoaders = RyuLoader(preprocess)

print("Preprocess:")
print(preprocess)


trainsetHog = RyuImageset(
    trainData, trainLabel, loader=MyLoaders.tensor_hog_loader)
trainloader = DataLoader(
    trainsetHog, batch_size=train_batch_size, shuffle=train_shuffle)

testsetHog = RyuImageset(
    jpsc1400Data, jpsc1400Label, loader=MyLoaders.tensor_hog_loader)
testloader = DataLoader(
    testsetHog, batch_size=test_batch_size, shuffle=test_shuffle)

iter_data = iter(trainloader)
images, labels = next(iter_data)
hog_dim = images.size()[1]

# ----  log--------------------
log = dict()
it_num = 0
running_loss = 0.0

print("Training Start...\n")
log["hog_dim"] = hog_dim
net = model.MLP(hog_dim, 512, num_cls).cuda()
print(net)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


sumEpoch = 0
accuracy_history = dict()
loss_history = []

# ------------ Run ------------------
timeMemo.start()

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
        it_num += 1

        if iteration % 20 == 0:
            loss_str = "[Epoch: "+str(sumEpoch)+"]"" --- Iteration: " +\
                str(iteration+1)+", Loss: "+str(running_loss/it_num)+'.'
            loss_history.append(loss_str)
            if outputScreen:
                print(loss_str)

    acc, true_label, pred_label = evaluate.evaluate_model(
        net, testloader)
    print("At epoch:"+str(sumEpoch)+" accuracy: "+"%.4f" % acc)
    #  Save accuracy
    accuracy_history[sumEpoch]=acc

log["costTime"] = timeMemo.getTimeCostStr()
print("--------------------Training Finish---------------------------")
torch.save(net, savePath)
log["accHis"] = accuracy_history
log["lossHis"] = loss_history

yaml.saveyaml(config, logpath)
yaml.saveyaml({"Log":log}, logpath)

# Draw a graph

plt.xlabel("Training Epoch")
plt.ylabel("Accuracy")

plt.plot(range(1, epochs+1), accuracy_history)
maxacc, maxaccepoch = torch.max(torch.tensor(accuracy_history), 0)
maxaccepoch = maxaccepoch.item()+1

coord = (maxaccepoch-10, maxacc.item()+0.01)
plt.grid()

plt.annotate("%.3f" % maxacc.item() +
             " : "+str(maxaccepoch), coord, xytext=coord)
if outputScreen:
    plt.show()
else:
    plt.savefig(accGraphSavePath)
