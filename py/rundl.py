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
from ryuutils import ryuyaml as yaml
from ryuutils import sscd 
from ryuutils import ryutime
from ryuutils import example 
from ryuutils import ryudataset

import ryutorch
import ryutorch.transform as rtt
from ryutorch import model,evaluate
from ryutorch import example as torchex

import ryutorch.dataset
from ryutorch.dataset.baseset import RyuImageset
from ryutorch.dataset.loader import RyuLoader

# ------------program global mode -----------
timeMemo = ryutime.TimeMemo()
nowTimeStr = timeMemo.nowTimeStr()

# Set print goal
# outputScreen = False
outputScreen = True

# ----------- training parameter ---------
configPath = "/home/eugene/workspace/ryuocr/py/config/torchconfig.yml"

config = yaml.loadyaml(configPath)

# ---------- global parameter -----
paramGlobal = config["Global"]

# Network model save path
saveModelPath = paramGlobal["save_model_dir"] + nowTimeStr+'.pt'
# Accuracy History Graph Save Path
accGraphSavePath = paramGlobal["save_result_dir"] + nowTimeStr+"_acc.png"
logPath = paramGlobal["save_result_dir"] + nowTimeStr+".yml"


if not outputScreen:
    outputfilename = paramGlobal["save_result_dir"] + nowTimeStr+"_log.txt"
    pfile = open(outputfilename, mode='w')
    sys.stdout = pfile

epochs = paramGlobal["epoch"]
print_iter_step = paramGlobal["print_iter_step"]

# ----------  Optimizer ----
paramOpt = config["Optimizer"]

learning_rate = paramOpt["learning_rate"]
loss_func = nn.CrossEntropyLoss()

# ------------ Dataset -------------

# Train dataset parameter
paramTrain = config["Train"]
train_batch_size = paramTrain["loader"]["batch_size"]
train_shuffle = paramTrain["loader"]["shuffle"]

# Font data parameter
ttfpath=paramTrain["dataset"]["ttfpath"]
dictpath=paramTrain["dataset"]["dict"]
fontsize=paramTrain["dataset"]["fontsize"]

# Read Font Data
char_list,char_dict=sscd.dict.readDict(dictpath)
num_cls=len(char_list)
config["Train"]["dataset"]["dict_len"]=num_cls

fontData,fontLabel=sscd.font.multi_ttfdictget(ttfpath,char_dict,size=fontsize)

# Create sscd
timeMemo.reset()

def transformFunction(originData):
    transData=sscd.transform.uniformPerspective(originData,scale=0.5,p=0.7,inplace=False)
    transData=sscd.transform.randomColorizeSet(transData,inplace=True)
    return transData
# ryuutils.example.showExamplePIL(transData,8,4,shuffle=True)
print(getsource(transformFunction))
# sscdData,sscdLabel=sscd.transform.sscdCreate(fontData,fontLabel,transformFunction,epoch=20)
sscdData=transformFunction(fontData)
trainData=ryutorch.transform.pil2Tensor(sscdData)
trainLabel = torch.tensor(fontLabel)
# trainLabel = torch.tensor(sscdLabel)
print("Create sscd cost: "+timeMemo.getTimeCostStr())
print(trainData.size())
print(trainLabel.size())

# Test dataset
paramTest = config["Test"]
test_batch_size = paramTest["loader"]["batch_size"]
test_shuffle = paramTest["loader"]["shuffle"]

testsetName = paramTest["dataset"]["name"]
testsetPath = paramTest["dataset"]["dir"]

testData,testLabelStr=ryudataset.getDataset(testsetPath)
testLabel=torch.tensor(ryuutils.sscd.dict.getNumberLabel(char_dict,testLabelStr))


print("Loding data success.")
# ------------- Preprocess and Dataloader

# Define Transforms
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
trainPreprocess = transforms.Compose([
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    normalize
])
testPreprocess=transforms.Compose([
    transforms.Resize((fontsize,fontsize)),
    normalize
])
TrainLoaders = RyuLoader(trainPreprocess)
TestLoaders = RyuLoader(testPreprocess)

# ---------- Create Dataset and Loader

# Hog Version
trainset = RyuImageset(
    trainData, trainLabel, loader=TrainLoaders.tensor_loader)
trainloader = DataLoader(
    trainset, batch_size=train_batch_size, shuffle=train_shuffle)

testset = RyuImageset(
    testData, testLabel, loader=TestLoaders.file_loader)
testloader = DataLoader(
    testset, batch_size=test_batch_size, shuffle=test_shuffle)
# Deep feature version


iter_data = iter(trainloader)
images, labels = next(iter_data)
hog_dim = images.size()[1]


# ----  log--------------------
log = dict()

it_num = 0
running_loss = 0.0

print("Training Start...\n")

log["hog_dim"] = hog_dim
# net = model.getNetwork(config["Architecture"]["network"],num_cls).cuda()
net = models.AlexNet(num_cls).cuda()
print(net)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

sumEpoch = 0
accuracy_history = dict()
loss_history = []

# ------------ Run ------------------
timeMemo.reset()

for epoch in range(epochs):

    transData=transformFunction(fontData)
    # ryuutils.example.showExamplePIL(transData,8,4,shuffle=True)
    trainData=ryutorch.transform.pil2Tensor(transData)
    trainset.changeData(trainData)
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

        if (iteration+1) % print_iter_step == 0:
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
print("--------------Cost Time :"+timeMemo.getTimeCostStr()+"---------------------------")
torch.save(net, saveModelPath)
log["accHis"] = accuracy_history
log["lossHis"] = loss_history

yaml.saveyaml(config, logPath)
yaml.saveyaml({"Log":log}, logPath)

# Draw a graph

# plt.xlabel("Training Epoch")
# plt.ylabel("Accuracy")

# plt.plot(range(1, epochs+1), accuracy_history)
# maxacc, maxaccepoch = torch.max(torch.tensor(accuracy_history), 0)
# maxaccepoch = maxaccepoch.item()+1

# coord = (maxaccepoch-10, maxacc.item()+0.01)
# plt.grid()

# plt.annotate("%.3f" % maxacc.item() +
#              " : "+str(maxaccepoch), coord, xytext=coord)
# if outputScreen:
#     plt.show()
# else:
#     plt.savefig(accGraphSavePath)

