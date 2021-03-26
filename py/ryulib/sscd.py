import torch


def sscdCreate(originData,originLabel,transfunction,epoch=1):

    sscd=[]
    labels=[]
    for _ in range(epoch):

        trainData = originData.clone().detach()
        transfunction(trainData, originData)
        sscd.append(trainData)
        labels.append(originLabel.clone().detach())

    return torch.cat(sscd),torch.cat(labels)
