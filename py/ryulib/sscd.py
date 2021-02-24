import torch


def sscdCreate(originData,transfunction,epoch=1):
    sscd = []

    for _ in range(epoch):

        trainData = originData.clone().detach()
        transfunction(trainData, originData)
        sscd.append(trainData)

    return torch.cat(sscd)
