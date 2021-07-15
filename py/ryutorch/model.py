
# Import
import torch
from torch.utils.data import DataLoader, dataloader

from . import dataset, transform, deepnet

import numpy


class TorchModel:
    def __init__(self):

        # model
        self.net = None
        self.size = None
        self.optimizer = None
        self.loss_func = None

        self.loss_history = []

    def _getLabel(self, prob):
        return numpy.argmax(prob, axis=1)

    def _getCharLabel(self, num_label):
        return numpy.array([self.num_char_dict[i] for i in num_label])

    def setCharDict(self, num_char_dict: dict):
        self.num_char_dict = num_char_dict

    # Set Network Parts
    def setNetwork(self, **kwagrs):
        self.net, self.size = deepnet.getNetwork(**kwagrs)

    def setOptimizer(self, **kwargs):
        self.optimizer = deepnet.getOptimizer(self.net, **kwargs)

    def setLoss(self, **kwargs):
        self.loss_func = deepnet.getLoss(**kwargs)

    def setProcess(self, epoch: int, batchsize: int, shuffle: bool, print=False, iter_step=50):
        self.epoch = epoch
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.print = print
        self.iter_step = iter_step

    def _train(self,  trainLoader, epochs: int, it_step=50, print_log=False):

        epoch_num = 0
        it_num = 0
        running_loss = 0

        loss_str_ori = "[Epoch: {}] --- Iteration: {}, Loss: {}."
        for epoch in range(epochs):
            epoch_num += 1
            for _, data in enumerate(trainLoader):

                # Take the inputs and the labels for 1 batch.
                inputs, labels = data

                # Move inputs and labels into GPU
                inputs = inputs.cuda()
                labels = labels.cuda()

                # Remove old gradients for the optimizer.
                self.optimizer.zero_grad()

                # Compute result (Forward)
                outputs = self.net(inputs)

                # Compute loss
                loss = self.loss_func(outputs, labels)

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                it_num += 1

                if (it_num) % it_step == 0:
                    loss_str = loss_str_ori.format(
                        epoch_num, it_num, running_loss / it_num)
                    self.loss_history.append(loss_str)
                    if print_log:
                        print(loss_str)

    def train(self, trainData, trainLabel):
        trainData = transform.pil2Tensor(trainData)
        trainLabel = torch.tensor(trainLabel)
        
        loader = dataset.loader.getLoader(self.size, is_path=False)
        trainset = dataset.baseset.RyuImageset(
            trainData, trainLabel, loader=loader)
        trainLoader = DataLoader(
            trainset, batch_size=self.batchsize, shuffle=self.shuffle)

        self._train(trainLoader, self.epoch,
                    it_step=self.iter_step, print_log=self.print)

    def _inference(self, testLoader, proba=False):
        self.net.eval()
        predict = []
        gt = []

        for _, test_data in enumerate(testLoader):

            inputs, labels = test_data
            bch = inputs.size(0)

            # Move inputs and labels into GPU
            inputs = inputs.cuda()
            labels = labels.cuda()

            # Forward
            outputs = self.net(inputs)

            # Get predicted classes
            confidence, pred_cls = torch.max(outputs, 1)

            gt.append(labels)
            predict.append(pred_cls)

        self.net.train()
        predict = torch.cat(predict).cpu()
        gt = torch.cat(gt).cpu()

        return predict.numpy()

    def inference(self, testData, testLabel=None, num_label=False, is_path=True):

        loader = dataset.loader.getLoader(self.size, is_path=is_path)
        if testLabel == None:
            testLabel = torch.zeros(len(testData))
        testSet = dataset.baseset.RyuImageset(testData, testLabel, loader)
        testLoader = DataLoader(testSet, self.batchsize, shuffle=False)

        res = self._inference(testLoader)
        if not num_label:
            res = self._getCharLabel(res)
        return res

    def save(save_path):
        pass


def getTorchModel(args: dict, num_cls):
    model = TorchModel()
    model.setNetwork(**args["architecture"], num_cls=num_cls)
    print(model.net)
    if args["gpu"]:
        model.net.cuda()
    model.setOptimizer(**args["optimizer"])
    model.setProcess(**args["process"])
    model.setLoss(**args["loss"])
    return model
