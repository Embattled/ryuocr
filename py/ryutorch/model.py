
# Import
from sys import int_info
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
        self.optim_sched = None

        self.loss_func = None
        self.loss_history = []
        self.process = None
        self.total_epochs = 0

        # dataset
        self.sscdSet = None
        self.validSet = None
        self.highestValidAcc = 0.0
        self.highestValidAccStr = None

    def _getLabel(self, prob):
        return numpy.argmax(prob, axis=1)

    def _getCharLabel(self, num_label):
        return numpy.array([self.num_char_dict[i] for i in num_label])

    def setCharDict(self, num_char_dict: dict):
        self.num_char_dict = num_char_dict

    def setSSCD(self, sscd):
        self.sscdSet = sscd
        self.setCharDict(self.sscdSet.getLabelNum2CharDict())

    def setValidSet(self, validset):
        self.validSet = validset

    # Set Network Parts
    def setNetwork(self, **kwagrs):
        self.net, self.size = deepnet.getNetwork(**kwagrs)

    def setLoss(self, **kwargs):
        self.loss_func = deepnet.getLoss(**kwargs)

    def _train(self,  trainLoader, epochs: int, it_step=50, print_log=False, valid=False):

        if valid == True and self.validSet == None:
            raise ValueError("Valid set is None.")

        it_num = 0
        running_loss = 0

        loss_str_ori = "[Epoch: {}] --- Iteration: {}, Loss: {}."
        for epoch in range(epochs):
            self.total_epochs += 1

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
                        self.total_epochs, it_num, running_loss / it_num)
                    self.loss_history.append(loss_str)
                    if print_log:
                        print(loss_str)
                    if valid:
                        acc = self.validation()
                        print("Accuracy on validation set {}".format(acc))
                        if acc > self.highestValidAcc:
                            self.highestValidAccStr = "Best : [Epoch: {}] --- Iteration: {}, Acc: {}.".format(
                                self.total_epochs, it_num, acc)
                            self.highestValidAcc = acc
            if valid:
                acc = self.validation()
                print("Accuracy on validation set after {} epochs :{}".format(
                    self.total_epochs, acc))
                if acc > self.highestValidAcc:
                    self.highestValidAccStr = "Best : [Epoch: {}] --- Acc: {}.".format(
                        self.total_epochs, acc)
                    self.highestValidAcc = acc

            if self.optim_sched != None:
                self.optim_sched.step()
                print("Update learning rate, now  %1.10f" %
                      (self.optim_sched.get_last_lr()[0]))
        self.optimizer.zero_grad()

    def _startTrain(self, trainData, trainLabel,
                    epoch,
                    batchsize,
                    shuffle,
                    print,
                    iter_step,
                    valid=False):

        trainData = transform.pil2Tensor(trainData)
        trainLabel = torch.tensor(trainLabel)

        loader = dataset.loader.getLoader(self.size, is_path=False)
        trainset = dataset.baseset.RyuImageset(
            trainData, trainLabel, loader=loader)
        trainLoader = DataLoader(
            trainset, batch_size=batchsize, shuffle=shuffle)

        self._train(trainLoader, epoch,
                    it_step=iter_step, print_log=print, valid=valid)


    def train(self):
        if self.process == None:
            raise ValueError("Emply train process")

        print("Start training.")
        num_process = 0
        for process in self.process:
            num_process += 1
            print("Start training process {}".format(num_process))

            self.optimizer, self.optim_sched = deepnet.getOptimizer(
                self.net, **process["optimizer"])

            for _ in range(process["loop"]):

                trainData, trainLabel = self.sscdSet.getTrainData(
                    **process["sscd"])
                self._startTrain(trainData, trainLabel, **process["para"])
        

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

        if not is_path:
            testData = transform.pil2Tensor(testData)
        loader = dataset.loader.getLoader(self.size, is_path=is_path)
        if testLabel == None:
            testLabel = torch.zeros(len(testData))
        testSet = dataset.baseset.RyuImageset(testData, testLabel, loader)
        testLoader = DataLoader(testSet, 64, shuffle=False)

        res = self._inference(testLoader)
        if not num_label:
            res = self._getCharLabel(res)
        return res

    def validation(self):
        res = self.inference(
            self.validSet.dataPath, num_label=False, is_path=True)
        acc = self.validSet.evaluate(res)
        return acc

    def save(save_path):
        pass


def getTorchModel(args: dict, num_classes):
    model = TorchModel()
    model.setNetwork(**args["architecture"], num_classes=num_classes)
    if args["gpu"]:
        model.net.cuda()
    print(model.net)
    model.process = args["process"]
    model.setLoss(**args["loss"])
    return model
