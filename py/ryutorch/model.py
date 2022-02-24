# Import
import os.path
import copy


import torch
from torch.utils.data import DataLoader, dataloader

from . import dataset, transform, deepnet

import numpy


class TorchModel:
    def __init__(self):

        # model
        self.net = None
        self.netBestState = None

        self.size = None
        self.optimizer = None
        self.optim_sched = None

        self.loss_func = None
        self.process = None

        # history
        self.total_epochs = 0
        self.it_num = 0
        self.running_loss = 0

        self.loss_history = []
        self.validacc_history = []

        self.highestValidAcc = -1.0
        self.highestValidAccStr = None

        # dataset
        self.sscdSet = None
        self.validSet = None

        # dict
        self.num_char_dict = None

    def _getLabel(self, prob):
        return numpy.argmax(prob, axis=1)

    def _getCharLabel(self, num_label):
        if len(num_label.shape) == 1:
            return numpy.array([self.num_char_dict[i] for i in num_label])
        elif len(num_label.shape) == 2:
            return numpy.array([[self.num_char_dict[i] for i in row] for row in num_label])
        else:
            raise ValueError(
                "num_label has illegal shape {}".format(num_label.shape))

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

    def _train(self,  trainLoader, epochs: int or [str, float], it_step=50, print_log=False, valid=False):

        # topk
        topk = 2

        loss_str_ori = "[Epoch: {}] --- Iteration: {}, Loss: {}."

        def iteration_once():
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

                self.running_loss += loss.item()
                self.it_num += 1

                if (self.it_num) % it_step == 0:
                    loss_batch = self.running_loss / self.it_num
                    loss_str = loss_str_ori.format(
                        self.total_epochs, self.it_num, loss_batch)
                    self.loss_history.append(loss_batch)
                    if print_log:
                        print(loss_str)
                    if valid:
                        acctopk, acc = self.validation(topk=topk)
                        self.validacc_history.append(acc)

                        print(
                            "Accuracy on validation set, Top-1:{} , Top-{}: {}".format(acc, topk, acctopk))

                        if acc > self.highestValidAcc:
                            self.highestValidAccStr = "Best : [Epoch: {}] --- Iteration: {}, Acc: {}.".format(
                                self.total_epochs, self.it_num, acc)
                            self.highestValidAcc = acc
                            self.netBestState = copy.deepcopy(
                                self.net.state_dict())
            if valid:
                acctopk, acc = self.validation(topk=topk)
                print(
                    "Accuracy on validation set, Top-1:{} , Top-{}: {}".format(acc, topk, acctopk))
                if acc > self.highestValidAcc:
                    self.highestValidAccStr = "Best : [Epoch: {}] --- Acc: {}.".format(
                        self.total_epochs, acc)
                    self.highestValidAcc = acc
                    self.netBestState = copy.deepcopy(self.net.state_dict())

            if self.optim_sched != None:
                self.optim_sched.step()
                print("Update learning rate, now  %1.10f" %
                      (self.optim_sched.get_last_lr()[0]))

        # Judge
        if valid == True and self.validSet == None:
            raise ValueError("Valid set is None.")
        if isinstance(epochs, int):
            for epoch in range(epochs):
                iteration_once()

        elif isinstance(epochs, list):
            if epochs[0] == "loss" and isinstance(epochs[1], float):
                if self.it_num == 0:
                    iteration_once()
                while self.running_loss/self.it_num > epochs[1]:
                    iteration_once()
        else:
            raise ValueError("Illegal parameter epochs '{}'".format(epochs))

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

    def _inference(self, testLoader, topk: int = 10, proba=False):
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
            # confidence, pred_cls = torch.max(outputs, 1)
            confidence, pred_cls = torch.topk(outputs, topk)

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

    def validation(self, topk=1):
        res = self.inference(
            self.validSet.dataPath, num_label=False, is_path=True)
        acck, acc1 = self.validSet.evaluate(res, topk)
        return acck, acc1

    # IO
    def saveModel(self, save_path, name="model.pt"):
        torch.save(self.net, os.path.join(save_path, name))

    def save(self, save_path, log=False):
        self.saveModel(save_path)

        if self.netBestState != None:
            self.net.load_state_dict(self.netBestState)
            self.saveModel(save_path, "model_best.pt")

        numpy.save(os.path.join(save_path, "dict.npy"), self.num_char_dict)
        numpy.savetxt(os.path.join(save_path, "size.txt"), self.size,fmt="%d")

        if log == True:
            numpy.savetxt(os.path.join(
                save_path, "loss_his.txt"), self.loss_history)
            numpy.savetxt(os.path.join(
                save_path, "validacc_his.txt"), self.validacc_history)

    def load(self, load_path, best: bool = False, **kwargs):
        del self.size
        del self.net
        del self.num_char_dict

        if best:
            self.net = torch.load(os.path.join(load_path, "model_best.pt"))
        else:
            self.net = torch.load(os.path.join(load_path, "model.pt"))

        self.num_char_dict = numpy.load(os.path.join(
            load_path, "dict.npy"), allow_pickle=True).tolist()

        self.size = tuple(numpy.loadtxt(
            os.path.join(load_path, "size.txt"), dtype=int))


def getTorchModel(args: dict, num_classes):
    model = TorchModel()
    model.setNetwork(**args["architecture"], num_classes=num_classes)
    if args["gpu"]:
        model.net.cuda()
    print(model.net)
    model.process = args["process"]
    model.setLoss(**args["loss"])
    return model
