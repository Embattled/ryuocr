from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import torch
import torch.nn as nn
import torch.tensor
import torch.optim as optim

from PIL import Image
import pathlib
import matplotlib.pyplot as plt

import sys
import ryulib


fontImageDirPath = "/home/eugene/workspace/dataset/JPSC1400font/png"
fontImagesPath = ryulib.dataset.getImagesPath(fontImageDirPath)

fontlabels = []
for path in fontImagesPath:
    fontlabels.append(pathlib.Path(path).stem)


num_class = len(fontlabels)
code2label = zip(fontlabels, range(num_class))
trainset = ryulib.dataset.FontTrainSet(fontImagesPath, range(num_class),size=16)

trainloader = DataLoader(trainset, batch_size=32, shuffle=True)


# JPSC=ryulib.dataset.sscd.JPSC1400()
# testloader=DataLoader(JPSC,batch_size=50,shuffle=False)
# testloader = DataLoader(trainset, batch_size=100, shuffle=False)


it=iter(trainloader)
images,label=next(it)

print(images[0])
# show_imgs = utils.make_grid(images, nrow=10,pad_value=255).numpy().transpose((1, 2, 0))
# print(type(show_imgs))
# plt.imshow(show_imgs)
# plt.show()

sys.exit(0)

# net = models.AlexNet(num_class).cuda()
net = ryulib.model.MLP(64*64*3, 512, num_class).cuda()


print(net)


def evaluate_model():
    print("Testing the network...")
    net.eval()
    total_num = 0
    correct_num = 0
    for test_iter, test_data in enumerate(testloader):
        # Get one batch of test samples
        inputs, labels = test_data
        bch = inputs.size(0)

        # inputs = inputs.view(bch, -1)

        # Move inputs and labels into GPU
        inputs = inputs.cuda()
        labels = labels.cuda()

        # Forward
        outputs = net(inputs)

        # Get predicted classes
        pred_cls1, pred_cls = torch.max(outputs, 1)

#     if total_num == 0:
#        print("True label:\n", labels)
#        print("Prediction:\n", pred_cls)
        # Record test result
        print("Label:", labels)
        print("Pred_cls:", outputs[0])
        print("Pred_cls:", pred_cls1)
        print("Pred_cls:", pred_cls)

        correct_num += (pred_cls == labels).float().sum().item()
        total_num += bch
    net.train()
    print("Accuracy: "+"%.3f" % (correct_num/float(total_num)))


# ------ We define the loss function and the optimizer -------
loss_func = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.5, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.01)
epochs = 40

ct_num = 0
running_loss = 0.0

for epoch in range(epochs):
    for iteration, data in enumerate(trainloader):
        # Take the inputs and the labels for 1 batch.
        inputs, labels = data
        bch = inputs.size(0)
        # inputs = inputs.view(bch, -1) <-- We don't need to reshape inputs here (we are using CNNs).

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

    print("[Epoch:"+str(epoch+1)+"]---Loss:"+str(running_loss/ct_num))
evaluate_model()
