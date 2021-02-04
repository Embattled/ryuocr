import torch
import torch.nn as nn
import torch.tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models

from PIL import Image
import pathlib
import matplotlib.pyplot as plt

from skimage import feature
from skimage import transform as sktr
from skimage import io as skio
import numpy

import sys
import ryulib



sys.exit(0)

fontCode = []
for path in fontImagesPath:
    fontCode.append(pathlib.Path(path).stem)


num_class = len(fontCode)
code2label = dict(zip(fontCode, range(num_class)))


# define hog loader
# def hog_loader(path):
#     img_ski = skio.imread(path)
#     img_ski = sktr.resize(img_ski, (64, 64))
#     img_ski = feature.hog(img_ski).astype(numpy.float32)
#     img_tensor = torch.from_numpy(img_ski)
#     return img_tensor


trainset = ryulib.dataset.FontTrainSet(
    fontImagesPath, range(num_class))
# data set


trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
iter_data = iter(trainloader)
images, labels = next(iter_data)

show_imgs = utils.make_grid(
    images, nrow=10).numpy().transpose((1, 2, 0))
plt.imshow(show_imgs)
plt.show()


sys.exit(0)


# Train
net = ryulib.model.MLP(fea_len, 512, num_class).cuda()
print(net)


# ------ We define the loss function and the optimizer -------
loss_func = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.0001)
epochs = 300

ct_num = 0
running_loss = 0.0

for epoch in range(epochs):
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
        #print("Iteration: "+str(iteration+1)+", Loss: "+str(running_loss/ct_num)+'.')
        # print("[Epoch: "+str(epoch+1)+"]"" --- Iteration: " +
        #     str(iteration+1)+", Loss: "+str(running_loss/ct_num)+'.')

    print("[Epoch:"+str(epoch+1)+"]---TrueLoss:"+str(running_loss/ct_num))
    print("[Epoch:"+str(epoch+1)+"]---MeanLoss:"+str(running_loss/ct_num))
    # if(epoch % 100 == 99):
    #     ryulib.evaluate.evaluate_model(net, testloader, False)

savePath = '/home/eugene/workspace/ryuocr/py/trained/lastTrained.pt'
torch.save(net, savePath)


unicodeJPSC = ryulib.dataset.sscd.getCodeJPSC()
labelJPSC = []
for code in unicodeJPSC:
    labelJPSC.append(code2label[str(code)])

JPSC1400 = ryulib.dataset.sscd.JPSC1400(labelJPSC, loader=hog_loader)
testloader = DataLoader(JPSC1400, batch_size=128, shuffle=False)

acc, true_label, pred_label = ryulib.evaluate.evaluate_model(
    net, testloader, False)
print("Accuracy: "+"%.3f" % acc)
