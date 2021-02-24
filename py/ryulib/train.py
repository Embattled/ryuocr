import torch.nn as nn
from skimage import feature

def train(net,trainloader,epochs,optimizer,loss_func):

    meanloss_history = []

    ct_num = 0
    running_loss = 0.0

    for epoch in range(epochs):

        sumEpoch += epoch
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
            # if iteration == 0:
            #print("Iteration: "+str(iteration+1)+", Loss: "+str(running_loss/ct_num)+'.')
            # print("[Epoch: "+str(epoch+1)+"]"" --- Iteration: " +
            #     str(iteration+1)+", Loss: "+str(running_loss/ct_num)+'.')


        meanloss = running_loss/ct_num
        meanloss_history.append(meanloss)

        print("[Epoch:"+str(sumEpoch)+"]--MeanLoss:" +
              str(meanloss)+"  TrueLoss:"+str(loss.item()))


