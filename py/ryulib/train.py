import torch.nn as nn
from skimage import feature

def train(net,epochs,optimizer,loss_func=nn.CrossEntropyLoss(),):
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    epochs = 5000

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

        print("[Epoch:"+str(epoch+1)+"]---Loss:"+str(running_loss/ct_num))
        if(epoch % 100 == 0):
            evaluate_model()