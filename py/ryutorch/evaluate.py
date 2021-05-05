import torch


def evaluate_model(net, testloader):
    print("Testing the network...")
    net.eval()

    total_num = 0
    correct_num = 0

    groundT = []
    predLabel = []

    for _, test_data in enumerate(testloader):

        inputs, labels = test_data
        bch = inputs.size(0)

        # Move inputs and labels into GPU
        inputs = inputs.cuda()
        labels = labels.cuda()

        # Forward
        outputs = net(inputs)

        # Get predicted classes
        confidence, pred_cls = torch.max(outputs, 1)

        groundT.append(labels)
        predLabel.append(pred_cls)

        correct_num += (pred_cls == labels).float().sum().item()
        total_num += bch

    net.train()
    accuracy = correct_num/float(total_num)

    groundT = torch.cat(groundT)
    predLabel = torch.cat(predLabel)
    return accuracy, groundT, predLabel

'''
    if need_flatten:
        for _, test_data in enumerate(testloader):

            inputs, labels = test_data
            bch = inputs.size(0)

            # Flatten
            inputs = inputs.view(bch, -1)

            # Move inputs and labels into GPU
            inputs = inputs.cuda()
            labels = labels.cuda()

            # Forward
            outputs = net(inputs)

            # Get predicted classes
            confidence, pred_cls = torch.max(outputs, 1)

            groundT.append(labels)
            predLabel.append(pred_cls)

            correct_num += (pred_cls == labels).float().sum().item()
            total_num += bch
'''
