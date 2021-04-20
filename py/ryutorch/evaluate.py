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


def evaluate_ensemble(nets, testloader, num_cls):
    net_num = len(nets)
    print("Testing the "+str(net_num)+" ensembled networks...")

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

        voter = torch.zeros((bch, num_cls), dtype=torch.float32).cuda()

        for net in nets:
            net.eval()

            # Forward
            outputs = net(inputs)

            voter += outputs

            net.train()

        # Get predicted classes
        confidence, pred_cls = torch.max(voter, 1)

        groundT.append(labels)
        predLabel.append(pred_cls)

        correct_num += (pred_cls == labels).float().sum().item()
        total_num += bch

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
