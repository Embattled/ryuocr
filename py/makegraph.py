from matplotlib import pyplot as plt
import torch


accuracy_history = torch.tensor([
    0.4250,
    0.4443,
    0.4500,
    0.4529,
    0.4521,
    0.4571,
    0.4579,
    0.4579,
    0.4579,
    0.4593,
    0.4600,
    0.4593,
    0.4579,
    0.4564,
    0.4586
])

x = range(1, accuracy_history.size()[0]+1)

plt.xlabel("Number of Models")
plt.ylabel("Accuracy")

# maxacc, maxaccepoch = torch.max(accuracy_history, 0)

# maxaccepoch = maxaccepoch.item()+1
# print(maxaccepoch)
# coord = (maxaccepoch-150, maxacc.item())
plt.grid()

# plt.annotate("Max accuracy: "+"%.4f" % maxacc.item() +
            #  " epoch: "+str(maxaccepoch), coord, xytext=coord)


plt.plot(x, accuracy_history)
plt.savefig("Result.png")
# plt.show()
