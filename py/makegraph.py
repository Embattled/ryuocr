from matplotlib import pyplot as plt
import torch




accuracy_history = torch.tensor([
    0.0150,
    0.0350,
    0.0893,
    0.1543,
    0.2779,
    0.3679,
    0.3814,
    0.3964,
    0.3943,
    0.4086,
    0.4071,
    0.4036,
    0.4136,
    0.4071,
    0.4107,
    0.3971,
    0.4179,
    0.4236,
    0.4136,
    0.4107,
    0.4129,
    0.4121,
    0.4293,
    0.4264,
    0.4343,
    0.4214,
    0.4464,
    0.4329,
    0.4414,
    0.4450,
    0.4321,
    0.4550,
    0.4514,
    0.4464,
    0.4464,
    0.4586,
    0.4493,
    0.4764,
    0.4721,
    0.4607,
    0.4686,
    0.4800,
    0.4821,
    0.4750,
    0.4757,
    0.4721,
    0.4614,
    0.4686,
    0.4764,
    0.4700,
    0.4871,
    0.4779,
    0.4850,
    0.4707,
    0.4721,
    0.4850,
    0.4879,
    0.5021,
    0.4929,
    0.5000,
    0.4914,
    0.4864,
    0.4850,
    0.4850,
    0.4879,
    0.4957,
    0.5043,
    0.5157,
    0.5086,
    0.5014,
    0.5064,
    0.5200,
    0.5164,
    0.5157,
    0.5050,
    0.5086,
    0.5293,
    0.5171,
    0.5236,
    0.5193,
    0.5307,
    0.5250,
    0.5264,
    0.5150,
    0.5343,
    0.5207,
    0.5200,
    0.5171,
    0.5129,
    0.5107,
    0.4907,
    0.5179,
    0.5121,
    0.5079,
    0.5307,
    0.5129,
    0.5157,
    0.5221,
    0.5229,
    0.5307,
    0.5371,
    0.5257,
    0.5079,
    0.5171,
    0.5157,
    0.5143,
    0.5050,
    0.5214,
    0.5229,
    0.5371,
    0.5207,
    0.5171,
    0.5264,
    0.5307,
    0.5321,
    0.5236,
    0.5379,
    0.5214,
    0.5350,
    0.5271,
    0.5414,
    0.5400,
    0.5329,
    0.5471,
    0.5307,
    0.5307,
    0.5421,
    0.5321,
    0.5393,
    0.5400,
    0.5350,
    0.5421,
    0.5364,
    0.5564,
    0.5471,
    0.5457,
    0.5500,
    0.5471,
    0.5400,
    0.5464,
    0.5464,
    0.5600,
    0.5636,
    0.5400,
    0.5407,
    0.5321,
    0.5471,
    0.5379,
    0.5550,
    0.5486,
    0.5500,
    0.5371,
    0.5364,
    0.5400,
    0.5379,
    0.5443,
    0.5414,
    0.5486,
    0.5357,
    0.5393,
    0.5529,
    0.5321,
    0.5429,
    0.5586,
    0.5536,
    0.5571,
    0.5521,
    0.5500,
    0.5400,
    0.5693,
    0.5586,
    0.5629,
    0.5507,
    0.5493,
    0.5471,
    0.5450,
    0.5507,
    0.5536,
    0.5586,
    0.5629,
    0.5593,
    0.5529,
    0.5564,
    0.5514,
    0.5636,
    0.5607,
    0.5664,
    0.5650,
    0.5586,
    0.5493,
    0.5507,
    0.5550,
    0.5579,
    0.5543,
    0.5543,
    0.5564,
    0.5550,
    0.5536,
    0.5629,
    0.5657])


x = range(1, accuracy_history.size()[0]+1)

plt.xlabel("Training Epoch")
plt.ylabel("Accuracy")
plt.grid()

maxacc, maxaccepoch = torch.max(accuracy_history, 0)

maxaccepoch=maxaccepoch.item()+1
# print(maxaccepoch)
coord = (maxaccepoch-150, maxacc.item())
plt.grid()

plt.annotate("Max accuracy: "+"%.4f" % maxacc.item() +
             " epoch: "+str(maxaccepoch), coord, xytext=coord)


plt.plot(x, accuracy_history)
plt.show()
