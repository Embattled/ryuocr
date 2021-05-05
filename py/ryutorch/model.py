import torch
import torch.nn as nn

def getNetwork(para:dict,cls_num:int,input_dim=None):
    try:
        name=para["name"]

        if name=="MLP":
            hid_l=para["hidden_layer"]
            if input_dim!=None:
                model=MLP(input_dim,hid_l,cls_num)
            else:
                raise ValueError("Illegal input dimension.")
        else:
            raise ValueError("Undefined model name.")
        return model
    except:
        raise ValueError("Illegal model parameter.")
    

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class):
        super(MLP, self).__init__()

        self.classifier=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim,num_class),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)
