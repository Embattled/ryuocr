import pandas as pd 
import os.path


def getDataset(pathOfLabelFile):
    
    data=pd.read_csv(pathOfLabelFile,sep=',',index_col=None,header=None,names=["path","label"])
    data["path"]=os.path.dirname(pathOfLabelFile)+"/"+data["path"]
    
    return list(data["path"].values),list(data["label"])

if __name__ =="__main__":
    jpsc="/home/eugene/workspace/dataset/scenetext/JPSC1400-20201218/rec_gt_test.txt"
    a,b=getDataset(jpsc)

    print(a)
    print(b)



