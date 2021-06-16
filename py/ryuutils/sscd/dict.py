import pathlib
import os.path
import pandas as pd

def readDict(path):
    """ 
    Read a dict file, no column name and index.
    """
    data=pd.read_table(path,quoting=3,sep=" ",squeeze=True,header=None,index_col=False)

    list_data= list(data)
    label_dict= dict(zip(list_data, range(len(list_data))))
    return list_data,label_dict

def getNumberLabel(label_dict,label_str):
    return [label_dict[i] for i in label_str]

if __name__ == '__main__':
    a=readDict("/home/eugene/workspace/dataset/font/dict3107jp.txt")
    print(a)