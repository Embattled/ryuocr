import pathlib
import os.path
import pandas as pd

def readDict(path):
    """ 
    Read a dict file, no column name and index.
    """
    data=pd.read_table(path,quoting=3,sep=" ",squeeze=True,header=None,index_col=False)
    return list(data)

if __name__ == '__main__':
    a=readDict("/home/eugene/workspace/dataset/font/dict3107jp.txt")
    print(a)