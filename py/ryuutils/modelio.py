
import pickle
import os.path

def saveModel(model:object,path:str,name:str):
    path=os.path.abspath(path)
    path=os.path.join(path,name+".pickle")

    with open(path,'wb') as f:
        pickle.dump(model,f)

def loadModel(path:str,name:str):
    path=os.path.abspath(path)
    path=os.path.join(path,name+".pickle")

    with open(path,'rb') as f:
        model = pickle.load(f)
    return model



