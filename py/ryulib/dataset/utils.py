import pathlib
import os.path

def getImagesPath(dirpath):
    """ 
    Input dirpath and return list of files path
     """
    if os.path.isdir(dirpath):
        path = pathlib.Path(dirpath)
        path_list = list(path.iterdir())
        return path_list

def kanji2unicode(kanjilist):
    code=[]
    for kanji in kanjilist:
        code.append(int(kanji.encode('unicode-escape').decode()[2:], 16))
    return code

if __name__ == '__main__':
    JPSC1400Path = "/home/eugene/workspace/dataset/JPSC1400-20201218/label.txt"
    RJPSC1400Path = "/home/eugene/workspace/dataset/JPSC1400-20201218/labelryu.txt"
    dirpath = "/home/eugene/workspace/dataset/fontImage"
    if os.path.isdir(dirpath):
        path = pathlib.Path(dirpath)
        path_list = path.iterdir()
        print(path_list)

        import pandas as pd