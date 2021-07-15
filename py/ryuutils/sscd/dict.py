import pathlib
import os.path
import pandas as pd


def reviseJapanDict(d: list):
    he = ['へ', 'ヘ']
    be = ['べ', 'ベ']
    pe = ['ぺ', 'ペ']

    ambigList = [he, be, pe]
    for i in range(len(d)):
        for am in ambigList:
            if d[i] in am:
                d[i] = am[0]


def readDict(path):
    """ 
    Read a dict file, no column name and index.
    """
    data = pd.read_table(path, quoting=3, sep=" ",
                         squeeze=True, header=None, index_col=False)

    reviseJapanDict(data)

    list_data = list(set(data))
    # dict is  [ char : number ]
    char_dict = dict(zip(list_data, range(len(list_data))))
    num_dict = dict(zip(range(len(list_data)), list_data))

    return len(list_data), num_dict, char_dict


if __name__ == '__main__':
    a = readDict("/home/eugene/workspace/dataset/font/dict3107jp.txt")
    print(a)
