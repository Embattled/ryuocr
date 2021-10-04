from os import listxattr
import pathlib
import os.path
import pandas as pd


def reviseJapanDict(d: list):
    ambigList = [
        ['へ', 'ヘ'],
        ['べ', 'ベ'],
        ['ぺ', 'ペ'],

        ['エ', '工'],
        ['カ', '力'],
        ['ロ', '口'],
        ['タ', '夕'],
        ['ニ', '二'],
        ['ハ', '八'],

        ['オ', '才'],
    ]

    for i in range(len(d)):
        for am in ambigList:
            if d[i] in am:
                d[i] = am[0]


def readDict(dictpath, revise=False):
    """ 
    Read a dict file, no column name and index.
    """
    if isinstance(dictpath, str):
        data = pd.read_table(dictpath, quoting=3, sep=" ",
                             squeeze=True, header=None, index_col=False)
        if revise:
            reviseJapanDict(data)
        list_data = data.drop_duplicates()
    elif isinstance(dictpath, list):
        list_data=dictpath
    else:
        raise ValueError("Invalid dict path")

    # dict is  [ char : number ]
    char_dict = dict(zip(list_data, range(len(list_data))))
    num_dict = dict(zip(range(len(list_data)), list_data))

    return len(list_data), num_dict, char_dict


if __name__ == '__main__':
    a = readDict("/home/eugene/workspace/dataset/font/dict3107jp.txt")
    print(a)
