# package ryuutils




## ryudataset

`def getDataset(pathOfLabelFile):`  
输入一个label文件的路径, 返回数据的路径和 label.  

label 文件有两行, 没有标题行, 第一列是数据的相对路径, 第二列是 label.  
返回的是数据的真实路径和 label.  


## sscd

包含了创建 sscd 的相关函数, 基于 pillow.  


### font


`def multi_ttfdictget(fonts:list, dicts:dict, size=64, padding=2,background=0, fill=255)`  
传入 ttf 路径列表, 传入 dicts, 返回获取到的字体图像和对应的数字label.  
字体图像的类型是 list(pillow.images)

### transform

提供了图像的类型转换和面向整个图像集的 transform.  
* `def pilSet2Numpy(images)`
* `def numpySet2PilSet(npimages)`  