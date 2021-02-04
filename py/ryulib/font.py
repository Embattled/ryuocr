from PIL import Image, ImageFont, ImageDraw, ImageChops, ImageMorph
import pathlib
import os


def ttfshow(fontpath, label, fontpoint=64):
    """ 
    Return font images of unicode list
    """
    font = ImageFont.truetype(fontpath, fontpoint)

    # image
    image = Image.new('RGB', (fontpoint, fontpoint))
    draw = ImageDraw.Draw(image)

    draw.text((0, 0), chr(label), font=font, anchor="lt")
    image.show()


def ttfimageget(fontpath, labels, size=64, padding=0):
    """ 
    Return font images of unicode list
    """

    image = Image.new('RGB', (size, size))

    fontpoint = size-padding*2

    font = ImageFont.truetype(fontpath, fontpoint)

    # image
    draw = ImageDraw.Draw(image)

    images = []
    # read
    for label in labels:
        draw.text((padding, padding), chr(label), font=font, anchor="lt")
        images.append(image)
    return images


def ttfwrite(fontpath, outputdir, labels, size=64, padding=0):
    """ 
    Write font images of unicode list to files
    """
    fontpoint = size-padding*2

    font = ImageFont.truetype(str(fontpath), fontpoint)
    fontName = pathlib.Path(fontpath).stem
    savePath=outputdir+"/"+fontName
    if not os.path.exists(savePath):
        os.mkdir(savePath)

    xy=(size/2,size/2)

    for label in labels:
        code = chr(int(label,16))

        image = Image.new('RGB', (size, size))
        draw = ImageDraw.Draw(image)
        draw.text(xy, code, font=font, anchor="mm")

        fileName = savePath+"/"+str(label)+".png"
        image.save(fileName)
        print("Save font image :"+fileName)


def utf8tounicode(utf8code, integer=True):
    """  
    utf8code: list of utf8code
    interger: If true return number of unicode, else return character
    """
    if integer:
        unico = []
        for code in data['utf8code']:
            unico.append(ord(bytes.fromhex(code).decode('utf-8')))
        return unico
    else:
        c = []
        for code in data['utf8code']:
            c.append(bytes.fromhex(code).decode('utf-8'))
        return c


# Run code
if __name__ == "__main__":
    import pandas as pd
    import dataset

    # head = ['utfcode', 'char']
    data = pd.read_csv(
        "/home/eugene/workspace/dataset/3107jp.csv",index_col=0)
    print(data.head())
    # data.to_csv("/home/eugene/workspace/dataset/3107jp.csv")













    # Extract 7font images, 3000 clases
    # font7dir = "/home/eugene/workspace/resource/font/7"

    # fontPaths = dataset.getFilesPath(font7dir)
    # print(fontPaths)

    # saveDir="/home/eugene/workspace/dataset/7font"
    # head = ['utfcode', 'char']
    # data = pd.read_csv(
    #     "/home/eugene/workspace/dataset/3107jp_utf16", names=head, sep=' ')

    # print(data.head())
    # for font in fontPaths:
    #     ttfwrite(font,saveDir,data['utfcode'].values,padding=2)
    #     pass



