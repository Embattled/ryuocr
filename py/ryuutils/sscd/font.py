from PIL import Image, ImageFont, ImageDraw, ImageChops, ImageMorph
import pathlib
import os

def ttfimageshow(fontpath, label, size=64,padding=0):
    """ 
    Show a font character image.
    """
    fontpoint = size-padding*2
    font = ImageFont.truetype(fontpath, fontpoint)

    # image
    image = Image.new('RGB', (fontpoint, fontpoint))
    draw = ImageDraw.Draw(image)

    draw.text((0, 0), label, font=font, anchor="lt")
    image.show()


def ttfimageget(fontpath, labels, size=64, padding=0):
    """ 
    Return a font character image.
    """
    fontpoint = size-padding*2
    font = ImageFont.truetype(fontpath, fontpoint)

    # image
    image = Image.new('RGB', (size, size))
    draw = ImageDraw.Draw(image)

    images = []
    # read
    for label in labels:
        draw.text((padding, padding), label, font=font, anchor="lt")
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


# Run code
if __name__ == "__main__":
    ttfpath="/home/eugene/workspace/resource/font/7/AozoraMinchoRegular.ttf"

    ttfimageshow(ttfpath,'龍',size=64,padding=1)