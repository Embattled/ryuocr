from PIL import Image, ImageFont, ImageDraw, ImageChops, ImageMorph
import pathlib
import os

font_file = "AozoraMinchoRegular.ttf"
dir_path = os.path.dirname(os.path.abspath(__file__))
default_font_path = os.path.join(dir_path, font_file)


def fontpathCharImageGet(char, fontpath=default_font_path, size=64, padding=2, background=0, fill=255):
    """ 
    Given ttf file path and character.
    Return a font character image.
    """
    fontpoint = size-padding*2
    font = ImageFont.truetype(fontpath, fontpoint)

    # image
    image = Image.new('L', size=(size, size), color=background)
    draw = ImageDraw.Draw(image)

    xy = (size/2, size/2)    
    draw.text(xy, str(char), font=font, anchor="mm", fill=fill)
    return image

def fontpathLabelImageGet(char, fontpath=default_font_path, size=(64,64), padding=2, background=0, fill=255):
    """ 
    Given ttf file path and number.
    Return a font number image.
    """

    fontpoint = min(size[0],size[1])-padding*2
    font = ImageFont.truetype(fontpath, fontpoint)

    # image
    image = Image.new('L', size=size, color=background)
    draw = ImageDraw.Draw(image)

    xy = (size[0]//2, size[1]//2)
    draw.text(xy, str(char), font=font, anchor="mm", fill=fill)

    return image

def fontCharImageShow(fontpath, character, size=64, padding=2, background=0, fill=255):
    """ 
    Given ttf file path and character.
    Show a font character image.
    """
    image = fontpathCharImageGet(
        character, fontpath, size, padding, background, fill)
    image.show(title=character)


def fontobjCharImageGet(font, character, size, background=0, fill=255):
    """ 
    Given font object and character.
    Return a font character image.
    """
    # image
    image = Image.new('L', size=(size, size), color=background)
    draw = ImageDraw.Draw(image)

    xy = (size/2, size/2)
    draw.text(xy, character, font=font, anchor="mm", fill=fill)

    return image


def fontCharImageDictGet(fontpath, dictionary: list, size=64, padding=2, background=0, fill=255):
    """ 
    Return a list of font character image with a given dictionary.
    """

    fontpoint = size-padding*2
    font = ImageFont.truetype(fontpath, fontpoint)

    images = []
    # read
    for character in dictionary:
        images.append(fontobjCharImageGet(
            font, character, size, background, fill))
    return images


def multiFontCharImageDictget(fonts: list, dicts: dict, size=64, padding=None, background=0, fill=255):
    """ 
    Return a font character image with given dict list and multiple ttf files.
    """
    images = []
    labels = []

    if padding==None:
        padding = size//32

    for font in fonts:
        images.extend(fontCharImageDictGet(
            font, dicts.keys(), size, padding, background, fill))
        labels.extend(dicts.values())

    return images, labels


def getExampleFontImage(char,anchor="mm",size=64, padding=2, background=0, fill=255):
    fontpoint = size-padding*2
    font = ImageFont.truetype(default_font_path, fontpoint)

    # image
    image = Image.new('L', size=(size, size), color=background)
    draw = ImageDraw.Draw(image)

    # xy = (padding,padding)    
    xy = (size/2, size/2)    
    draw.text(xy, str(char), font=font, anchor=anchor, fill=fill)

    return image

if __name__ =="__main__":
    
    img=getExampleFontImage("国",anchor="mm",size=64,padding=2)
    img.save("/home/eugene/workspace/ryuocr/fontImage.png")