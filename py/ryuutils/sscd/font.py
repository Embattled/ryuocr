from PIL import Image, ImageFont, ImageDraw, ImageChops, ImageMorph
import pathlib
import os


def ttfimageget(fontpath, word, size=64, padding=2, background=0, fill=255):
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
    draw.text(xy, word, font=font, anchor="mm", fill=fill)

    return image


def ttfimageshow(fontpath, word, size=64, padding=2, background=0, fill=255):
    """ 
    Given ttf file path and character.
    Show a font character image.
    """
    image = ttfimageget(fontpath, word, size, padding, background, fill)
    image.show(title=word)


def fontimageget(font, word, size, background=0, fill=255):
    """ 
    Given font object and character.
    Return a font character image.
    """
    # image
    image = Image.new('L', size=(size, size), color=background)
    draw = ImageDraw.Draw(image)

    xy = (size/2, size/2)
    draw.text(xy, word, font=font, anchor="mm", fill=fill)

    return image


def fontimageshow(font, word, size, background=0, fill=255):
    image = fontimageget(font, word, size, background, fill)
    image.show(title=word)


def ttfdictget(fontpath, dicts, size=64, padding=2, background=0, fill=255):
    """ 
    Return a font character image with given dict list.
    """

    fontpoint = size-padding*2
    font = ImageFont.truetype(fontpath, fontpoint)

    images = []
    # read
    for word in dicts:
        images.append(fontimageget(font, word, size, background, fill))
    return images


def multi_ttfdictget(fonts:list, dicts:dict, size=64, padding=2,background=0, fill=255):
    """ 
    Return a font character image with given dict list and multiple ttf files.
    """
    images = []
    labels = []

    for font in fonts:
        images.extend(ttfdictget(font, dicts.keys(), size, padding, background, fill))
        labels.extend(dicts.values())

    return images, labels


