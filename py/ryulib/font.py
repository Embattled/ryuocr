from PIL import Image, ImageFont, ImageDraw, ImageChops, ImageMorph
import pathlib


def ttfshow(fontpath, labels, fontpoint=64):
    """ 
    Return font images of unicode list
    """
    font = ImageFont.truetype(fontpath, fontpoint)

    # image
    image = Image.new('RGB', (fontpoint, fontpoint))
    draw = ImageDraw.Draw(image)

    images = []
    # read
    for label in labels:
        draw.text((0, 0), chr(label), font=font)
        images.append(image)
    return images


def ttfwrite(fontpath, outputdir,labels, fontpoint=64):
    """ 
    Write font images of unicode list to files
    """
    image = Image.new('RGB', [fontpoint, fontpoint])
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(fontpath, fontpoint)


    for label in labels:
        code=chr(label)
        draw.text((0, 0), code, font=font)
        image.save(outputdir+'/'+)