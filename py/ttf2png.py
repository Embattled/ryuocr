#USAGE:
#python TCODfont.py /path/to/font/font.ttf 12

from PIL import Image, ImageFont, ImageDraw, ImageChops,ImageMorph,

fontpath = "yumin.ttf"
fontpoint = 100

image = Image.new('RGB', [fontpoint, fontpoint])
draw = ImageDraw.Draw(image)
font = ImageFont.truetype(fontpath, fontpoint)
code=0x6771

# i = 0
# for row in range(0, 16):
    # for col in range(0, 16):


draw.text((0, 0), chr(code), font=font)
        # i += 1

image=ImageChops.invert(image)

# mo=ImageMorph.MorphOp()

imageaf=image.transform((100,100),Image.AFFINE,())

# image.save('tcod.png')
imageaf.save('2.png')