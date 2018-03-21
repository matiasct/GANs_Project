from PIL import Image, ImageFilter, ImageEnhance
import sys,os

'''
This file loads images from a given folder, creates a new set of augmented images,
and saves them in another folder.
'''
from PIL import Image
import os, os.path

imgs = []
newimg = []
name = []
path = "chairs_vale"
valid_images = [".jpg",".gif",".png",".tga", ".jpeg"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append(Image.open(os.path.join(path,f)))

for image in imgs:
    rgb_im = image.convert('RGB')
    newimg.append(rgb_im.resize((256,256)))

for i, image in enumerate(newimg):
    name = 'chairs_vale_processed/Image '+str(i+3000)+'.jpg'
    image.save(name, 'JPEG')
