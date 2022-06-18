import cv2
import os
import sys
import pydicom as dicom
from PIL import Image,ImageOps
from pydicom import dcmread
import numpy as np
from numpy import load
from numpy import expand_dims


for root, dirs, files in os.walk(sys.argv[1]):
    for filename in files:
        fn=os.path.join(root, filename)
        plan = dicom.read_file(fn)
        img=(plan.pixel_array.astype("float"))
        im = cv2.convertScaleAbs(img-np.min(img), alpha=(255.0 / min(np.max(img)-np.min(img), 10000)))
        im = np.uint8(im)
        im=Image.fromarray(im) 
        im.save(filename+'.jpg')
