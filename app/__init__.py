# __init__.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import cv2

def grayscale(source, destination):
  if not os.path.exists(source):
    raise Exception("%s image does not exist" % source)

  # read the image file
  image = cv2.imread(source)

  # process the image
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # write the file
  cv2.imwrite(destination, gray_image)


