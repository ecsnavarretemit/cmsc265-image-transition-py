#!/usr/bin/env python

# run.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
from app.image_manipulator import create_sequence
from app.image_manipulator.exceptions import NoImagesExeption, \
  InvalidDlibPredictorException, InvalidOpenCVCascadeException

ASSETS_PATH = os.path.join(os.getcwd(), "assets/img")
OUTPUT_PATH = os.path.join(os.getcwd(), "out")

if __name__ == '__main__':
  try:
    create_sequence(ASSETS_PATH, OUTPUT_PATH, extensions=['jpg'])
  except (NoImagesExeption, InvalidDlibPredictorException, InvalidOpenCVCascadeException) as err:
    print(err.msg)


