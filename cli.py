#!/usr/bin/env python

# cli.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import click
from app.image_manipulator import create_sequence
from app.image_manipulator.exceptions import NoImagesExeption, \
  InvalidDlibPredictorException, InvalidOpenCVCascadeException

@click.command()
@click.option('--source-imgs-path', default="assets/img", help='Source directory containing the images', type=click.Path())
@click.option('--output-path', default="out", help='Output directory to place the generated images', type=click.Path())
@click.option('--output-extension', default="jpg", help='File extension of the generated images')
@click.option('--sequence-prefix', default="seq-", help='Prefix of the image sequences e.g. seq-')
@click.option('--extension', default=['jpg'], multiple=True)
@click.option('--dlib-predictor-path',
              default="data/shape-predictor/shape_predictor_68_face_landmarks.dat",
              help='Path to the file of the DLib predictor data file',
              type=click.Path())
@click.option('--opencv-cascade-path',
              default="data/opencv-cascades/haarcascades/haarcascade_frontalface_default.xml",
              help='Path to the file of the OpenCV cascade XML file',
              type=click.Path())
def create(source_imgs_path, output_path, output_extension, sequence_prefix, extension, dlib_predictor_path, opencv_cascade_path):
  resolved_source_path = os.path.join(os.getcwd(), source_imgs_path)
  resolved_output_path = os.path.join(os.getcwd(), output_path)

  # set the environment variables for the process
  os.environ['PREDICTOR_PATH'] = os.path.join(os.getcwd(), dlib_predictor_path)
  os.environ['CASCADE_PATH'] = os.path.join(os.getcwd(), opencv_cascade_path)

  try:
    # execute the function
    create_sequence(resolved_source_path,
                    resolved_output_path,
                    extensions=list(extension),
                    output_extension=output_extension,
                    sequence_prefix=sequence_prefix)

  except (NoImagesExeption, InvalidDlibPredictorException, InvalidOpenCVCascadeException) as err:
    # show the error message
    click.echo("Error - %s" % err.msg)

if __name__ == '__main__':
  create()


