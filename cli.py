#!/usr/bin/env python

# cli.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import click
import shutil
import subprocess
from app.image_manipulator import create_sequence
from app.image_manipulator.exceptions import NoImagesExeption, \
  InvalidDlibPredictorException, InvalidOpenCVCascadeException

@click.group()
def cli():
  pass

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
def create_image_sequence(source_imgs_path, output_path, output_extension, sequence_prefix, extension, dlib_predictor_path, opencv_cascade_path):
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

@click.command()
@click.option('--frame-delay', default=10, help='Time in seconds to display the next image after pausing')
@click.option('--input-images',
              default="./out/*.jpg",
              help='Glob pointing to the path of input images',
              type=click.Path())
@click.option('--output-path',
              default="./out/sequence.mpeg",
              help='Path and filename of the output video',
              type=click.Path())
def create_video(input_images, output_path, frame_delay):
  convert = shutil.which('convert')
  ffmpeg = shutil.which('ffmpeg')

  # show error message and terminate immediately
  if convert is None:
    click.echo("ImageMagick must be installed to use this command")
    return

  # show error message and terminate immediately
  if ffmpeg is None:
    click.echo("FFmpeg must be installed to use this command")
    return

  try:
    resolved_input_images = os.path.join(os.getcwd(), input_images)
    resolved_output_path = os.path.join(os.getcwd(), output_path)

    # show a message before generating the video
    click.echo("Generating video out of the image sequence.")

    command = "convert -delay %s -quality 100 %s %s" % (frame_delay, resolved_input_images, resolved_output_path)

    # run the command
    subprocess.run(command, shell=True, check=True)

    # show a message after generating the video
    click.echo("Done generating video Please check the file at: %s" % os.path.realpath(resolved_output_path))
  except subprocess.CalledProcessError as err:
    # show the error message
    click.echo(err)

if __name__ == '__main__':
  # add commands to the cli
  cli.add_command(create_image_sequence)
  cli.add_command(create_video)

  cli()


