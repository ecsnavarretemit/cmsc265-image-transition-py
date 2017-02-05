#!/usr/bin/env python

# cli.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import click
from app.image_manipulator import create_sequence, NoImagesExeption

@click.command()
@click.option('--source-imgs-path', default="assets/img", help='Source directory containing the images')
@click.option('--output-path', default="out", help='Output directory to place the generated images')
@click.option('--output-extension', default="jpg", help='File extension of the generated images')
@click.option('--sequence-prefix', default="seq-", help='Prefix of the image sequences e.g. seq-')
@click.option('--extension', default=['jpg'], multiple=True)
def create(source_imgs_path, output_path, output_extension, sequence_prefix, extension):
  resolved_source_path = os.path.join(os.getcwd(), source_imgs_path)
  resolved_output_path = os.path.join(os.getcwd(), output_path)

  try:
    # execute the function
    create_sequence(resolved_source_path,
                    resolved_output_path,
                    extensions=list(extension),
                    output_extension=output_extension,
                    sequence_prefix=sequence_prefix)

  except NoImagesExeption as err:
    # show the error message
    click.echo(err.msg)

if __name__ == '__main__':
  create()


