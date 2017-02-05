# __init__.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import sys
import glob
import cv2
import numpy as np
from app.image_manipulator.coords import get_points, get_landmarks
from app.image_manipulator.triangles import calculate_delaunay_triangles, morph_triangle

# create custom exception
class NoImagesExeption(Exception):
  def __init__(self, msg):
    # call the parent class init function
    Exception.__init__(self, msg)

    # save the message of the exception
    self.msg = msg

def create_sequence(source_directory, output_directory, **kwargs):
  detected_extensions = kwargs.get('extensions', ['jpg', 'png'])
  output_extension = kwargs.get('output_extension', 'jpg')
  sequence_prefix = kwargs.get('sequence_prefix', 'seq-')

  # get all jpg image using a glob
  images = glob.glob(source_directory + '/*.' + ','.join(detected_extensions))

  # terminate if no images are found
  if len(images) == 0:
    raise NoImagesExeption("No images in the source directory: %s" % source_directory)

  # execution variables
  num_images = len(images)
  num_exec = num_images - 1
  num_images_per_exec = 2
  num_iterations = 10

  # convert images list to cv image instances list
  cv_img_instances = list(map(cv2.imread, images))

  # check if the output path exists, if not create it
  if not os.path.exists(output_directory):
    print("Creating output directory: %s" % output_directory)

    os.makedirs(output_directory)

  # show message the image sequences creation has begun
  print("Creating image sequences.")

  img_ctr = 0
  for i in range(0, num_exec):
    # get two opencv image instances from the array of cv image instances
    img_set = cv_img_instances[i:num_images_per_exec + i]

    # get the facial landmarks. after that get the points in the image
    # with the corresponding landmarks including the image borders
    points1 = get_points(get_landmarks(img_set[0]))
    points2 = get_points(get_landmarks(img_set[1]))

    # get the image width and height
    height, width, _ = img_set[0].shape

    # loop inclusive of the last number depending on the value of num_iterations
    for j in range(0, num_iterations + 1):
      # calculate the alpha (opacity) of the interweaved images
      alpha = j / num_iterations

      # Compute weighted average point coordinates
      points = []
      for i in range(0, len(points1)):
        x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
        y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]

        points.append((x, y))

      # Delaunay triangulation
      rect = (0, 0, width, height)
      triangles = calculate_delaunay_triangles(rect, points)

      # Allocate space for final output
      morphed_image = np.zeros(img_set[0].shape, dtype=img_set[0].dtype)

      # loop through all triangles found in the images
      for triangle in triangles:
        x, y, z = triangle

        x = int(x)
        y = int(y)
        z = int(z)

        t1 = [points1[x], points1[y], points1[z]]
        t2 = [points2[x], points2[y], points2[z]]
        t = [points[x], points[y], points[z]]

        # Morph one triangle at a time.
        morph_triangle(img_set[0], img_set[1], morphed_image, t1, t2, t, alpha)

      # increment the image counter
      img_ctr += 1

      # resolve the filename and the path
      image_output_path = output_directory + "/" + sequence_prefix + str(img_ctr).zfill(3) + "." + output_extension

      # write the image to the file system
      cv2.imwrite(image_output_path, morphed_image)

  # show message the image sequences creation has ended
  print("Done creating image sequences. Please check the directory: %s" % output_directory)


