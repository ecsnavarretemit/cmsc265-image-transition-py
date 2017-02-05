# coords.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import cv2
import dlib
import numpy as np

# OpenCV and DLib cascade and predictor paths
cwd = os.getcwd()
DEFAULT_PREDICTOR_PATH = os.path.join(cwd, "data/shape-predictor/shape_predictor_68_face_landmarks.dat")
DEFAULT_CASCADE_PATH = os.path.join(cwd, "data/opencv-cascades/haarcascades/haarcascade_frontalface_default.xml")

predictor = None
cascade = None

def get_dlib_predictor():
  global predictor

  # return the resolved predictor if it is not None
  if predictor is not None:
    return predictor

  predictor_path = os.environ.get('PREDICTOR_PATH', DEFAULT_PREDICTOR_PATH)

  if not os.path.exists(predictor_path) or not os.path.isfile(predictor_path):
    pass

  predictor = dlib.shape_predictor(predictor_path)

  return predictor

def get_opencv_cascade():
  global cascade

  # return the resolved cascade if it is not None
  if cascade is not None:
    return cascade

  cascade_path = os.environ.get('CASCADE_PATH', DEFAULT_CASCADE_PATH)

  if not os.path.exists(cascade_path) or not os.path.isfile(cascade_path):
    pass

  cascade = cv2.CascadeClassifier(cascade_path)

  return cascade

# Check if a point is inside a rectangle
def rect_contains(rect, point):
  if point[0] < rect[0]:
    return False
  elif point[1] < rect[1]:
    return False
  elif point[0] > rect[2]:
    return False
  elif point[1] > rect[3]:
    return False

  return True

def get_landmarks(img):
  rects = get_opencv_cascade().detectMultiScale(img, 1.3, 5)

  # dereference list elements into separate variables
  x, y, w, h = rects[0]

  # get the image width and height
  height, width, _ = img.shape

  # convert all parameters to int doing so will prevent errors with incompatibility with boost-python
  rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

  # create 2 dimensional array for the matrix
  items = []
  for p in get_dlib_predictor()(img, rect).parts():
    items.append([p.x, p.y])

  x_start_point = 0
  x_half_point = int((width / 2) - 1)
  x_end_point = int(width - 1)

  y_start_point = 0
  y_half_point = int((height / 2) - 1)
  y_end_point = int(height - 1)

  # append the edges and the halfway point from the edges
  items.append([x_start_point, y_start_point])
  items.append([x_start_point, y_half_point])
  items.append((x_start_point, y_end_point))
  items.append([x_half_point, y_start_point])
  items.append([x_end_point, y_start_point])
  items.append((x_end_point, y_end_point))
  items.append((x_half_point, y_end_point))
  items.append((x_end_point, y_half_point))

  # return a new numpy matrix
  return np.matrix(items)

def get_points(landmarks):
  points = []

  for point in landmarks:
    pos = (point[0, 0], point[0, 1])

    points.append(pos)

  return points


