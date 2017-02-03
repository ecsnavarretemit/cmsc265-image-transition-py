#!/usr/bin/env python

# run.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import cv2
import dlib
import numpy as np

ASSETS_PATH = os.path.join(os.getcwd(), "assets/img")
PREDICTOR_PATH = os.path.join(os.getcwd(), "data/shape-predictor/shape_predictor_68_face_landmarks.dat")
CASCADE_PATH = os.path.join(os.getcwd(), "data/opencv-cascades/haarcascades/haarcascade_frontalface_default.xml")

predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade = cv2.CascadeClassifier(CASCADE_PATH)

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

def get_landmarks(im):
  rects = cascade.detectMultiScale(im, 1.3, 5)

  # dereference list elements into separate variables
  x, y, w, h = rects[0]

  # get the image width and height
  height, width, _ = im.shape

  # convert all parameters to int doing so will prevent errors with incompatibility with boost-python
  rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

  # create 2 dimensional array for the matrix
  items = []
  for p in predictor(im, rect).parts():
    items.append([p.x, p.y])

  # append the edges and the halfway point from the edges
  items.append([0, 0])
  items.append([0, int(height / 2)])
  items.append((0, int(height - 1)))
  items.append([int(width / 2), 0])
  items.append([int(width - 1), 0])
  items.append((int(width - 1), int(height - 1)))
  items.append((int(width / 2), int(height - 1)))
  items.append((int(width - 1), int(height / 2)))

  # return a new numpy matrix
  return np.matrix(items)

def get_points(landmarks):
  points = []

  for _, point in enumerate(landmarks):
    pos = (point[0, 0], point[0, 1])

    points.append(pos)

  return points

def get_triangles(rect, points):
  triangles = []

  subdiv = cv2.Subdiv2D(rect)

  for point in points:
    subdiv.insert(point)

  triangle_list = subdiv.getTriangleList()

  for triangle in triangle_list:
    point1 = (triangle[0], triangle[1])
    point2 = (triangle[2], triangle[3])
    point3 = (triangle[4], triangle[5])

    if rect_contains(rect, point1) and rect_contains(rect, point2) and rect_contains(rect, point3):
      triangles.append((point1, point2, point3))

  return triangles

def draw_triangles(im, landmarks):
  # clone the image instance to preserve the original instance
  im = im.copy()

  # assemble the rectangle
  size = im.shape
  rect = (0, 0, size[1], size[0])

  # get the coordinates of the points of all triangles in the picture
  triangles = get_triangles(rect, get_points(landmarks))

  # draw lines between the three points to form a rectangle
  for point1, point2, point3 in triangles:
    cv2.line(im, point1, point2, (255, 255, 255), 1, cv2.LINE_AA, 0)
    cv2.line(im, point2, point3, (255, 255, 255), 1, cv2.LINE_AA, 0)
    cv2.line(im, point3, point1, (255, 255, 255), 1, cv2.LINE_AA, 0)

  # return the manipulated image
  return im

if __name__ == '__main__':
  img1 = cv2.imread(ASSETS_PATH + '/img-1.jpg')
  cv2.imshow('Result', draw_triangles(img1, get_landmarks(img1)))
  cv2.waitKey(0)
  cv2.destroyAllWindows()


