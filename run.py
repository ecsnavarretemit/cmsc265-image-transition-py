#!/usr/bin/env python

# run.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import glob
import cv2
import dlib
import numpy as np

ASSETS_PATH = os.path.join(os.getcwd(), "assets/img")
OUTPUT_PATH = os.path.join(os.getcwd(), "out")
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

# Apply affine transform calculated using source_triangle and destination_triangle to src and
# output an image of size.
def apply_affine_transform(src, source_triangle, destination_triangle, size):
  # Given a pair of triangles, find the affine transform.
  warped_matrix = cv2.getAffineTransform(np.float32(source_triangle), np.float32(destination_triangle))

  # Apply the Affine Transform just found to the src image
  dst = cv2.warpAffine(src,
                       warped_matrix,
                       (size[0], size[1]),
                       None,
                       flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)

  return dst

# Warps and alpha blends triangular regions from img1 and img2 to img
def morph_triangle(img1, img2, img, t1, t2, t, alpha):
  # Find bounding rectangle for each triangle
  r1 = cv2.boundingRect(np.float32([t1]))
  r2 = cv2.boundingRect(np.float32([t2]))
  r = cv2.boundingRect(np.float32([t]))

  # Offset points by left top corner of the respective rectangles
  t1_rect = []
  t2_rect = []
  t_rect = []

  for i in range(0, 3):
    t_rect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
    t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
    t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

  # Get mask by filling triangle
  mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
  cv2.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)

  # Apply warpImage to small rectangular patches
  img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
  img2_rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

  size = (r[2], r[3])
  warped_image1 = apply_affine_transform(img1_rect, t1_rect, t_rect, size)
  warped_image2 = apply_affine_transform(img2_rect, t2_rect, t_rect, size)

  # Alpha blend rectangular patches
  img_rect = (1.0 - alpha) * warped_image1 + alpha * warped_image2

  # Copy triangular region of the rectangular patch to the output image
  img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * (1 - mask) + img_rect * mask

# Calculate delanauy triangle
def calculate_delaunay_triangles(rect, points):
  triangles = get_triangles(rect, points)

  delaunay_triangles = []
  pt = []
  count = 0

  for triangle in triangles:
    point1, point2, point3 = triangle

    pt.append(point1)
    pt.append(point2)
    pt.append(point3)

    count = count + 1
    ind = []

    for j in range(0, 3):
      for k in range(0, len(points)):
        if abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0:
          ind.append(k)

    if len(ind) == 3:
      delaunay_triangles.append((ind[0], ind[1], ind[2]))

    pt = []

  return delaunay_triangles

def draw_triangles(im, landmarks):
  # clone the image instance to preserve the original instance
  im = im.copy()

  # get the image width and height
  height, width, _ = img1.shape

  # assemble the rectangle
  rect = (0, 0, height, width)

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
  # get all jpg image using a glob
  images = glob.glob(ASSETS_PATH + '/*.jpg')

  # execution variables
  num_images = len(images)
  num_exec = num_images - 1
  num_images_per_exec = 2
  num_iterations = 10

  # convert images list to cv image instances list
  cv_img_instances = list(map(cv2.imread, images))

  # check if the output path exists, if not create it
  if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

  img_ctr = 0
  for i in range(0, num_exec):
    img_set = cv_img_instances[i:num_images_per_exec + i]

    # loop inclusive of the last number depending on the value of num_iterations
    for j in range(0, num_iterations + 1):
      # calculate the alpha (opacity) of the interweaved images
      alpha = j / num_iterations

      # get the facial landmarks. after that get the points in the image
      # with the corresponding landmarks including the image borders
      points1 = get_points(get_landmarks(img_set[0]))
      points2 = get_points(get_landmarks(img_set[1]))

      # get the image width and height
      height, width, _ = img_set[0].shape

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

      # write the image to the file system
      cv2.imwrite(OUTPUT_PATH + "/seq-" + str(img_ctr).zfill(3) + ".jpg", morphed_image)


