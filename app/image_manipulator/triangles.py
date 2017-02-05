# triangles.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import cv2
import numpy as np
from app.image_manipulator.coords import rect_contains, get_points

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

def draw_triangles(img, landmarks):
  # clone the image instance to preserve the original instance
  img = img.copy()

  # get the image width and height
  height, width, _ = img.shape

  # assemble the rectangle
  rect = (0, 0, height, width)

  # get the coordinates of the points of all triangles in the picture
  triangles = get_triangles(rect, get_points(landmarks))

  # draw lines between the three points to form a rectangle
  for point1, point2, point3 in triangles:
    cv2.line(img, point1, point2, (255, 255, 255), 1, cv2.LINE_AA, 0)
    cv2.line(img, point2, point3, (255, 255, 255), 1, cv2.LINE_AA, 0)
    cv2.line(img, point3, point1, (255, 255, 255), 1, cv2.LINE_AA, 0)

  # return the manipulated image
  return img


