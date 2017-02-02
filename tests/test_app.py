# test_app.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import sys
import unittest
import shutil

# modify the path
sys.path.insert(0, os.path.abspath(__file__+"/../.."))

import app

class GrayScaleTest(unittest.TestCase):
  out_dir = os.path.join(os.getcwd(), "out")

  def setUp(self):
    if not os.path.exists(self.out_dir):
      os.makedirs(self.out_dir)

  def test_grayscale(self):
    app.grayscale(os.path.join(os.getcwd(), "assets/img/leaf.jpg"), self.out_dir + "/grayscale.jpg")

    self.assertTrue(os.path.exists(self.out_dir + "/grayscale.jpg"))

  def tearDown(self):
    shutil.rmtree(self.out_dir)


