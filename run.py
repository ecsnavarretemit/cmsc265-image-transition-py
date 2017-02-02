#!/usr/bin/env python

# run.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import app

out_dir = os.path.join(os.getcwd(), "out")

if not os.path.exists(out_dir):
  os.makedirs(out_dir)

app.grayscale(os.path.join(os.getcwd(), "assets/img/leaf.jpg"), out_dir + "/grayscale.jpg")


