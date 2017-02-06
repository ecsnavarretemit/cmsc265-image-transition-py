# exceptions.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

class NoImagesExeption(Exception):
  def __init__(self, msg):
    # call the parent class init function
    Exception.__init__(self, msg)

    # save the message of the exception
    self.msg = msg

class InvalidDlibPredictorException(Exception):
  def __init__(self, msg):
    # call the parent class init function
    Exception.__init__(self, msg)

    # save the message of the exception
    self.msg = msg

class InvalidOpenCVCascadeException(Exception):
  def __init__(self, msg):
    # call the parent class init function
    Exception.__init__(self, msg)

    # save the message of the exception
    self.msg = msg

