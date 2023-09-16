# -*- coding: utf-8 -*-
"""
Exceptions
=======
This module contains custom exceptions.
author: Damian Wilary
email: dmw51@cam.ac.uk
"""


class BaseRDEException(Exception):
    """
    Base exception class
    """


class NoArrowsFoundException(BaseRDEException):
    """
    Raised when no arrows have been found in an image
    """

class NoDiagramsFoundException(BaseRDEException):
    """
    Raised when no diagrams have been found in an image
    """

class SchemeReconstructionFailedException(BaseRDEException):
    """
    Raised when the scheme could not be found
    """