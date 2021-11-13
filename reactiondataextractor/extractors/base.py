# -*- coding: utf-8 -*-
"""
Base
=====
This module contains the base extraction class
author: Damian Wilary
email: dmw51@cam.ac.uk
"""
from abc import abstractmethod
import logging

log = logging.getLogger(__name__)


class BaseExtractor:

    @abstractmethod
    def __init__(self, fig):
        self.fig = fig

    @abstractmethod
    def extract(self):
        """This method extracts objects (arrows, conditions, diagrams or labels) from ``fig``"""
        pass

    @abstractmethod
    def plot_extracted(self, ax):
        """This method places extracted objects on canvas of ``ax``"""
        pass

    @property
    @abstractmethod
    def extracted(self):
        """This method returns extracted objects"""
        pass

    @property
    def img(self):
        return self.fig.img


class Candidate:
    """Allows objects to behave like intermediate attribute storage that can pass their attributes when as final
    object is instantiated"""

    def pass_attributes(self):
        """Returns all attributes in the form of a dictionary"""
        return vars(self)
