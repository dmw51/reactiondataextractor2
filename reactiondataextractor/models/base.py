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
        self._fig = fig

    @property
    def fig(self):
        return self._fig

    @abstractmethod
    def extract(self):
        """This method extracts objects (arrows, conditions, diagrams or labels) from ``self.fig``"""
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


class TextRegion:
    """A class for text region objects such as labels and conditions, which ought to be assigned to their respective
    parent region - diagrams and arrows respectively"""

    def set_nearest_as_parent(self, panels):
        """Sets parent panel as the nearest of all `panels`"""
        parent = min(panels, key=self.panel.edge_separation)
        parent.children.append(self)
        # self.parent_panel = None # Ensure no cyclic reference
