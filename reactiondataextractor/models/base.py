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
from typing import List

log = logging.getLogger(__name__)


class BaseExtractor:
    """Base for all extractor classes. All subclasses should implement `extract`, and `plot_extracted` methods
    as well as `extracted` property"""

    @abstractmethod
    def __init__(self, fig: 'Figure'):
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

    def set_nearest_as_parent(self, objs: List['Panel'], below_panel:bool=False):
        """Sets nearest object from objs as the panel. if 'below_panel' is set to True, then
        the nearest object from objs below self in the image is selected

        :param objs: list of potential parents
        :type objs: List[Panel]
        :param below_panel: whether only objects below self in the iamge should be considered, defaults to False
        :type below_panel: bool, optional
        """

        if below_panel:
            orientations = [obj.panel.find_relative_orientation(self.panel) for obj in objs]
            # Select overlapping labels and those below the panel
            objs = [objs[idx] for idx in range(len(objs)) if sum(orientations[idx]) == 0 or orientations[idx][2] == 1]
        try:
            parent = min(objs, key=self.panel.edge_separation)
            parent.children.append(self)
        except ValueError:
            pass
