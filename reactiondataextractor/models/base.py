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

    def set_nearest_as_parent(self, objs, below_panel=False):
        """Sets parent panel as the nearest of all `panels`"""
        if below_panel:
            orientations = [obj.panel.find_relative_orientation(self.panel) for obj in objs]
            # Select overlapping labels and those below the panel
            objs = [objs[idx] for idx in range(len(objs)) if sum(orientations[idx]) == 0 or orientations[idx][2] == 1]
        # dists = sorted([(obj, self.panel.edge_separation(obj)) for obj in objs], key=lambda x: x[1])[:2]
        # if len(dists) == 1:
        #     parent, dist = dists[0]
        #     parent.children.append(self)
        # else:
        try:
            parent = min(objs, key=self.panel.edge_separation)
            # if not parent.children:
            parent.children.append(self)
        except ValueError:
            pass
        # else:
            #Conditionally replace the old children, if the old child can be reassigned, or assign it to the second closest
            # diag, or add if neither can be done
            # dists = sorted([(obj, self.panel.edge_separation(obj)) for obj in objs], key=lambda x: x[1])[:2]
    # if len(dists) == 1:
    #     parent, dist = dists[0]
    #     parent.children.append(self)
        # self.parent_panel = None # Ensure no cyclic reference
