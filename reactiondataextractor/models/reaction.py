# -*- coding: utf-8 -*-
"""
Reaction
=======

This module contains classes for representing reaction elements.

author: Damian Wilary
email: dmw51@cam.ac.uk

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import logging
import os
import re
from collections import Counter
from typing import Sequence
from enum import Enum

import cv2
import numpy as np


from configs import ExtractorConfig
from processors import Isolator, Binariser
from reactiondataextractor.configs import Config


from .base import TextRegion, Candidate
from .geometry import Point, Line
from .segments import Panel, PanelMethodsMixin, Figure
# from reactiondataextractor.extractors.labels import Label
log = logging.getLogger('extract.reaction')

parent_dir = os.path.dirname(os.path.abspath(__file__))
r_group_correct_file = os.path.join(parent_dir, '..', 'dict', 'r_placeholders.txt')

class BaseReactionClass(object):
    """
    This is a base.py reaction class placeholder
    """


class LabelType(Enum):
    SIMPLE = 1
    VARIANTS = 2
    UNKNOWN = 3

    @staticmethod
    def assign_type(label_text):
        numbers_only = r'^\d+$'
        chars_only = r'^[[:alpha:]]+$'
        label_text = label_text.strip()
        if len(label_text) < 8 and re.search(r'\d+-\d+$', label_text) or re.search(r'\d+[A-Za-z]-[A-Za-z]$', label_text):
            return LabelType.VARIANTS
        else:
            return LabelType.SIMPLE
        # if re.match(numbers_only, label_text) or re.match(chars_only, label_text):
        #     return LabelType.SIMPLE
        #
        # elif '-' in label_text:
        #     return LabelType.VARIANTS
        # else: re.search(r'\d[a-z]', label_text):
        #     return LabelType.SINGLE_VARIANT

class Diagram(BaseReactionClass, PanelMethodsMixin):
    """This is a base.py class for chemical structures species found in diagrams (e.g. reactants and products)

    :param panel: bounding box a diagrams
    :type panel: Panel
    :param labels: label associated with a diagram
    :type labels: Label
    :param smiles: SMILES associated with a diagram
    :type smiles: str
"""

    @classmethod
    def from_coords(cls, left, right, top, bottom, label=None, smiles=None):
        """Class method used for instantiation from coordinates, as used within chemschematicresolver"""
        panel = Panel((left, right, top, bottom) )
        return cls(panel=panel, labels=[label], smiles=smiles)

    def __init__(self, panel, labels=None, smiles=None):
        self._panel = panel
        # self.labels = labels
        self._smiles = smiles
        self._base_smiles = ''
        self.smarts = []
        self.children = [] if labels is None else labels
        self.reaction_steps = []
        self.text_chars = []
        # self.markush_extensions = []
        self.r_groups = []
        self.r_group_variants = {}
        self.r_group_placeholders = []
        self.molecule = None
        self._base_molecule = None
        self._fingerprint = None
        self._corrected = False
        self.corners = []
        self.adjacency_matrix = []
        # self.positional_markush = False
        self.positions_markush_groups = []
        self.markush_freq_identifiers = []
        self.repeating_units = []
        super().__init__()

    def __eq__(self, other):
        if isinstance(other, Diagram):  # Only compare exact same types
            return self.panel == other.panel
        return False

    def __hash__(self):
        return hash(self.panel)

    def __repr__(self):
        return f'{self.__class__.__name__}(panel={self.panel}, smiles={self.smiles}, labels={self.labels})'

    def __str__(self):
        return f'{self.smiles if self.smiles else "???"}, labels: {self.labels}'

    @property
    def labels(self):
        return self.children

    @property
    def panel(self):
        return self._panel

    @property
    def crop(self):
        return self._panel.crop

    @property
    def center(self):
        return self._panel.center



    @property
    def smiles(self):
        return self._smiles

    @smiles.setter
    def smiles(self, smiles):
        self._smiles = smiles


class ResolvedDiagram:

    def __init__(self, smiles, label_text):
        self.smiles = smiles
        self.label_text = label_text
        self.molecule = AllChem.MolFromSmiles(self.smiles)

class ReactionStep(BaseReactionClass):
    """
    This class describes elementary steps in a reaction.

    :param arrow: arrow associated with this step
    :type arrow: BaseArrow
    :param reactants: reactants of a reaction step
    :type reactants: frozenset[Diagram]
    :param products: products of a reaction step
    :type products: frozenset[Diagram]

    """

    @property
    def conditions(self):
        return self.arrow.conditions

    def __init__(self, arrow, reactants, products, single_line=True):
        self.arrow = arrow
        self.reactants = reactants
        self.products = products
        self._diags = reactants + products
        self.single_line = single_line
        for diag in self._diags:
            diag.reaction_steps.append(self)

    def __eq__(self, other):
        if isinstance(other, ReactionStep):
            return (self.reactants == other.reactants and self.products == other.products and
                    self.arrow == other.arrow)
        return False

    def __repr__(self):
        return f'ReactionStep(reactants=({self.reactants}),products=({self.products}),{self.conditions})'

    def __str__(self):
        reactant_strings = [elem.smiles if elem.smiles else '???' for elem in self.reactants]
        product_strings = [elem.smiles if elem.smiles else '???' for elem in self.products]
        return ' + '.join(reactant_strings)+'  -->  ' + ' + '.join(product_strings)

    def __hash__(self):
        all_species = [species for group in iter(self) for species in group]
        species_hash = sum([hash(species) for species in all_species])
        return species_hash

    def __iter__(self):
        return iter((self.reactants, self.products))

    @property
    def nodes(self):
        return [self.reactants, self.conditions, self.products]

    def visualize(self, fig):
        _X_SEPARATION = 50
        elements = self.reactants + self.products + [self.arrow]
        orig_coords = [e.panel.in_original_fig() for e in elements]

        canvas_width = np.sum([c[3] - c[1] for c in orig_coords]) + _X_SEPARATION * (len(elements) - 1)
        canvas_height = max([c[2] - c[0] for c in orig_coords])

        canvas = np.zeros([canvas_height, canvas_width])
        x_end = 0
        for diag in self.reactants:
            self._place_panel_on_canvas(diag.panel, canvas,fig,  (x_end, 0))
            orig_coords = diag.panel.in_original_fig()
            x_end += orig_coords[3] - orig_coords[1] + _X_SEPARATION
        self._place_panel_on_canvas(self.arrow.panel, canvas, fig, (x_end, int(canvas_height//2)))
        orig_coords = self.arrow.panel.in_original_fig()
        x_end += orig_coords[3] - orig_coords[1] + _X_SEPARATION
        for diag in self.products:
            self._place_panel_on_canvas(diag.panel, canvas, fig, (x_end, 0))
            orig_coords = diag.panel.in_original_fig()
            x_end += orig_coords[3] - orig_coords[1] + _X_SEPARATION

        return canvas

    def _place_panel_on_canvas(self, panel, canvas,fig,  left_top):

        ## Specify coords of the paste region
        x, y = left_top
        w, h = panel.width, panel.height

        ## Specify coords of the crop region
        top, left, bottom, right = panel

        canvas[y:y+h, x:x+w] = fig.img[top:bottom, left:right]

        # return canvas


class Conditions(TextRegion):
    """
    This class describes conditions region and associated text

    :param panel: extracted region containing conditions
    :type panel: Panel
    :param conditions_dct: dictionary with all parsed conditions
    :type conditions_dct: dict
    :param parent_panel: reaction arrow, around which the search for conditions is performed
    :type parent_panel: SolidArrow
    :param diags: bounding boxes of all chemical structures found in the region
    :type diags: list[Panel]
    """

    def __init__(self, panel, conditions_dct, parent_panel=None, text=None, diags=None, _prior_class=None):
        self.panel = panel
        self.text = text
        self.conditions_dct = conditions_dct

        self._prior_class = _prior_class

        if diags is None:
            diags = []
        self._diags = diags

        self._parent_panel = parent_panel
        # if parent_panel:
        #     parent_panel.children.append(self)



    @property
    def arrow(self):
        return self._parent_panel

    def __repr__(self):
        return f'Conditions({self.panel}, {self.conditions_dct}, {self.arrow})'

    def __str__(self):
        delimiter = '\n------\n'
        return delimiter + 'Step conditions:' + \
               '\n'.join(f'{key} : {value}' for key, value in self.conditions_dct.items() if value)  + delimiter

    def __eq__(self, other):
        if other.__class__ == self.__class__:
            return self.panel == other.panel
        else:
            return False

    def __hash__(self):
        return hash(sum(self.panel.coords))

    @property
    def diags(self):
        return self._diags

    @property
    def anchor(self):
        a_pixels = self.arrow.pixels
        return a_pixels[len(a_pixels)//2]

    @property
    def coreactants(self):
        return self.conditions_dct['coreactants']

    @property
    def catalysts(self):
        return self.conditions_dct['catalysts']

    @property
    def other_species(self):
        return self.conditions_dct['other species']

    @property
    def temperature(self):
        return self.conditions_dct['temperature']

    @property
    def time(self):
        return self.conditions_dct['time']

    @property
    def pressure(self):
        return self.conditions_dct['pressure']

    @property
    def yield_(self):
        return self.conditions_dct['yield']

    def merge_conditions_regions(self, other_region):
        keys = self.conditions_dct.keys()
        new_dict = {}
        for k in keys:
            if isinstance(self.conditions_dct[k], Sequence):
                new_value = self.conditions_dct[k] + other_region.conditions_dct[k]
            else:
                val = self.conditions_dct[k]
                new_value = val if val else other_region.conditions_dct[k]
            new_dict[k] = new_value
        panel = self.panel.create_megapanel([self.panel, other_region.panel], fig=self.panel.fig)
        text = self.text + other_region.text
        diags = self._diags + other_region._diags

        return Conditions(panel=panel, conditions_dct=new_dict, parent_panel=self._parent_panel, text=text,diags=diags,
                          _prior_class=self._prior_class)


class Label(TextRegion, PanelMethodsMixin):
    """Describes labels and recognised text

    :param panel: bounding box a labels
    :type panel: Panel
    :param text: labels text
    :type text: str
    :param r_group: generic r_groups associated with a labels
    :type r_group: str"""

    @classmethod
    def from_coords(cls, left, right, top, bottom, text):
        panel = Panel((left, right, top, bottom))
        return cls(panel, text)

    def __init__(self, panel, text, r_groups=None, *, _prior_class=None):
        if r_groups is None:
            r_groups = []
        # if text is None:
        #     text = []
        self.panel = panel
        self._text = text
        self.r_groups = r_groups
        self._prior_class = _prior_class
        if self.text:
            self.type = LabelType.assign_type(self.text[0])
        else:
            self.type = LabelType.UNKNOWN
        self.root = None
        self.variant_indicators = None
        self.resolve_variants = False
        # self._parent_panel = parent_panel

    # @property
    # def diagram(self):
    #     return self._parent_panel

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    def __repr__(self):
        return f'Label(panel={self.panel}, text={self.text}, r_groups={self.r_groups})'

    def __str__(self):
        return f'Label(Text: {", ".join(sent for sent in self.text)})'

    def __hash__(self):
        return hash(self.panel)

    def __eq__(self, other):
        return isinstance(other, Label) and self.panel == other.panel


    def add_r_group_variables(self, var_value_label_tuples):
        """ Updates the R-groups for this labels."""

        self.r_group.append(var_value_label_tuples)

    def is_similar_to(self, other_label):
        text = ' '.join(self.text)
        other_text = ' '.join(other_label.text)
        chemical_number_self = re.match(r'\d+', text).group(0)
        chemical_number_other = re.match(r'\d+', other_text).group(0)
        if chemical_number_other == chemical_number_self:
            return True
        return False




class BaseArrow(PanelMethodsMixin):
    """Base arrow class common to all arrows
    :param pixels: pixels forming the arrows
    :type pixels: list[Point] or list[(int, int)]
    :param panel: bounding box of an arrow
    :type panel: Panel
    :param line: line associated with an arrow (if any)
    :type line: Line
    :param contour: contour af an arrow
    :type contour: np.ndarray"""

    def __init__(self, panel, line=None, contour=None):
        # if not all(isinstance(pixel, Point) for pixel in pixels):
            # self.pixels = [Point(row=coords[0], col=coords[1]) for coords in pixels]
        # else:

        self.panel = panel
        self.pixels = panel.pixels

        self.line = line
        self.contour = contour
        # slope = self.line.slope
        self._center_px = None
        self.reference_pt = self.compute_reaction_reference_pt()
        self.initialize()
        self.children = []  # set dynamically

    @property
    def conditions(self):
        return self.children
    
    def __repr__(self):
        return f'{self.__class__.__name__}(panel={self.panel})'

    def __str__(self):
        top, left, bottom, right = self.panel
        return f'{self.__class__.__name__}({top, left, bottom, right})'

    def merge_children(self):
        """Merges child regions if they're close together"""
        # TODO: This is not the exact solution, consider when i-th region gets merged with n > 1 regions
        new_children = []
        unmerged_idx = list(range(len(self.children)))
        for i in range(len(self.children)):
            for j in range(i+1, len(self.children)):
                if self.children[i].panel.edge_separation(self.children[j].panel) < 150 and \
                        self.children[i].arrow == self.children[j].arrow:
                    unmerged_idx.remove(i)
                    unmerged_idx.remove(j)
                    new_children.append(self.children[i].merge_conditions_regions(self.children[j]))
        for i in unmerged_idx:
            new_children.append(self.children[i])

    def initialize(self):
        """Given `pixels` and `panel` attributes, this method checks if other (relevant) initialization attributes
        have been precomputed. If not, these should be computed and set accordingly."""
        # if self.line is None:
            # self.line = Line.approximate_line(self.pixels[0], self.pixels[-1])

        if self.contour is None:
            img = np.zeros_like(self.panel.fig.img)
            img[self.panel.pixels] = 255
            cnt, _ = cv2.findContours(img,
                                      ExtractorConfig.CURLY_ARROW_CNT_MODE, ExtractorConfig.CURLY_ARROW_CNT_METHOD)
            assert len(cnt) <=2
            self.contour = cnt

    def compute_reaction_reference_pt(self):
        """Computes a reference point for a reaction step. This point alongside arrow's center point is used to decide
        whether a diagram belongs to reactants or products of a step (by comparing pairwise distances).
        This reference point is a centre of mass in an eroded arrow crop (erosion further moves the original centre of
        mass away from the center point to facilitate comparison
        return: row, col coordinates of the centre of mass of the eroded arrow
        rtype: tuple"""
        ''
        scaling_factor = 2
        pad_width = 10
        img = np.zeros_like(self.panel.fig.img)
        img[self.panel.pixels] = 255
        crop = self.panel.create_padded_crop(Figure(img, img), pad_width).img
        crop = cv2.resize(crop, (0, 0), fx=scaling_factor, fy=scaling_factor)
        eroded = cv2.erode(crop, np.ones((3, 3)), iterations=2)

        #Compute COM in the crop, then transform back to main figure coordinates
        rows, cols = np.nonzero(eroded>200)
        rows, cols = rows/scaling_factor - pad_width, cols/scaling_factor - pad_width
        row, col = int(np.mean(rows)), int(np.mean(cols))
        top, left = self.panel.top, self.panel.left
        row, col = row + top, col + left        
        return col, row  # x, y

    # def sort_pixels(self):
    #     """
    #     Simple pixel sort.
    #     Sorts pixels by column in all arrows.
    #     :return:
    #     """
    #     self.pixels.sort(key=lambda pixel: pixel.col)



class SolidArrow(BaseArrow):
    """
    Class used to represent simple solid reaction arrows.
    :param pixels: pixels forming the arrows
    :type pixels: list[Point]
    :param line: line found by Hough transform, underlying primitive,
    :type line: Line
    :param panel: bounding box of an arrow
    :type panel: Panel"""

    def __init__(self, panel, line=None, contour=None):

        self.line = line
        self.contour = contour
        # self.react_side = None
        # self.prod_side = None
        super(SolidArrow, self).__init__(panel)
        # self.sort_pixels()

    @property
    def is_vertical(self):
        return self.line.is_vertical

    @property
    def slope(self):
        return self.line.slope



    def __eq__(self, other):
        if not isinstance(other, BaseArrow):
            return False
        return self.panel == other.panel

    def __hash__(self):
        return hash(pixel for pixel in self.pixels)

    # def sort_pixels(self):
    #     """
    #     Simple pixel sort.
    #     Sorts pixels by row in vertical arrows and by column in all other arrows
    #     :return:
    #     """
    #     if self.is_vertical:
    #         self.pixels.sort(key=lambda pixel: pixel.row)
    #     else:
    #         self.pixels.sort(key=lambda pixel: pixel.col)


class CurlyArrow(BaseArrow):

    def __init__(self, panel, contour=None, line=None):
        """Class used to represent curly arrows. Does not make use of the ``line`` attribute,
        and overrides the ``initialize`` method to account for this"""
        self.contour = contour
        super().__init__(panel)
        self.line = None

    # def initialize(self):
    #     if self.contour is None:
    #         isolated_arrow_fig = Isolator(None, self, isolate_mask=True).process()
    #         cnt, _ = cv2.findContours(isolated_arrow_fig.img,
    #                                   ExtractorConfig.CURLY_ARROW_CNT_MODE, ExtractorConfig.CURLY_ARROW_CNT_METHOD)
    #         assert len(cnt) == 1
    #         self.contour = cnt[0]


class ResonanceArrow(BaseArrow):
    """Class used to represent resonance arrows"""
    def __init__(self, panel, line=None, contour=None):

        self.line = line
        self.contour = contour
        super().__init__(panel)
        # self.sort_pixels()

    # def initialize(self):
    #     if self.line is None:
    #         pass
    #     if self.contour is None:
    #         pass


class EquilibriumArrow(BaseArrow):
    """Class used to represent equilibrium arrows"""
    def __init__(self, panel, line=None, contour=None):

        self.line = line
        self.contour = contour
        super().__init__(panel)
        # self.sort_pixels()

    # def initialize(self):
    #     if self.line is None:
    #         pass
    #     if self.contour is None:
    #         pass
