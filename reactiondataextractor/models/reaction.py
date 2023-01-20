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

import logging
from typing import Sequence

import cv2
import numpy as np

from configs import ExtractorConfig
from processors import Isolator, Binariser
from .base import TextRegion, Candidate
from .geometry import Point, Line
from .segments import Panel, PanelMethodsMixin

log = logging.getLogger('extract.reaction')


class BaseReactionClass(object):
    """
    This is a base.py reaction class placeholder
    """


class Diagram(BaseReactionClass, PanelMethodsMixin):
    """This is a base.py class for chemical structures species found in diagrams (e.g. reactants and products)

    :param panel: bounding box a diagrams
    :type panel: Panel
    :param label: label associated with a diagram
    :type label: Label
    :param smiles: SMILES associated with a diagram
    :type smiles: str
    :param crop: crop containing the diagram
    :type crop: Crop"""

    @classmethod
    def from_coords(cls, left, right, top, bottom, label=None, smiles=None, crop=None):
        """Class method used for instantiation from coordinates, as used within chemschematicresolver"""
        panel = Panel((left, right, top, bottom) )
        return cls(panel=panel, label=label, smiles=smiles, crop=crop)

    def __init__(self, panel, label=None, smiles=None, crop=None):
        self._panel = panel
        self._label = label
        self._smiles = smiles
        self._crop = crop
        self.children = []
        self.reaction_steps = []
        super().__init__()

    def __eq__(self, other):
        if isinstance(other, Diagram):  # Only compare exact same types
            return self.panel == other.panel
        return False

    def __hash__(self):
        return hash(self.panel)

    def __repr__(self):
        return f'{self.__class__.__name__}(panel={self.panel}, smiles={self.smiles}, label={self.label})'

    def __str__(self):
        return f'{self.smiles if self.smiles else "???"}, label: {self.label}'

    @property
    def labels(self):
        return self.children

    @property
    def panel(self):
        return self._panel

    @property
    def center(self):
        return self._panel.center

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def smiles(self):
        return self._smiles

    @smiles.setter
    def smiles(self, smiles):
        self._smiles = smiles

    @property
    def crop(self):
        """ Cropped Figure object of the specific diagram"""
        return self._crop

    @crop.setter
    def crop(self, fig):
        self._crop = fig

    # def compass_position(self, other):
    #     """ Determines the compass position (NSEW) of other relative to self"""
    #
    #     length = other.center[0] - self.center[0]
    #     height = other.center[1] - self.center[1]
    #
    #     if abs(length) > abs(height):
    #         if length > 0:
    #             return 'E'
    #         else:
    #             return 'W'
    #     elif abs(length) < abs(height):
    #         if height > 0:
    #             return 'S'
    #         else:
    #             return 'N'
    #
    #     else:
    #         return None


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

    :param panel: bounding box a label
    :type panel: Panel
    :param text: label text
    :type text: str
    :param r_group: generic r_groups associated with a label
    :type r_group: str"""

    @classmethod
    def from_coords(cls, left, right, top, bottom, text):
        panel = Panel((left, right, top, bottom))
        return cls(panel, text)

    def __init__(self, panel, text=None, r_group=None, *, _prior_class=None):
        if r_group is None:
            r_group = []
        if text is None:
            text = []
        self.panel = panel
        self._text = text
        self.r_group = r_group
        self._prior_class = _prior_class
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
        return f'Label(panel={self.panel}, text={self.text}, r_group={self.r_group})'

    def __str__(self):
        return f'Label(Text: {", ".join(sent for sent in self.text)})'

    def __hash__(self):
        return hash(self.panel)

    def __eq__(self, other):
        return isinstance(other, Label) and self.panel == other.panel


    def add_r_group_variables(self, var_value_label_tuples):
        """ Updates the R-groups for this label."""

        self.r_group.append(var_value_label_tuples)


# class BaseArrow(PanelMethodsMixin):
#     """Base arrow class common to all arrows
#
#     :param pixels: pixels forming the arrows
#     :type pixels: list[Point]
#     :param line: line found by Hough transform, underlying primitive,
#     :type line: Line
#     :param panel: bounding box of an arrow
#     :type panel: Panel"""
#
#     def __init__(self, pixels, line, panel):
#         if not all(isinstance(pixel, Point) for pixel in pixels):
#             self.pixels = [Point(row=coords[0], col=coords[1]) for coords in pixels]
#         else:
#             self.pixels = pixels
#
#         self.line = line
#         self._panel = panel
#         slope = self.line.slope
#         self.sort_pixels()
#         self._center_px = None
#
#     @property
#     def panel(self):
#         return self._panel
#
#     @property
#     def is_vertical(self):
#         return self.line.is_vertical
#
#     @property
#     def center_px(self):
#         """
#         Based on a geometric centre of an arrow panel, looks for a pixel nearby that belongs to the arrow.
#
#         :return: coordinates of the pixel that is closest to geometric centre and belongs to the object.
#         If multiple pairs found, return the floor average.
#         :rtype: Point
#         """
#         if self._center_px is not None:
#             return self._center_px
#
#         log.debug('Finding center of an arrow...')
#         x, y = self.panel.center
#
#         log.debug('Found an arrow with geometric center at (%s, %s)' % (y, x))
#
#         # Look at pixels neighbouring center to check which actually belong to the arrow
#         x_candidates = [x+i for i in range(-2, 3)]
#         y_candidates = [y+i for i in range(-2, 3)]
#         center_candidates = [candidate for candidate in product(x_candidates, y_candidates) if
#                              Point(row=candidate[1], col=candidate[0]) in self.pixels]
#
#         log.debug('Possible center pixels: %s', center_candidates)
#         if center_candidates:
#             self._center_px = np.mean(center_candidates, axis=0, dtype=int)
#             self._center_px = Point(row=self._center_px[1], col=self._center_px[0])
#         else:
#             raise NotAnArrowException('No component pixel lies on the geometric centre')
#         log.debug('Center pixel found: %s' % self._center_px)
#
#         return self._center_px
#
#     def sort_pixels(self):
#         """
#         Simple pixel sort.
#
#         Sorts pixels by row in vertical arrows and by column in all other arrows
#         :return:
#         """
#         if self.is_vertical:
#             self.pixels.sort(key=lambda pixel: pixel.row)
#         else:
#             self.pixels.sort(key=lambda pixel: pixel.col)
#
#
# class SolidArrow(BaseArrow):
#     """
#     Class used to represent simple reaction arrows.
#
#     :param pixels: pixels forming the arrows
#     :type pixels: list[Point]
#     :param line: line found by Hough transform, underlying primitive,
#     :type line: Line
#     :param panel: bounding box of an arrow
#     :type panel: Panel"""
#
#     def __init__(self, pixels, line, panel):
#         super(SolidArrow, self).__init__(pixels, line, panel)
#         self.react_side = None
#         self.prod_side = None
#         a_ratio = self.panel.aspect_ratio
#         a_ratio = 1/a_ratio if a_ratio < 1 else a_ratio
#         if a_ratio < 3:
#             raise NotAnArrowException('aspect ratio is not within the accepted range')
#
#         self.react_side, self.prod_side = self.get_direction()
#         pixel_majority = len(self.prod_side) - len(self.react_side)
#         num_pixels = len(self.pixels)
#         min_pixels = min(int(0.02 * num_pixels), 15)
#         if pixel_majority < min_pixels:
#             raise NotAnArrowException('insufficient pixel majority')
#         elif pixel_majority < 2 * min_pixels:
#             log.warning('Difficulty detecting arrow sides - low pixel majority')
#
#         log.debug('Arrow accepted!')
#
#     def __repr__(self):
#         return f'SolidArrow(pixels={self.pixels}, line={self.line}, panel={self.panel})'
#
#     def __str__(self):
#         left, right, top, bottom = self.panel
#         return f'SolidArrow({left, right, top, bottom})'
#
#     def __eq__(self, other):
#         return self.panel == other.panel
#
#     def __hash__(self):
#         return hash(pixel for pixel in self.pixels)
#
#     @property
#     def hook(self):
#         """
#         Returns the last pixel of an arrow hook.
#         :return:
#         """
#         if self.is_vertical:
#             prod_side_lhs = True if self.prod_side[0].row < self.react_side[0].row else False
#         else:
#             prod_side_lhs = True if self.prod_side[0].col < self.react_side[0].col else False
#         return self.prod_side[0] if prod_side_lhs else self.prod_side[-1]
#
#     def get_direction(self):
#         """Retrieves the direction of an arrow by looking at the number of pixels on each side.
#
#         Splits an arrow in the middle depending on its slope and calculated the number of pixels in each part."""
#         center = self.center_px
#         # center = Point(center[1], center[0])
#         if self.is_vertical:
#             part_1 = [pixel for pixel in self.pixels if pixel.row < center.row]
#             part_2 = [pixel for pixel in self.pixels if pixel.row > center.row]
#
#         elif self.line.slope == 0:
#             part_1 = [pixel for pixel in self.pixels if pixel.col < center.col]
#             part_2 = [pixel for pixel in self.pixels if pixel.col > center.col]
#
#         else:
#             p_slope = -1/self.line.slope
#             p_intercept = center.row - center.col*p_slope
#             p_line = lambda point: point.col*p_slope + p_intercept
#             part_1 = [pixel for pixel in self.pixels if pixel.row < p_line(pixel)]
#             part_2 = [pixel for pixel in self.pixels if pixel.row > p_line(pixel)]
#
#         if len(part_1) > len(part_2):
#             react_side = part_2
#             prod_side = part_1
#         else:
#             react_side = part_1
#             prod_side = part_2
#
#         log.debug('Established reactant and product sides of an arrow.')
#         log.debug('Number of pixel on reactants side: %s ', len(react_side))
#         log.debug('product side: %s ', len(prod_side))
#         return react_side, prod_side

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

    def __init__(self, pixels, panel, line=None, contour=None):
        if not all(isinstance(pixel, Point) for pixel in pixels):
            self.pixels = [Point(row=coords[0], col=coords[1]) for coords in pixels]
        else:
            self.pixels = pixels

        self.panel = panel
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
        if self.line is None:
            self.line = Line.approximate_line(self.pixels[0], self.pixels[-1])

        if self.contour is None:
            isolated_arrow_fig = Isolator(None, self, isolate_mask=True).process()
            cnt, _ = cv2.findContours(isolated_arrow_fig.img,
                                      ExtractorConfig.CURLY_ARROW_CNT_MODE, ExtractorConfig.CURLY_ARROW_CNT_METHOD)
            assert len(cnt) == 1
            self.contour = cnt[0]

    def compute_reaction_reference_pt(self):
        """Computes a reference point for a reaction step. This point alongside arrow's center point is used to decide
        whether a diagram belongs to reactants or products of a step (by comparing pairwise distances).
        This reference point is a centre of mass in an eroded arrow crop (erosion further moves the original centre of
        mass away from the center point to facilitate comparison
        return: row, col coordinates of the centre of mass of the eroded arrow
        rtype: tuple"""
        scaling_factor = 2
        pad_width = 10
        isolated_arrow = Isolator(None, self, isolate_mask=True).process()
        arrow_crop = self.panel.create_padded_crop(isolated_arrow, pad_width=pad_width)
        arrow_crop.img = cv2.resize(arrow_crop.img, (0, 0), fx=scaling_factor, fy=scaling_factor)
        binarised = Binariser(arrow_crop).process()
        eroded = cv2.erode(binarised.img, np.ones((6, 6)), iterations=2)

        #Compute COM in the crop, then transform back to main figure coordinates
        rows, cols = np.where(eroded == 255)
        rows, cols = rows/scaling_factor - pad_width, cols/scaling_factor - pad_width
        row, col = int(np.mean(rows)), int(np.mean(cols))
        row, col = arrow_crop.in_main_fig((row, col))

        return col, row  # x, y

    # @property
    # def center_px(self):
    #     """
    #     Based on a geometric centre of an arrow panel, looks for a pixel nearby that belongs to the arrow.
    #     :return: coordinates of the pixel that is closest to geometric centre and belongs to the object.
    #     If multiple pairs found, return the floor average.
    #     :rtype: Point
    #     """
    #     if self._center_px is not None:
    #         return self._center_px
    #
    #     log.debug('Finding center of an arrow...')
    #     x, y = self.panel.geometric_centre
    #
    #     log.debug('Found an arrow with geometric center at (%s, %s)' % (y, x))
    #
    #     # Look at pixels neighbouring center to check which actually belong to the arrow
    #     x_candidates = [x+i for i in range(-3, 4)]
    #     y_candidates = [y+i for i in range(-3, 4)]
    #     center_candidates = [candidate for candidate in product(x_candidates, y_candidates) if
    #                          Point(row=candidate[1], col=candidate[0]) in self.pixels]
    #
    #     log.debug('Possible center pixels: %s', center_candidates)
    #     if center_candidates:
    #         self._center_px = np.mean(center_candidates, axis=0, dtype=int)
    #         self._center_px = Point(row=self._center_px[1], col=self._center_px[0])
    #     else:
    #         raise NotAnArrowException('No component pixel lies on the geometric centre')
    #     log.debug('Center pixel found: %s' % self._center_px)
    #
    #     return self._center_px

    def sort_pixels(self):
        """
        Simple pixel sort.
        Sorts pixels by column in all arrows.
        :return:
        """
        self.pixels.sort(key=lambda pixel: pixel.col)



class SolidArrow(BaseArrow):
    """
    Class used to represent simple solid reaction arrows.
    :param pixels: pixels forming the arrows
    :type pixels: list[Point]
    :param line: line found by Hough transform, underlying primitive,
    :type line: Line
    :param panel: bounding box of an arrow
    :type panel: Panel"""

    def __init__(self, pixels, panel, line=None, contour=None):

        self.line = line
        self.contour = contour
        # self.react_side = None
        # self.prod_side = None
        super(SolidArrow, self).__init__(pixels, panel)
        self.sort_pixels()

    # def initialize(self):
    #     if self.line is None:
    #         self.line = Line.approximate_line(self.pixels[0], self.pixels[-1])
    #
    #     if self.contour is None:
    #         isolated_arrow_fig = Isolator(None, self, isolate_mask=True).process()
    #         cnt, _ = cv2.findContours(isolated_arrow_fig.img,
    #                                   ExtractorConfig.CURLY_ARROW_CNT_MODE, ExtractorConfig.CURLY_ARROW_CNT_METHOD)
    #         assert len(cnt) == 1
    #         self.contour = cnt[0]

    @property
    def is_vertical(self):
        return self.line.is_vertical

    @property
    def slope(self):
        return self.line.slope

    def __repr__(self):
        return f'SolidArrow(pixels={self.pixels[:5]},..., line={self.line}, panel={self.panel})'

    def __str__(self):
        top, left, bottom, right = self.panel
        return f'SolidArrow({top, left, bottom, right})'

    def __eq__(self, other):
        if not isinstance(other, BaseArrow):
            return False
        return self.panel == other.panel

    def __hash__(self):
        return hash(pixel for pixel in self.pixels)

    def sort_pixels(self):
        """
        Simple pixel sort.
        Sorts pixels by row in vertical arrows and by column in all other arrows
        :return:
        """
        if self.is_vertical:
            self.pixels.sort(key=lambda pixel: pixel.row)
        else:
            self.pixels.sort(key=lambda pixel: pixel.col)


class CurlyArrow(BaseArrow):

    def __init__(self, pixels, panel, contour=None, line=None):
        """Class used to represent curly arrows. Does not make use of the ``line`` attribute,
        and overrides the ``initialize`` method to account for this"""
        self.contour = contour
        super().__init__(pixels, panel)
        self.line = None

    def initialize(self):
        if self.contour is None:
            isolated_arrow_fig = Isolator(None, self, isolate_mask=True).process()
            cnt, _ = cv2.findContours(isolated_arrow_fig.img,
                                      ExtractorConfig.CURLY_ARROW_CNT_MODE, ExtractorConfig.CURLY_ARROW_CNT_METHOD)
            assert len(cnt) == 1
            self.contour = cnt[0]


class ResonanceArrow(BaseArrow):
    """Class used to represent resonance arrows"""
    def __init__(self, pixels, panel, line=None, contour=None):

        self.line = line
        self.contour = contour
        super().__init__(pixels, panel)
        self.sort_pixels()

    # def initialize(self):
    #     if self.line is None:
    #         pass
    #     if self.contour is None:
    #         pass


class EquilibriumArrow(BaseArrow):
    """Class used to represent equilibrium arrows"""
    def __init__(self, pixels, panel, line=None, contour=None):

        self.line = line
        self.contour = contour
        super().__init__(pixels, panel)
        self.sort_pixels()

    # def initialize(self):
    #     if self.line is None:
    #         pass
    #     if self.contour is None:
    #         pass


class ArrowCandidate(Candidate):
    """A class to store any attributes that have been computed in the arrow proposal stage. Acts as a cache of values
    which can be reused when an arrow candidate is accepted. All instances are required to have a `pixels` attribute,
    which is used to isolate the relevant connected component prior to arrow detection stage"""

    def __init__(self, pixels, panel=None, *, line=None, contour=None):
        self.pixels = np.array(pixels)
        self.panel = panel
        self.line = line
        self.contour = contour
