# -*- coding: utf-8 -*-
"""
Labels
======

This module contains LabelExtractor and methods concerning labels assignment, as well as other classes
for RGroup resolution

author: Damian Wilary
email: dmw51@cam.ac.uk

Code snippets for merging loops and RGroup and RGroupResolver taken from chemschematicresolver (MIT licence) by Edward
Beard (ejb207@cam.ac.uk)

"""
import csv
import logging
import string

from matplotlib.patches import Rectangle
import os
import re
# from urllib.error import URLError

# import cirpy

from models.segments import PanelMethodsMixin, Panel
from reactiondataextractor.models.base import BaseExtractor, TextRegion
from reactiondataextractor.ocr import ASSIGNMENT, SEPARATORS, CONCENTRATION, LABEL_WHITELIST, img_to_text
from reactiondataextractor.models.reaction import Label, LabelType

log = logging.getLogger('extract.labels')

# BLACKLIST_CHARS = ASSIGNMENT + SEPARATORS + CONCENTRATION
#
# # Regular Expressions
# NUMERIC_REGEX = re.compile('^\d{1,3}$')
# ALPHANUMERIC_REGEX = re.compile('^((d-)?(\d{1,2}[A-Za-z]{1,2}[′″‴‶‷⁗]?)(-d))|(\d{1,3})?$')
#
# # Commonly occuring tokens for R-Groups:
# r_group_indicators = ['R', 'X', 'Y', 'Z', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'Y2', 'D', "R'",
#                       "R''", "R'''", "R''''"]
# r_group_indicators = r_group_indicators + [val.lower() for val in r_group_indicators]

# # Standard path to superatom dictionary file
# parent_dir = os.path.dirname(os.path.abspath(__file__))
# superatom_file = os.path.join(parent_dir, '..', 'dict', 'superatom.txt')
# spelling_file = os.path.join(parent_dir, '..', 'dict', 'spelling.txt')





class LabelExtractor(BaseExtractor):
    """This class is responsible for extracting information from detected labels"""

    def __init__(self, fig, priors):
        super().__init__(fig)
        self.priors = priors
        self.ocr_fig = None

    def extract(self):
        """Main extraction method"""
        labels = [self.read_label(cand) for cand in self.priors]
        self._extracted = labels
        return self.extracted

    @property
    def extracted(self):
        """Returns extracted objects"""
        return self._extracted

    def plot_extracted(self, ax):
        """Adds extracted panels onto a canvas of ``ax``"""
        params = {'facecolor': (66 / 255, 93 / 255, 166 / 255),
                  'edgecolor': (6 / 255, 33 / 255, 106 / 255),
                  'alpha': 0.4}
        for label in self._extracted:
                panel = label.panel
                rect_bbox = Rectangle((panel.left - 1, panel.top - 1), panel.right - panel.left,
                                      panel.bottom - panel.top, **params)
                ax.add_patch(rect_bbox)

    def read_label(self, label_candidate):
        #TODO: Same thing as for conditions, can use the same figure actually
        crop = label_candidate.panel.create_crop(self.ocr_fig)
        text = img_to_text(crop.img, whitelist=LABEL_WHITELIST)
        label = Label(text=text, **label_candidate.pass_attributes())
        if label.type == LabelType.VARIANTS:
            label.root, label.variant_indicators = self.infer_variant_indicators(label.text[0])

        elif label.type == LabelType.SIMPLE:
            label.root, label.variant_indicators = label.text[0], None
        return label

    def infer_variant_indicators(self, label_text, sep='-'):
        """Infers the range of variants in a labels. Supported variants are: n-m, where n,m are integers, nx-y where n
        is an integer, and x and y are characters, x-y where x, y are characters"""
        parts = label_text.split(sep)
        assert len(parts) == 2
        digits_only_right = all(char in string.digits for char in parts[1])
        # assert digits_only_right
        digits_only_left = all(char in string.digits for char in parts[0])
        if digits_only_left and digits_only_right:
            return ['', list(map(str, range(int(parts[0], int(parts[1])+1))))]

        assert all(char in string.ascii_letters for char in parts[1])
        chars_only_left = all(char in string.ascii_letters for char in parts[0])
        if not chars_only_left:
            left = re.match(r'(\d+)([A-Za-z]+)', parts[0])
            label_root, variant_start = left.group(1), left.group(2)
        else:
            label_root, variant_start = '', parts[0]

        idx_left = string.ascii_letters.index(variant_start)
        idx_right = string.ascii_letters.index(parts[1])
        return label_root, string.ascii_letters[idx_left:idx_right+1]



# class Label(TextRegion, PanelMethodsMixin):
#     """Describes labels and recognised text

#     :param panel: bounding box a labels
#     :type panel: Panel
#     :param text: labels text
#     :type text: str
#     :param r_group: generic r_groups associated with a labels
#     :type r_group: str"""

#     @classmethod
#     def from_coords(cls, left, right, top, bottom, text):
#         panel = Panel((left, right, top, bottom))
#         return cls(panel, text)

#     def __init__(self, panel, text, r_groups=None, *, _prior_class=None):
#         if r_groups is None:
#             r_groups = []
#         # if text is None:
#         #     text = []
#         self.panel = panel
#         self._text = text
#         self.r_groups = r_groups
#         self._prior_class = _prior_class
#         if self.text:
#             self.type = LabelType.assign_type(self.text[0])
#         else:
#             self.type = LabelType.UNKNOWN
#         self.root = None
#         self.variant_indicators = None
#         self.resolve_variants = False
#         # self._parent_panel = parent_panel

#     # @property
#     # def diagram(self):
#     #     return self._parent_panel

#     @property
#     def text(self):
#         return self._text

#     @text.setter
#     def text(self, value):
#         self._text = value

#     def __repr__(self):
#         return f'Label(panel={self.panel}, text={self.text}, r_groups={self.r_groups})'

#     def __str__(self):
#         return f'Label(Text: {", ".join(sent for sent in self.text)})'

#     def __hash__(self):
#         return hash(self.panel)

#     def __eq__(self, other):
#         return isinstance(other, Label) and self.panel == other.panel


#     def add_r_group_variables(self, var_value_label_tuples):
#         """ Updates the R-groups for this labels."""

#         self.r_group.append(var_value_label_tuples)

#     def is_similar_to(self, other_label):
#         text = ' '.join(self.text)
#         other_text = ' '.join(other_label.text)
#         chemical_number_self = re.match(r'\d+', text).group(0)
#         chemical_number_other = re.match(r'\d+', other_text).group(0)
#         if chemical_number_other == chemical_number_self:
#             return True
#         return False

