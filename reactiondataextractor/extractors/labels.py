# -*- coding: utf-8 -*-
"""
Labels
======

This module contains LabelExtractor and methods concerning labels assignment, as well as other classes
for RGroup resolution

author: Damian Wilary
email: dmw51@cam.ac.uk

"""
import logging
import string
from typing import List, Tuple

from matplotlib.patches import Rectangle
import re

from reactiondataextractor.models.base import BaseExtractor
from reactiondataextractor.ocr import ASSIGNMENT, SEPARATORS, CONCENTRATION, LABEL_WHITELIST, img_to_text
from reactiondataextractor.models.reaction import Label, LabelType

log = logging.getLogger('extract.labels')


class LabelExtractor(BaseExtractor):
    """This class is responsible for extracting information from detected labels
    :param priors: label detection from the postprocessing routine
    :param ocr_fig: the reaction scheme figure cleaned from arrows and diagrams; set during text detection postprocessing"""
    
    def __init__(self, fig: 'Figure', priors: List['Panel']):
        
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

    def read_label(self, label_candidate: 'TextRegionCandidate') -> 'Label':
        """Recognises and reads the label. Assigns the type to a candidate for later use

        :param label_candidate: label candidate for OCR processing
        :type label_candidate: TextRegionCandidate
        :return: label with recognised text and assigned type
        :rtype: Label
        """
        crop = label_candidate.panel.create_crop(self.ocr_fig)
        text = img_to_text(crop.img, whitelist=LABEL_WHITELIST)
        label = Label(text=text, **label_candidate.pass_attributes())
        if label.type == LabelType.VARIANTS:
            label.root, label.variant_indicators = self.infer_variant_indicators(label.text[0])

        elif label.type == LabelType.SIMPLE:
            label.root, label.variant_indicators = label.text[0], None
        return label

    def infer_variant_indicators(self, label_text: str) -> Tuple[str]:
        """Infers the range of variants in a labels. Supported variants are: n-m, where n,m are integers, nx-y where n
        is an integer, and x and y are characters, x-y where x, y are characters
        :param label_text: text returned by the OCR engine
        :type label_text: str
        """
        parts = label_text.split('-')
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
