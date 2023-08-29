"""This module contains classes and routines associated with manipulating smiles, including resolving R-groups into
individual chemical compounds"""
import copy
import os
import re
from collections import Counter
from itertools import product

import numpy as np


from utils.utils import erase_elements, euclidean_distance
from utils.vectorised import DiagramVectoriser

from reactiondataextractor.models import BaseExtractor
from reactiondataextractor.models.reaction import ResolvedDiagram
# parent_dir = os.path.dirname(os.path.abspath(__file__))
# r_group_correct_file = os.path.join(parent_dir, '..', 'dict', 'r_placeholders.txt')

class SmilesExtractor(BaseExtractor):
    def __init__(self, diagrams, recogniser):
        self.diagrams = diagrams
        self.recogniser = recogniser
        self.vectoriser = DiagramVectoriser()
        # self._corners = None
        # self._adjacency_matrix = None


    def extract(self):
        """Takes in diagrams, uses the recogniser, then resolves R groups into individual smiles """
        for diag in self.diagrams:
            self.vectoriser.diag = diag
            self.vectoriser.create_vectorised_diagram_graph()
            self.recognise(diag)

    def recognise(self, diag):
        crop = diag.crop
        lone_groups = []

        chemical_structure = self.recogniser.decode_image(crop.img_detectron)
        predicted_tokens = self.recogniser.model(chemical_structure)
        predicted_SMILES = self.recogniser.detokenize_output(predicted_tokens)
        diag.smiles = predicted_SMILES

