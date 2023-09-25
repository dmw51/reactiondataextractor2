"""This module contains classes and routines associated with manipulating smiles, including resolving R-groups into
individual chemical compounds"""
from typing import List

from utils.utils import erase_elements, euclidean_distance
from utils.vectorised import DiagramVectoriser

from reactiondataextractor.models import BaseExtractor

class SmilesExtractor(BaseExtractor):
    """Wrapper class. Wraps the main Optical Chemical Structure Recognition class.
    
    :param diagrams: List of detected chemical diagrams
    :type diagrams: List[Diagram]
    :param recogniser: instance of the wrapped OCSR class that performs the recognition
    :type recogniser: DecimerRecogniser
    :param vectoriser: instance of a vectorisation class"""
    def __init__(self, diagrams: List['Diagram'], recogniser: 'DecimerRecogniser'):
        self.diagrams = diagrams
        self.recogniser = recogniser
        self.vectoriser = DiagramVectoriser()

    def extract(self) -> None:
        """This method is a wrapper method that call the OCSR engine"""
        print('Running OCSR engine...')
        for diag in self.diagrams:
            self.vectoriser.diag = diag
            # self.vectoriser.create_vectorised_diagram_graph()
            self.recognise(diag)

    def recognise(self, diag) -> None:
        """This is the recognition method. Image patch corresponding to the input diagram 
        is preprocessed, fed through the model, and detokenised to give SMILES representation. 
        Updates `smiles` attribute of the diagram.
        :param diag: input chemical diagram for optical recognition
        :type diag: Diagram
        """
        crop = diag.crop
        lone_groups = []

        chemical_structure = self.recogniser.decode_image(crop.img_detectron)
        predicted_tokens = self.recogniser.model(chemical_structure)
        predicted_SMILES = self.recogniser.detokenize_output(predicted_tokens)
        diag.smiles = predicted_SMILES
