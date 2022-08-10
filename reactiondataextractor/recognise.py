# -*- coding: utf-8 -*-
"""
Recognise
========

This module contains optical chemical structure recognition tools and routines.

author: Damian Wilary
email: dmw51@cam.ac.uk

Recognition is achieved using OSRA and performed via a pyOsra wrapper.
"""
import os
import itertools
import logging
from PIL import Image

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import efficientnet.tfkeras as efn

from DECIMER.config import get_bnw_image, delete_empty_borders, central_square_image, PIL_im_to_BytesIO
from DECIMER.decimer import tokenizer, DECIMER_V2

# from DECIMER.decimer import load_trained_model, evaluate, decoder
from models.reaction import Diagram
from reactiondataextractor.models.segments import FigureRoleEnum, Figure
from reactiondataextractor.utils import isolate_patches

log = logging.getLogger()
# superatom_file = os.path.join(settings.ROOT_DIR, 'dict', 'superatom.txt')
# spelling_file = os.path.join(settings.ROOT_DIR,  'dict', 'spelling.txt')
# superatom_file = ''
# spelling_file = ''


class DecimerRecogniser:
    def __init__(self, model_id='Canonical'):
        assert model_id.capitalize() in ['Canonical', 'Isomeric', 'Augmented'], "model_id has to be one of the following:\
                                                                            ['Canonical', 'Isomeric', 'Augmented']"
        # self.model_id = model_id
        # self.feature_extractor, self.transformer, self.max_length, self.SELFIES_tokenizer = load_trained_model(model_id)
        # self.temp_path = f'diag_temp.png'
        self.model = DECIMER_V2

    def predict_SMILES(self, fig: Figure, diagram: Diagram) -> str:
        """
        This function takes a figure and a diagram inside it, and returns the SMILES
        representation of the depicted molecule (str).
        Args:
            fig (Figure): Analysed reaction scheme figure
            diagram (Diagram): diagram inside the figure
        Returns:
            (str): SMILES representation of the molecule in the input image
        """
        diag_crop = diagram.panel.create_crop(fig)
        chemical_structure = self.decode_image(diag_crop.img_detectron)
        predicted_tokens = self.model(chemical_structure)
        predicted_SMILES = self.detokenize_output(predicted_tokens)
        diagram.smiles = predicted_SMILES

        return predicted_SMILES
    # def _recognise_diagram(self, fig, diagram):
    #     desired_h_and_w = 299
    #     w = diagram.panel.width
    #     h = diagram.panel.height
    #     pad_h = (desired_h_and_w - h)//2
    #     pad_w = (desired_h_and_w - w)//2
    #     if pad_h <= 0 or pad_w <= 0: # img is larger than the desired size
    #         diag_crop = diagram.panel.create_crop(fig)
    #         diag_crop = isolate_patches(diag_crop, [cc for cc in diag_crop.connected_components
    #                                                 if cc.role == FigureRoleEnum.DIAGRAMPART])
    #         img = cv2.resize(diag_crop.img, (desired_h_and_w, desired_h_and_w))
    #     else:
    #         pad_width = (pad_h, pad_h), (pad_w, pad_w)
    #         diag_crop = diagram.panel.create_padded_crop(fig, pad_width)
    #         diag_crop = isolate_patches(diag_crop, [cc for cc in diag_crop.connected_components
    #                                         if cc.role in [FigureRoleEnum.DIAGRAMPART, FigureRoleEnum.DIAGRAMPRIOR]])
    #         img = diag_crop.img
    #
    #     plt.imsave(self.temp_path, img, cmap='binary')
    #     smiles = self.predict_SMILES(self.temp_path)
    #     diagram.smiles = smiles
    #     os.remove(self.temp_path)
    #     return smiles

    # def recognise_diagrams(self, fig, diagrams):
    #     preprocessed = [self.preprocess_diagram(diag) for diag in diagrams]
    #     temp_files = [plt.imsave(self.temp_paths(i), diag) for i, diag in zip(range(len(preprocessed)), preprocessed)]

    def decode_image(self, img: np.ndarray):
        """
        Loads and preprocesses an image
        Args:
            img (np.ndarray): image array
        Returns:
            Processed image
        """
        # img = self.remove_transparent(img)
        img = get_bnw_image(img)
        img = delete_empty_borders(img)
        img = central_square_image(img)
        img = PIL_im_to_BytesIO(img)
        img = tf.image.decode_png(img.getvalue(), channels=3)
        img = tf.image.resize(img, (299, 299))
        img = efn.preprocess_input(img)
        return img

    def detokenize_output(self, predicted_array: int) -> str:
        """
        This function takes the predited tokens from the DECIMER model
        and returns the decoded SMILES string.
        Args:
            predicted_array (int): Predicted tokens from DECIMER
        Returns:
            (str): SMILES representation of the molecule
        """
        outputs = [tokenizer.index_word[i] for i in predicted_array[0].numpy()]
        prediction = (
            "".join([str(elem) for elem in outputs])
            .replace("<start>", "")
            .replace("<end>", "")
        )

        return prediction
    # def remove_transparent(self, img: np.ndarray):
    #     """
    #     Removes the transparent layer from a PNG image with an alpha channel
    #     ___
    #     image_path (str): path of input image
    #     ___
    #     Output: PIL.Image
    #     """
    #     # png = Image.open(image_path).convert("RGBA")
    #     png = Image.fromarray(img, "RGBA")
    #     background = Image.new("RGBA", png.size, (255, 255, 255))
    #
    #     alpha_composite = Image.alpha_composite(background, png)

# class PyOsraRecogniser:
#     """Used to optical chemical structure recognition of diagrams
#
#     :param diagrams: extracted chemical diagrams
#     :type diagrams: list[Diagram]
#     :param allow_wildcards: whether to allow or discard partially recognised diagrams
#     :type allow_wildcards: bool"""
#
#     def __init__(self, diagrams, allow_wildcards=False):
#         self.diagrams = diagrams
#         self.allow_wildcards = allow_wildcards
#         self._tag_multiple_r_groups()
#
#     def recognise_diagrams(self):
#         """Main recognition method. Dispatches recognition to one of two routines depending on whether generic R-groups
#         were detected"""
#         for diag in self.diagrams:
#             if diag.r_groups:
#                 diag.smiles = self._get_rgroup_smiles(diag)
#             else:
#                 diag.smiles = self._read_diagram_pyosra(diag)
#
#     def _tag_multiple_r_groups(self):
#         for diag in self.diagrams:
#             if diag.label and diag.label.r_group and len(diag.label.r_group) > 1:
#                 diag.r_groups = True
#             else:
#                 diag.r_groups = False
#
#     def _get_rgroup_smiles(self, diag, extension='jpg', debug=False, superatom_path=superatom_file,
#                           spelling_path=spelling_file):
#         """ Extract SMILES from a chemical diagram (powered by pyosra)
#
#         :param diag: Input Diagram
#         :param extension: String indicating format of input file
#         :param debug: Bool to indicate debugging
#
#         :return labels_and_smiles: List of Tuple(List of label candidates, SMILES) objects
#         """
#         # Work around a dict bug which derails the process (OSRA-specific?)
#         contents = read_contents([superatom_path, spelling_path])
#
#
#         # Add some padding to image to help resolve characters on the edge
#         img = diag.crop.raw_img
#         if len(img.shape) == 3:
#             padded_img = pad(img, ((5, 5), (5, 5), (0, 0)), mode='constant', constant_values=1)
#         elif len(img.shape) == 2:
#             padded_img = pad(img, ((5, 5), (5, 5)), mode='constant', constant_values=1)
#
#
#         # Save a temp image
#         img_name = 'r_group_temp.' + extension
#         io_.imsave(img_name, padded_img)
#
#         osra_input = []
#         # label_cands = []
#
#         # Format the extracted rgroup
#         for tokens in diag.label.r_group:
#             token_dict = {}
#             for token in tokens:
#                 token_dict[token[0].text] = token[1].text
#
#             # Assigning var-var cases to true value if found (eg. R1=R2=H)
#             for a, b in itertools.combinations(token_dict.keys(), 2):
#                 if token_dict[a] == b:
#                     token_dict[a] = token_dict[b]
#
#             osra_input.append(token_dict)
#             # label_cands.append(tokens[0][2])
#
#         # Run osra on temp image
#         smiles = osra_rgroup.read_rgroup(osra_input, input_file=img_name, verbose=False, debug=debug,
#                                          superatom_file=superatom_path, spelling_file=spelling_path)
#         if not smiles:
#             log.warning('No SMILES strings were extracted for diagram %s' % diag.tag)
#
#         if not debug:
#             io_.imdel(img_name)
#         clean_dict_files(contents, [superatom_path, spelling_path])
#         return smiles
#
#     def _read_diagram_pyosra(self, diag, extension='jpg', debug=False, pad_val=1, superatom_path=superatom_file,
#                             spelling_path=spelling_file):
#         """ Converts a diagram to SMILES using pyosra
#
#         :param diag: Diagram to be extracted
#         :param extension: String file extension
#         :param debug: Bool inicating debug mode
#
#         :return smile: String of extracted chemical SMILE
#
#         """
#         diag_crop = diag.crop
#         if hasattr(diag_crop, 'clean_raw_img'):  # Choose either the cleaned (de-noised) version or a raw crop
#             img = diag_crop.clean_raw_img
#         else:
#             img = diag_crop.raw_img
#         # Add some padding to image to help resolve characters on the edge
#         if len(img.shape) == 3:
#             padded_img = pad(img, ((20, 20), (20, 20), (0, 0)), mode='constant', constant_values=pad_val)
#         else:
#             padded_img = pad(img, ((20, 20), (20, 20)), mode='constant', constant_values=pad_val)
#
#         # Save a temp image
#         temp_img_fname = 'osra_temp.' + extension
#         io_.imsave(temp_img_fname, padded_img)
#
#         # Run osra on temp image
#         try:
#             smile = osra_rgroup.read_diagram(temp_img_fname, debug=debug, superatom_file=superatom_path,
#                                              spelling_file=spelling_path)
#         except Exception as e:
#             print(str(e))
#
#         if not smile:
#             log.warning('No SMILES string was extracted for diagram %s' % diag.tag)
#
#         if not debug:
#             io_.imdel(temp_img_fname)
#
#         smile = clean_output(smile)
#         return smile
#
#     def is_false_positive(self, diag, allow_wildcards=False):
#         """ Identifies failures from incomplete / invalid smiles
#
#         :rtype bool
#         :returns : True if result is a false positive
#         """
#
#         # label_candidates, smiles = diag.label, diag.smiles
#         smiles = diag.smiles
#         # Remove results without a label
#         # if len(label_candidates) == 0:
#         #     return True
#
#         # Remove results containing the wildcard character in the SMILE
#         if '*' in smiles and not allow_wildcards:
#             return True
#
#         # Remove results where no SMILE was returned
#         if smiles == '':
#             return True
#
#         return False
#
# def clean_dict_files(old_contents, files):
#     for content, file in zip(old_contents, files):
#         with open(file, 'w') as f:
#             f.write(content)
#
# def read_contents(files):
#     contents = []
#     for file in files:
#         with open(file) as f:
#             contents.append(f.read())
#
#     return contents