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

from DECIMER.config import get_bnw_image, delete_empty_borders, central_square_image, PIL_im_to_BytesIO, get_resize, increase_contrast
from DECIMER.decimer import tokenizer, DECIMER_V2

from models.reaction import Diagram
from reactiondataextractor.models.segments import FigureRoleEnum, Figure
from utils.utils import isolate_patches

log = logging.getLogger()


class DecimerRecogniser:
    def __init__(self, model_id='Canonical'):
        assert model_id.capitalize() in ['Canonical', 'Isomeric', 'Augmented'], "model_id has to be one of the following:\
                                                                            ['Canonical', 'Isomeric', 'Augmented']"
        self.model = DECIMER_V2

    def decode_image(self, img: np.ndarray) -> 'Tensor':
        """
        Loads and preprocesses an image
        :param img: image array for preprocessing
        :type img: np.ndarray
        """
        # img = self.remove_transparent(img)
        img = increase_contrast(img)
        img = get_bnw_image(img)
        img = get_resize(img)
        img = delete_empty_borders(img)
        img = central_square_image(img)
        img = PIL_im_to_BytesIO(img)
        img = tf.image.decode_png(img.getvalue(), channels=3)
        img = tf.image.resize(img, (512, 512), method="gaussian", antialias=True)
        img = efn.preprocess_input(img)
        return img

    def detokenize_output(self, predicted_array: int) -> str:
        """
        This function takes the predited tokens from the DECIMER model
        and returns the decoded SMILES string.
        :param predicted_array: Predicted tokens from DECIMER
        :type predicted_array: int
        :return: smiles representation of a diagram
        :rtype: str"""
        outputs = [tokenizer.index_word[i] for i in predicted_array[0].numpy()]
        prediction = (
            "".join([str(elem) for elem in outputs])
            .replace("<start>", "")
            .replace("<end>", "")
        )

        return prediction
    