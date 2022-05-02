# -*- coding: utf-8 -*-
"""
Extract
=======

Main extraction routines.

author: Damian Wilary
email: dmw51@cam.ac.uk

"""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import logging
import matplotlib.pyplot as plt
import os

from extractors.scheme_extractor import SchemeExtractor
from .actions import estimate_single_bond
from .config import Config
from .extractors.arrows import ArrowExtractor
from .extractors.unified import UnifiedExtractor
from .models.output import ReactionScheme
from .processors import ImageReader, EdgeExtractor
from .recognise import DecimerRecogniser

MAIN_DIR = os.getcwd()

import matplotlib

log = logging.getLogger('extract')
file_handler = logging.FileHandler(os.path.join(Config.ROOT_DIR, 'extract.log'))
log.addHandler(file_handler)

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, required=True, help='Path to a single image or to a directory with images to extract' )
parser.add_argument('--finegrained_search', action='store_true')
parser.add_argument('--save_output', action='store_true')
parser.add_argument('--output_dir', type=str)

opts = parser.parse_args()
if opts.to_json and not opts.output_dir:
    parser.error('Output directory path must be provided to save the output')

if __name__ == '__main__':
    extractor = SchemeExtractor(opts)
    extractor.extract()

