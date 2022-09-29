# -*- coding: utf-8 -*-
"""
Extract
=======

Main extraction routine. Run from the command line to start extraction.

author: Damian Wilary
email: dmw51@cam.ac.uk

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os

from extractors.scheme_extractor import SchemeExtractor
from extractors.arrows import ArrowClassifier, ArrowDetector
from configs.config import Config

MAIN_DIR = os.getcwd()

log = logging.getLogger('extract')
file_handler = logging.FileHandler(os.path.join(Config.ROOT_DIR, 'extract.log'))
log.addHandler(file_handler)

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, required=True, help='Path to a single image or to a directory with images to extract' )
parser.add_argument('--finegrained_search', action='store_true')
parser.add_argument('--output_dir', type=str)
#parser.add_argument('--visualize', action='store_true')

opts = parser.parse_args()

if __name__ == '__main__':
    extractor = SchemeExtractor(opts)
    extractor.extract()

