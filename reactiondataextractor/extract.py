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

import json
import logging
import matplotlib.pyplot as plt
import os

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


def extract_image(filename, debug=False):
    """
    Extracts reaction schemes from a single file specified by ``filename``. ``debug`` enables more detailed logging and
    plotting.

    :param str filename: name of the image file
    :param bool debug: bool enabling debug mode
    :return Scheme: Reaction scheme object
    """
    level = 'DEBUG' if debug else 'INFO'
    ch = logging.StreamHandler()
    log.setLevel(level)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)

    path = os.path.join(MAIN_DIR, filename)
    log.info(f'Extraction started...')

    reader = ImageReader(path, color_mode=ImageReader.COLOR_MODE.GRAY)
    fig = reader.process()

    Config.FIGURE = fig
    estimate_single_bond(fig)

    binarizer = EdgeExtractor(fig)
    fig = binarizer.process()

    arrow_extractor = ArrowExtractor(fig)
    solid_arrows, eq_arrows, res_arrows, curly_arrows = arrow_extractor.extract()
    arrows = solid_arrows + eq_arrows + res_arrows + curly_arrows
    f, ax = plt.subplots()
    ax.imshow(fig.img, cmap=plt.cm.binary)
    arrow_extractor.plot_extracted(ax)

    all_arrows = solid_arrows + eq_arrows + res_arrows + curly_arrows
    log.info(f'Detected {len(all_arrows)} arrows')
    unified_extractor = UnifiedExtractor(fig, all_arrows)
    diags, conditions, labels = unified_extractor.extract()
    log.info(f'Found {len(diags)} panels of chemical diagrams')
    log.info(f'Found {len(labels)} labels')
    unified_extractor.plot_extracted(ax)
    plt.show()
    recogniser = DecimerRecogniser()
    for d in diags:
        print(recogniser._recognise_diagram(fig, d))
    r = ReactionScheme(arrows, diags, fig)
    print(r.long_str())

    if debug:
        f = plt.figure()
        ax = f.add_axes([0, 0, 1, 1])
        ax.imshow(fig.img, cmap=plt.cm.binary)
        arrow_extractor.plot_extracted(ax)
        unified_extractor.plot_extracted(ax)
        ax.axis('off')
        ax.set_title('Segmented image')
        plt.show()

    scheme = ReactionScheme(arrows, diags, fig)
    log.info('Scheme completed without errors.')

    return scheme


def extract_images(indir_path, out_dir_path, debug=False):
    """Performs a series of extraction and outputs the graphs converted into a JSON format

    Extracts reaction schemes from all files in ``dirname`` and saves the output in the JSON format to a ``out_dirname``
    directory
    :param str indir_path: path to the directory containing input files
    :param str out_dir_path: path to the directory, where output will be saved
    """

    for filename in os.listdir(indir_path):
        try:
            if not os.path.exists(out_dir_path):
                os.mkdir(out_dir_path)

            path = os.path.join(indir_path, filename)
            if os.path.isdir(path):
                continue
            else:
                scheme = extract_image(path)

                out_file = '.'.join(filename.split('.')[:-1])+'.json'
                out_path = os.path.join(out_dir_path, out_file)
                with open(out_path, 'w') as out_file:
                    out_file.write(scheme.to_json())

        except Exception as e:
            print(f'Extraction failed for file {filename}')
            if debug:
                print(f'Exception message: {str(e)}')
            continue

    print('all schemes extracted')
