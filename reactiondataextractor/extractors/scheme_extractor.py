# -*- coding: utf-8 -*-
"""
Scheme Extractor
======

This module contains SchemeExtractor - the main extractor used to process chemical reaction schemes.
for RGroup resolution

author: Damian Wilary
email: dmw51@cam.ac.uk
"""
from pathlib import Path

import matplotlib.pyplot as plt

from utils.vectorised import estimate_single_bond
from configs.config import Config
from extractors.arrows import ArrowExtractor
from extractors.unified import UnifiedExtractor

from extractors.smiles import SmilesExtractor

from models.base import BaseExtractor
from reactiondataextractor.models.exceptions import NoArrowsFoundException, NoDiagramsFoundException
from models.output import ReactionScheme, RoleProbe
from processors import ImageReader, ImageScaler, ImageNormaliser, Binariser
from recognise import DecimerRecogniser


class SchemeExtractor(BaseExtractor):
    """The main, high-level scheme extraction class. Can be used for extracting from single images, or from directories.
    The extraction should be run from the command line using extract.py using the arguments listed there """

    def __init__(self, opts):
        """
        :param opts: options from the command line
        :type opts: argparse.Namespace
        """
        self.opts = opts


        self.path = Path(opts.path)

        self._extract_single_image = False if self.path.is_dir() else True
        if not self._extract_single_image:
            assert self.opts.output_dir, """For extraction from a directory, you need to provide a path to save the output using --output_dir flag"""

        self.arrow_extractor = ArrowExtractor(fig=None)
        self.unified_extractor = UnifiedExtractor(fig=None, arrows=[], use_tiler=self.opts.finegrained_search)
        self.recogniser = DecimerRecogniser()

        self.scheme = None


    @property
    def extracted(self):
        return self.scheme

    def extract(self):
        """The main extraction method. Allows extraction from single image or a directory using a single interface. """
        if not self._extract_single_image:
            return self.extract_from_dir()
        else:
            scheme = self.extract_from_image(self.path)
            self.scheme = scheme
            return scheme

    def plot_extracted(self, ax=None):
        """Currently plotting is supported only for single-image extraction"""
        if ax is None:
            f = plt.Figure(figsize=(10, 10))
            ax = f.add_axes([0, 0, 1, 1])
        ax.imshow(self.fig.img, cmap=plt.cm.binary)
        self.arrow_extractor.plot_extracted(ax)
        self.unified_extractor.plot_extracted(ax)
        plt.show()

    def extract_from_image(self, path):
        """Main extraction method used for extracting data from a single image. Returns the parsed Scheme object.
        If an output directory is provided in the arguments' list, then the output is also saved there.

        :param path: path to an image
        :type path: Path
        :return: parsed reaction scheme
        :rtype: ReactionScheme
        """
        reader = ImageReader(str(path), color_mode=ImageReader.COLOR_MODE.GRAY)
        fig = reader.process()
        scaler = ImageScaler(fig, resize_min_dim_to=1024)#, enabled=True)
        fig = scaler.process()
        normaliser = ImageNormaliser(fig)
        fig = normaliser.process()

        binarizer = Binariser(fig)
        fig = binarizer.process()
        Config.FIGURE = fig
        self._fig = fig
        self.arrow_extractor.fig = fig
        self.unified_extractor.fig = fig
        estimate_single_bond(fig)
            
        try:
            self.arrow_extractor.extract()
            diags_only = False
        except NoArrowsFoundException:
            diags_only = True
        self.unified_extractor.diags_only = diags_only
        self.unified_extractor.all_arrows = self.arrow_extractor.arrows
        try:
            diags, conditions, labels = self.unified_extractor.extract()
        except NoDiagramsFoundException:
            print(f"No diagrams have been found in the image ({path}). Skipping the image...")
            return
        if self.opts.visualize:
            import matplotlib.pyplot as plt
            self.plot_extracted()
        
        smiles_extractor = SmilesExtractor(diags, self.recogniser)
        smiles_extractor.extract()
        if not diags_only:
            p = RoleProbe(fig, self.arrow_extractor.arrows, diags)
            p.probe()

            output = ReactionScheme(fig, p.reaction_steps, p.is_incomplete)
        else:
            output = self.unified_extractor
        if self.opts.output_dir:
            self.save_output_to_disk(output, path)
        return output

    def extract_from_dir(self):
        """Main extraction method used for extracting data from a single image"""
        schemes = []
        for image_path in self.path.iterdir():
            try:
                scheme = self.extract_from_image(image_path)
                print(f'Extraction finished: {image_path}')
                schemes.append(scheme)
            except Exception as e:
                print(f'Extraction failed for {image_path}: {str(e)}')
                schemes.append(None)
        return schemes

    def save_output_to_disk(self, output, image_path):
        """Writes the reconstructed output to disk
        :param output: Reconstructed output object
        :type output: ReactionScheme
        :param image_path: path to the input image from which the scheme was extracted
        :type image_path: Path
        :return: None
        """
        out_name = Path(f'{image_path.stem}.json')
        outpath = self.opts.output_dir / out_name
        with open(outpath, 'w') as outfile:
            outfile.write(output.to_json())
