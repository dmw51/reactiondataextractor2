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

from actions import estimate_single_bond
from extractors.arrows import ArrowExtractor
from extractors.unified import UnifiedExtractor
from models.base import BaseExtractor
from models.output import ReactionScheme
from processors import ImageReader, ImageScaler, ImageNormaliser, EdgeExtractor
from recognise import DecimerRecogniser


class SchemeExtractor(BaseExtractor):

    def __init__(self, opts):
        self.opts = opts
        self.path = Path(opts.path)


    def extract(self):
        if self.path.is_dir():
            return self.extract_from_dir()
        else:
            self.extract_from_image(self.path)

    def plot_extracted(self, ax):
        pass

    def extract_from_image(self, path):
        """Main extraction method used for extracting data from a single image"""
        reader = ImageReader(path, color_mode=ImageReader.COLOR_MODE.GRAY)
        fig = reader.process()
        # orig_fig = deepcopy(fig)
        scaler = ImageScaler(fig, resize_min_dim_to=1024, enabled=True)
        # scaler = ImageScaler(fig, resize_max_dim_to=1024, enabled=True)
        fig = scaler.process()
        normaliser = ImageNormaliser(fig)
        fig = normaliser.process()

        # Config.FIGURE = fig

        binarizer = EdgeExtractor(fig)
        fig = binarizer.process()
        estimate_single_bond(fig)
        arrow_extractor = ArrowExtractor(fig)
        solid_arrows, eq_arrows, res_arrows, curly_arrows = arrow_extractor.extract()
        arrows = solid_arrows + eq_arrows + res_arrows + curly_arrows
        # import matplotlib.pyplot as plt
        # f, ax = plt.subplots(figsize=(20, 20))
        #
        # ax.imshow(fig.img, cmap=plt.cm.binary)
        # arrow_extractor.plot_extracted(ax)

        all_arrows = solid_arrows + eq_arrows + res_arrows + curly_arrows
        unified_extractor = UnifiedExtractor(fig, all_arrows, use_tiler=self.opts.finegrained_search)
        diags, conditions, labels = unified_extractor.extract()
        # unified_extractor.plot_extracted(ax)
        # plt.show()
        from reactiondataextractor.models.output import RoleProbe
        p = RoleProbe(fig, arrows, diags)

        [p.probe_around_arrow(arrow) for arrow in arrows]
        # [step.visualize(fig) for step in p.reaction_steps]
        # if inconsistent_nodes: # Resolve only if a problem is present
        # p.resolve_nodes()
        # p.visualize_steps()
        # d1, d2 = p.probe_around_arrow(arrows[0])
        # recogniser = DecimerRecogniser()
        # for d in diags:
        #     print(recogniser._recognise_diagram(fig, d))
        scheme = ReactionScheme(fig, p.reaction_steps)
        if self.opts.save_output:
            self.save_scheme_to_disk(scheme)
        return scheme

    def extract_from_dir(self):
        """Main extraction method used for extracting data from a single image"""
        schemes = []
        for image_path in self.path.iterdir():
            try:
                scheme = self.extract_from_image(image_path)
                schemes.append(scheme)
            except Exception as e:
                print(f'Extraction failed: {str(e)}')
        return schemes

    def save_scheme_to_disk(self, scheme):
        """Writes the reconstructed scheme to disk"""
        pass