import os
from copy import deepcopy
from enum import Enum, auto
from abc import ABC, abstractmethod
import imageio as imageio
import numpy as np
from typing import Union, Tuple

import cv2
from PIL import Image
from scipy.stats import mode

from configs.figure import GlobalFigureMixin
from reactiondataextractor.models.segments import Figure
from reactiondataextractor.configs import config


class ImageProcessor(ABC, GlobalFigureMixin):
    """Base class for all image processors. Each subclass has to implement the `process` method"""
    class COLOR_MODE(Enum):
        GRAY = auto()
        RGB = auto()

    def __init__(self, fig=None):
        self.fig = fig
        super().__init__(self.fig)
        self._img = self.fig.img if self.fig else None

    @abstractmethod
    def process(self):
        """The main processing method"""
        pass

    @property
    def img(self):
        return self._img


class ImageReader(ImageProcessor):
    """Class for reading an image file. Reads jpg/jpeg, png, bmp, as well as single images in gif format"""
    def __init__(self, filepath: str, color_mode: 'ImageProcessor.COLOR_MODE'):
        """init method. Takes in filepath as well as color mode (gray or RGB). RGB mode support is currently limited.
        

        :param filepath: absolute path to file
        :type filepath: str
        :param color_mode: processing mode - the image will be either kept as RGB or processed into grayscale
        :type color_mode: ImageProcessor.COLOR_MODE
        """
        config.Config.IMG_PATH = filepath
        assert color_mode in self.COLOR_MODE, "Color_mode must be one of ImageColor.COLORMODE enum members"
        assert os.path.exists(filepath), "Could not open file - Invalid path was entered"
        self.filepath = filepath
        self.color_mode = color_mode
        _, self.ext = os.path.splitext(filepath)
        super().__init__()

    def process(self):
        """Reads an image into an np.ndarray from .png, .jpg/.jpeg etc formats, as well as .gif format (used by some
        journals)"""
        if self.color_mode == self.COLOR_MODE.GRAY:
            img = cv2.imread(self.filepath, cv2.IMREAD_GRAYSCALE)
            img_detectron = cv2.imread(self.filepath)
        elif self.color_mode == self.COLOR_MODE.RGB:
            img = cv2.imread(self.filepath, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_detectron = cv2.imread(self.filepath)

        if img is None and self.ext == '.gif':   # Ensure this special case is treated

            try:
                img = imageio.mimread(self.filepath, format='gif')
                img = img[0]
            except ValueError:  # Binary images not handled above
                img = Image.open(self.filepath).convert('L')
                img = np.asarray(img)
            img, img_detectron = self._convert_gif(img)

        img = self.adjust_bg_value(img)
        img_detectron = self.adjust_bg_value(img_detectron, desired=255)
        self.fig = Figure(img=img, raw_img=img, img_detectron=img_detectron)  # Important that all used imgs have bg value 0 hence raw_img==img
        return self.fig

    def _convert_gif(self, img):
        img_detectron = img
        if len(img_detectron.shape) == 2:
            img_detectron = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR )
        elif len(img.shape) == 3 and img.shape[-1] == 4:  # RGBA
            img_detectron = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        if self.color_mode == self.COLOR_MODE.GRAY:
            if len(img.shape) == 3 and img.shape[-1] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY )
            elif len(img.shape) == 3 and img.shape[-1] == 3:  # RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif self.color_mode == self.COLOR_MODE.RGB:
            if len(img.shape) == 3 and img.shape[-1] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB )
            elif len(img.shape) == 2:  #Gray
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return img, img_detectron

    def adjust_bg_value(self, img, desired=0):
        """Flips the image if the background colour does not match the desired background colour
        :param img: image to be processed
        :type img: np.ndarray
        :param desired: the expected value associated with background (0 or 255)
        :type desired: int"""
        bg_value = mode(img.ravel(), keepdims=False)[0]

        if desired == 0:
            if bg_value in range(250, 256):
                img = np.invert(img)
        elif desired == 255:
            if bg_value in range(0, 10):
                img = np.invert(img)
        return img


class ImageScaler(ImageProcessor):
    """Processor used for scaling an image. Constant scale facilitates later processing"""

    def __init__(self, fig: 'Figure', resize_min_dim_to: int):
        """Init method, where the scale is set

        :param fig: Processed figure
        :type fig: Figure
        :param resize_min_dim_to: dimension to which the smaller dimension (W or H) will be resized
        :type resize_min_dim_to: int
        """
        self.resize_min_dim_to = resize_min_dim_to
        super().__init__(fig=fig)

    def process(self):
        """Scales an image so that the smaller dimension has the desired length (specified inside __init__)"""
        min_dim = min(self.fig.img.shape)
        y_dim, x_dim = self.fig.img.shape
        scaling_factor = self.resize_min_dim_to/min_dim
        img = cv2.resize(self.fig.img, (int(x_dim*scaling_factor), int(y_dim*scaling_factor)))
        self.fig._scaling_factor = scaling_factor
        self.fig.img = img
        return self.fig

class ImageNormaliser(ImageProcessor):
    """Processor used to normalise an image to a range between 0 and 1"""
    def __init__(self, fig: 'Figure'):
        super().__init__(fig=fig)

    def process(self):
        """Normalises an image to range [0, 1]"""
        img = self.fig.img
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        self.fig.img = img
        return self.fig


class Binariser(ImageProcessor):
    """Processor used for binarisation. Grayscale images are thresholed, whereas on RGB images,
    a Canny edge detector is used"""

    def __init__(self, fig: 'Figure', bin_thresh: Tuple[int]=None):
        """Init method. Input is a figure, and a pair of integers used by the thresholding algorithm

        :param fig: Processed Figure
        :type fig: Figure
        :param bin_thresh: a pair of integers to be used as the thresholding values by the algorithm, defaults to None
        :type bin_thresh: Tuple[int], optional
        """
        super().__init__(fig=fig)
        if len(self.img.shape) == 2:
            self.color_mode = self.COLOR_MODE.GRAY
        elif len(self.img.shape) == 3 and self.img.dtype == np.uint8:
            self.color_mode = self.COLOR_MODE.RGB
        self.bin_thresh = bin_thresh if bin_thresh else config.ProcessorConfig.BIN_THRESH

    def process(self):
        """Binarises gray images"""
        if self.color_mode == self.COLOR_MODE.GRAY:
            ret, img = cv2.threshold(self.img, *self.bin_thresh, cv2.THRESH_BINARY)
            fig_copy = deepcopy(self.fig)
            fig_copy.img = img
            return fig_copy


class Isolator(ImageProcessor):
    """Processor class used for isolating individual connected components"""
    def __init__(self, fig, to_isolate, isolate_mask):
        super().__init__(fig=fig)
        self.to_isolate = to_isolate
        self.isolate_mask = isolate_mask

    def _isolate_panel(self):
        #TODO
        pass

    def _isolate_mask(self):
        rows, cols = self.to_isolate.pixels
        mask = np.zeros_like(self.img, dtype=bool)
        mask[rows, cols] = True
        isolated_cc_img = np.zeros_like(self.img, dtype=np.uint8)
        isolated_cc_img[mask] = self.img[mask]
        fig_copy = deepcopy(self.fig)
        fig_copy.img = isolated_cc_img
        return fig_copy

    def process(self):
        """Isolates connected components either by using their panels or pixel-wise masks"""
        if self.isolate_mask:
            return self._isolate_mask()
        else:
            return self._isolate_panel()
