# -*- coding: utf-8 -*-
"""
Optical Character Recognition
=============================

Extract text from images using Tesseract.

Module adapted by:-
author: Damian Wilary
email: dmw51@cam.ac.uk

Initial implementation created by:
author: Matthew Swain
email: m.swain@me.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import locale

import PIL.Image
import cv2

from reactiondataextractor.config import OCRConfig
from reactiondataextractor.models.segments import Rect

locale.setlocale(locale.LC_ALL, 'C')
import collections
import enum
import logging
import warnings

import numpy as np
from PIL import Image, ImageEnhance
import tesserocr
from skimage import img_as_ubyte, img_as_uint
from skimage.filters import gaussian
from scipy import stats

# from chemdataextractor.doc.text import Sentence
# from chemdataextractor.parse.cem import BaseParser
# from .utils.processing import isolate_patches, erase_elements
# from .parse import ChemSchematicResolverTokeniser
# from .models.segments import Crop, Rect

log = logging.getLogger('extract.ocr')

# Whitelist for labels
ALPHABET_UPPER = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ALPHABET_LOWER = ALPHABET_UPPER.lower()
DIGITS = '0123456789'

SUPERSCRIPT = '⁰°'
ASSIGNMENT = ':=-'
HYPHEN = '-'
CONCENTRATION = '%()'
BRACKETS = '[]{}'
SEPARATORS = ',. '
OTHER = r'\'`/@'
LABEL_WHITELIST = (ASSIGNMENT + DIGITS + ALPHABET_UPPER + ALPHABET_LOWER + CONCENTRATION +
                   OTHER  + SUPERSCRIPT + SEPARATORS) + '+'
CONDITIONS_WHITELIST = DIGITS + ALPHABET_UPPER + ALPHABET_LOWER + CONCENTRATION + SEPARATORS + BRACKETS + SUPERSCRIPT \
                        + HYPHEN + OTHER
CHAR_WHITELIST = DIGITS + '+' + ALPHABET_UPPER

OCR_CONFIDENCE = 70

api = tesserocr.PyTessBaseAPI(oem=tesserocr.OEM.TESSERACT_ONLY)


def img_to_text(fig, region, whitelist, conf_threshold=70):
    top, left, bottom, right = region
    img = fig.img[top:bottom, left:right]
    img = cv2.imread(pil_enhance(cv2_preprocess(img)), cv2.IMREAD_GRAYSCALE)
    initial_ocr = get_text(img, psm=PSM.SINGLE_BLOCK, whitelist=whitelist, pad_val=0)
    analyser = OCRAnalyser(img, initial_ocr, conf_threshold=conf_threshold)
    return analyser.build_output()


def cv2_preprocess(img):

    img = cv2.resize(img, (0,0), fx=3, fy=3)
    kernel = np.ones((3, 3), np.uint8)


    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)


    # apply a blur
    # gaussian noise
    img = cv2.threshold(cv2.GaussianBlur(img, (0, 0), 1), 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # this can be used for salt and pepper noise (not necessary here)
    #img = cv2.adaptiveThreshold(cv2.medianBlur(img, 7), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    cv2.imwrite('temp.png', img)
    return 'temp.png'


def pil_enhance(image_path):
    image = Image.open(image_path)

    contrast = ImageEnhance.Contrast(image)
    contrast.enhance(4).save('temp.png')
    return 'temp.png'



# def read_label(fig, label, whitelist=LABEL_WHITELIST, pad_val=0):
#     """ Reads a label paragraph objects using ocr

#     :param Figure fig: main figure
#     :param Label label: Label object containing appropriate bounding box
#     :param whitelist: list of allowed characters
#     :param pad_val: value used for image padding (0 or 1)

#     :rtype List[chemdataextractor.sentence]
#     """
#     crop = label.panel.create_crop(fig)

#     blocks, avg_conf = choose_best_ocr_configuration(
#         crop.img, x_offset=label.left, y_offset=label.top,
#         psms=[PSM.SPARSE_TEXT, PSM.SINGLE_CHAR, PSM.SINGLE_LINE, PSM.SINGLE_WORD, PSM.SINGLE_BLOCK],
#         whitelist=whitelist, pad_val=pad_val)

#     if not blocks:
#         return [], 0
#     elif len(blocks) == 1 and blocks[0].text.strip() in ['+', '', ',', '.']:
#         return [], 0

#     log.debug('Confidence in OCR: %s' % avg_conf)

#     raw_sentences = get_sentences(blocks)

#     if len(raw_sentences) is not 0:
#         # Tag each paragraph
#         tagged_sentences = [Sentence(sentence, word_tokenizer=ChemSchematicResolverTokeniser(),
#                                      parsers=[BaseParser()]) for sentence in raw_sentences]
#     else:
#         tagged_sentences = []

#     return tagged_sentences, avg_conf


# def read_conditions(fig, region, conf_threshold=OCR_CONFIDENCE, whitelist=CONDITIONS_WHITELIST, pad_val=0):
#     """
#     Reads conditions' text in `fig.img` detected inside `text_line` and returns the recognised OCR objects.
#     :param Figure fig: figure containing unprocessed image
#     :param TextCandidate region:  a TextCandidate storing the region recognised as reaction conditions
#     :param int conf_threshold:
#     :param str whitelist: all characters that will be looked for in the text
#     :param int pad_val: value used for padding (0 or 1)
#     :return: tagged sentences if OCR is successful or empty list if failed
#     :rtype: list[chemdataextractor.Sentence]|list[]
#     """
#     panel = region.panel
#     region_crop = panel.create_padded_crop(fig, pad_width=(5, 5), pad_val=pad_val)

#     # try:
#     #     img = lift_subscripts(region_crop)
#     # except ValueError:    # Cannot broadcast
#     #     img = region_crop.img
#     # finally:
#     text, avg_conf = choose_best_ocr_configuration(region_crop.img, x_offset=panel.left, y_offset=panel.top,
#                                                    psms=[PSM.SINGLE_BLOCK,PSM.SINGLE_LINE, PSM.SINGLE_WORD],
#                                                    whitelist=whitelist, pad_val=pad_val)
#     return text
#     # raw_sentence = get_sentences(text)
#     #
#     # if len(raw_sentence) == 1:
#     #     # Tag each paragraph
#     #     tagged_sentence = Sentence(raw_sentence[0].strip(), word_tokenizer=ChemSchematicResolverTokeniser(),
#     #                                parsers=[BaseParser()])
#     # else:
#     #     tagged_sentence = []
#     #
#     # log.debug('Tagged sentence: %s' % tagged_sentence)
#     # log.debug('Confidence in OCR: %s' % avg_conf)
#     # return tagged_sentence if avg_conf > conf_threshold else []


# def lift_subscripts(textline_crop, shift=None):
#     """
#     Finds subscripts in a text line and moves them up.
#
#     Recognise individual, relevant connected components in a text line. If they're recognised as digits, moves them
#     up slightly to aid optical character recognition of the text line.
#     :param Crop textline_crop: a Crop object containing the text line region
#     :param shift: vertical shift used for lifting
#     :return: np.ndarray containing a new text line with subscripts moved up
#     :rtype: np.ndarray
#     """
#     subs = []
#     mode_bottom = stats.mode([cc.bottom for cc in textline_crop.connected_components]).mode[0]
#     for cc in textline_crop.connected_components:
#         if cc.bottom > mode_bottom:
#             cc_img = isolate_patches(textline_crop, [cc])
#             cc_img = cc_img.img
#             char, confidence = choose_best_ocr_configuration(cc_img, psms=[PSM.SINGLE_WORD, PSM.SINGLE_CHAR],
#                                                              whitelist=CONDITIONS_WHITELIST, pad_val=0)
#             if char and char[0].text.strip() in list('0123456789'):
#                 subs.append(cc)
#
#     f_erased = erase_elements(textline_crop, subs)
#     if subs:
#         if shift is None:
#             shift = int(np.mean([sub.bottom for sub in subs]) - mode_bottom)
#
#         for sub in subs:
#             arr = textline_crop.img[sub.top:sub.bottom + 1, sub.left:sub.right + 1]
#             f_erased.img[sub.top - shift:sub.bottom + 1 - shift, sub.left:sub.right + 1] = arr
#         return f_erased.img
#
#     else:
#         return textline_crop.img


# def read_isolated_conditions(isolated_block):
#     """
#     Helper function to read conditions from isolated connected components segmented as conditions' text characters
#     :param isolated_block:
#     :return: tagged sentences if OCR is successful or empty list if failed
#     :rtype: list[chemdataextractor.Sentence]|list[]
#     """
#     fig = isolated_block
#     conditions_region = isolated_block.get_bounding_box()
#     return read_conditions(fig, conditions_region)

def img_as_pil(arr, format_str=None):
    """Convert an scikit-image image (ndarray) to a PIL object.

    Derived from code in scikit-image PIL IO plugin.

    :param numpy.ndarray raw_img: The image to convert.
    :return: PIL image.
    :rtype: Image
    """
    if arr.ndim == 3:
        arr = img_as_ubyte(arr)
        mode = {3: 'RGB', 4: 'RGBA'}[arr.shape[2]]

    elif format_str in ['png', 'PNG']:
        mode = 'I;16'
        mode_base = 'I'

        if arr.dtype.kind == 'f':
            arr = img_as_uint(arr)

        elif arr.max() < 256 and arr.min() >= 0:
            arr = arr.astype(np.uint8)
            mode = mode_base = 'L'

        else:
            arr = img_as_uint(arr)

    else:
        arr = img_as_ubyte(arr)
        mode = 'L'
        mode_base = 'L'

    try:
        array_buffer = arr.tobytes()
    except AttributeError:
        array_buffer = arr.tostring()  # Numpy < 1.9

    if arr.ndim == 2:
        im = Image.new(mode_base, arr.T.shape)
        try:
            im.frombytes(array_buffer, 'raw', mode)
        except AttributeError:
            im.fromstring(array_buffer, 'raw', mode)  # PIL 1.1.7
    else:
        image_shape = (arr.shape[1], arr.shape[0])
        try:
            im = Image.frombytes(mode, image_shape, array_buffer)
        except AttributeError:
            im = Image.fromstring(mode, image_shape, array_buffer)  # PIL 1.1.7
    return im


# These enums just wrap tesserocr functionality, so we can return proper enum members instead of ints.
class Orientation(enum.IntEnum):
    """Text element orientations enum."""
    #: Up orientation.
    PAGE_UP = tesserocr.Orientation.PAGE_UP
    #: Right orientation.
    PAGE_RIGHT = tesserocr.Orientation.PAGE_RIGHT
    #: Down orientation.
    PAGE_DOWN = tesserocr.Orientation.PAGE_DOWN
    #: Left orientation.
    PAGE_LEFT = tesserocr.Orientation.PAGE_LEFT


class WritingDirection(enum.IntEnum):
    """Text element writing directions enum."""
    #: Left to right.
    LEFT_TO_RIGHT = tesserocr.WritingDirection.LEFT_TO_RIGHT
    #: Right to left.
    RIGHT_TO_LEFT = tesserocr.WritingDirection.RIGHT_TO_LEFT
    #: Top to bottom.
    TOP_TO_BOTTOM = tesserocr.WritingDirection.TOP_TO_BOTTOM


class TextlineOrder(enum.IntEnum):
    """Text line order enum."""
    #: Left to right.
    LEFT_TO_RIGHT = tesserocr.TextlineOrder.LEFT_TO_RIGHT
    #: Right to left.
    RIGHT_TO_LEFT = tesserocr.TextlineOrder.RIGHT_TO_LEFT
    #: Top to bottom.
    TOP_TO_BOTTOM = tesserocr.TextlineOrder.TOP_TO_BOTTOM


class Justification(enum.IntEnum):
    """Justification enum."""
    #: Unknown justification.
    UNKNOWN = tesserocr.Justification.UNKNOWN
    #: Left justified
    LEFT = tesserocr.Justification.LEFT
    #: Center justified
    CENTER = tesserocr.Justification.CENTER
    #: Right justified
    RIGHT = tesserocr.Justification.RIGHT


class PSM(enum.IntEnum):
    """Page Segmentation Mode enum."""
    OSD_ONLY = tesserocr.PSM.OSD_ONLY
    AUTO_OSD = tesserocr.PSM.AUTO_OSD
    AUTO_ONLY = tesserocr.PSM.AUTO_ONLY
    AUTO = tesserocr.PSM.AUTO
    SINGLE_COLUMN = tesserocr.PSM.SINGLE_COLUMN
    SINGLE_BLOCK_VERT_TEXT = tesserocr.PSM.SINGLE_BLOCK_VERT_TEXT
    SINGLE_BLOCK = tesserocr.PSM.SINGLE_BLOCK
    SINGLE_LINE = tesserocr.PSM.SINGLE_LINE
    SINGLE_WORD = tesserocr.PSM.SINGLE_WORD
    CIRCLE_WORD = tesserocr.PSM.CIRCLE_WORD
    SINGLE_CHAR = tesserocr.PSM.SINGLE_CHAR
    SPARSE_TEXT = tesserocr.PSM.SPARSE_TEXT
    SPARSE_TEXT_OSD = tesserocr.PSM.SPARSE_TEXT_OSD
    RAW_LINE = tesserocr.PSM.RAW_LINE
    COUNT = tesserocr.PSM.COUNT


class RIL(enum.IntEnum):
    """Page Iterator Level enum."""
    BLOCK = tesserocr.RIL.BLOCK
    PARA = tesserocr.RIL.PARA
    SYMBOL = tesserocr.RIL.SYMBOL
    TEXTLINE = tesserocr.RIL.TEXTLINE
    WORD = tesserocr.RIL.WORD


def get_words(blocks):
    """Convert list of text blocks into a flat list of the contained words.

    :param list[TextBlock] blocks: List of text blocks.
    :return: Flat list of text words.
    :rtype: list[TextWord]
    """
    words = []
    for block in blocks:
        for para in block:
            for line in para:
                for word in line:
                    words.append(word)
    return words


def get_lines(blocks):
    """Convert list of text blocks into a nested list of lines, each of which contains a list of words.

    :param list[TextBlock] blocks: List of text blocks.
    :return: List of sentences
    :rtype: list[list[TextWord]]
    """
    lines = []
    for block in blocks:
        for para in block:
            for line in para:
                words = []
                for word in line:
                    words.append(word)
                lines.append(words)
    return lines


def get_sentences(blocks):
    """Convert list of text blocks into a nested list of lines, each of which contains a list of words.

    :param list[TextBlock] blocks: List of text blocks.
    :return: List of sentences
    :rtype: list[list[TextWord]]
    """
    sentences = []
    for block in blocks:
        if hasattr(block, 'text'):
            sentences.append(block.text)
        else:
            for line in block:
                # sentences.append(line.text.replace(',', ' ')) # NB - commas switched for spaces to improve tokenization
                sentences.append(line.text)
    return sentences


def get_text(img, x_offset=0, y_offset=0, psm=PSM.SINGLE_LINE, img_padding=20, whitelist=None, img_orientation=None,
             pad_val=0):
    """Get text elements in image.

    When passing a cropped image to this function, use ``x_offset`` and ``y_offset`` to ensure the coordinate positions
    of the extracted text elements are relative to the original uncropped image.

    :param numpy.ndarray img: Input image.
    :param int x_offset: Offset to add to the horizontal coordinates of the returned text elements.
    :param int y_offset: Offset to add to the vertical coordinates of the returned text elements.
    :param PSM psm: Page segmentation mode.
    :param int img_padding: Padding to add to text element bounding boxes.
    :param string whitelist: String containing allowed characters. e.g. Use '0123456789' for digits.
    :param Orientation img_orientation: Main orientation of text in image, if known.
    :return: List of text blocks.
    :rtype: list[TextBlock]
    """
    log.debug(
        'get_text: %s x_offset=%s, y_offset=%s, padding=%s, whitelist=%s',
        img.shape, x_offset, y_offset, img_padding, whitelist
    )
    # Rescale image - make it larger

    # Add a buffer around the entire input image to ensure no text is too close to edges
    if img.ndim == 3:
        npad = ((img_padding, img_padding), (img_padding, img_padding), (0, 0))
    elif img.ndim == 2:
        npad = ((img_padding, img_padding), (img_padding, img_padding))
    else:
        raise ValueError('Unexpected image dimensions')
    img = np.pad(img, pad_width=npad, mode='constant', constant_values=pad_val)
    shape = img.shape

    # Rotate img before sending to tesseract if an img_orientation has been given
    if img_orientation == Orientation.PAGE_LEFT:
        img = np.rot90(img, k=3, axes=(0, 1))
    elif img_orientation == Orientation.PAGE_RIGHT:
        img = np.rot90(img, k=1, axes=(0, 1))
    elif img_orientation is not None:
        raise NotImplementedError('Unsupported img_orientation')

    def _get_common_props(it, ril):
        """Get the properties that apply to all text elements."""
        # Important: Call GetUTF8Text() before Orientation(). Former raises RuntimeError if no text, latter Segfaults.
        text = it.GetUTF8Text(ril)
        orientation, writing_direction, textline_order, deskew_angle = it.Orientation()
        bb = it.BoundingBox(ril, padding=0)

        # Translate bounding box and orientation if img was previously rotated
        if img_orientation == Orientation.PAGE_LEFT:
            orientation = {
                Orientation.PAGE_UP: Orientation.PAGE_LEFT,
                Orientation.PAGE_LEFT: Orientation.PAGE_DOWN,
                Orientation.PAGE_DOWN: Orientation.PAGE_RIGHT,
                Orientation.PAGE_RIGHT: Orientation.PAGE_UP
            }[orientation]
            left, right, top, bottom = bb[1], bb[3], shape[0] - bb[2], shape[0] - bb[0]
        elif img_orientation == Orientation.PAGE_RIGHT:
            orientation = {
                Orientation.PAGE_UP: Orientation.PAGE_RIGHT,
                Orientation.PAGE_LEFT: Orientation.PAGE_UP,
                Orientation.PAGE_DOWN: Orientation.PAGE_LEFT,
                Orientation.PAGE_RIGHT: Orientation.PAGE_DOWN
            }[orientation]
            left, right, top, bottom = shape[1] - bb[3], shape[1] - bb[1], bb[0], bb[2]
        else:
            left, right, top, bottom = bb[0], bb[2], bb[1], bb[3]

        common_props = {
            'text': text,
            'left': left + x_offset - img_padding,
            'right': right + x_offset - img_padding,
            'top': top + y_offset - img_padding,
            'bottom': bottom + y_offset - img_padding,
            'confidence': it.Confidence(ril),
            'orientation': Orientation(orientation),
            'writing_direction': WritingDirection(writing_direction),
            'textline_order': TextlineOrder(textline_order),
            'deskew_angle': deskew_angle
        }
        return common_props

    blocks = []

    #     with tesserocr.PyTessBaseAPI(psm=psm, oem=tesserocr.OEM.TESSERACT_LSTM_COMBINED) as api:

    # Convert image to PIL to load into tesseract (suppress precision loss warning)
    # with warnings.catch_warnings(record=True) as ws:
    #     pil_img = img_as_pil(img)
    # api.SetImage(pil_img)
    api.SetPageSegMode(psm)
    api.SetImage(PIL.Image.fromarray(img, mode='L'))
    if whitelist is not None:
        api.SetVariable('tessedit_char_whitelist', whitelist)
    # TODO: api.SetSourceResolution if we want correct pointsize on output?
    api.Recognize()
    it = api.GetIterator()
    block = None
    para = None
    line = None
    word = None
    it.Begin()

    while True:
        try:
            if it.IsAtBeginningOf(RIL.BLOCK):
                common_props = _get_common_props(it, RIL.BLOCK)
                block = TextBlock(**common_props)
                blocks.append(block)

            if it.IsAtBeginningOf(RIL.PARA):
                common_props = _get_common_props(it, RIL.PARA)
                justification, is_list_item, is_crown, first_line_indent = it.ParagraphInfo()
                para = TextParagraph(
                    is_ltr=it.ParagraphIsLtr(),
                    justification=Justification(justification),
                    is_list_item=is_list_item,
                    is_crown=is_crown,
                    first_line_indent=first_line_indent,
                    **common_props
                )
                if block is not None:
                    block.paragraphs.append(para)

            if it.IsAtBeginningOf(RIL.TEXTLINE):
                common_props = _get_common_props(it, RIL.TEXTLINE)
                line = TextLine(**common_props)
                if para is not None:
                    para.lines.append(line)

            if it.IsAtBeginningOf(RIL.WORD):
                common_props = _get_common_props(it, RIL.WORD)
                wfa = it.WordFontAttributes()
                if wfa:
                    common_props.update(wfa)
                word = TextWord(
                    language=it.WordRecognitionLanguage(),
                    from_dictionary=it.WordIsFromDictionary(),
                    numeric=it.WordIsNumeric(),
                    **common_props
                )
                if line is not None:
                    line.words.append(word)

            # Beware: Character level coordinates do not seem to be accurate in Tesseact 4!!
            common_props = _get_common_props(it, RIL.SYMBOL)
            symbol = TextSymbol(
                is_dropcap=it.SymbolIsDropcap(),
                is_subscript=it.SymbolIsSubscript(),
                is_superscript=it.SymbolIsSuperscript(),
                **common_props
            )
            word.symbols.append(symbol)
        except RuntimeError as e:
            # Happens if no text was detected
            log.debug(e)

        if not it.Next(RIL.SYMBOL):
            break
    return blocks


class TextElement():
    """Abstract base.py class for all text elements."""

    def __init__(self, text, left, right, top, bottom, orientation, writing_direction, textline_order, deskew_angle,
                 confidence):
        """

        :param string text: Recognized text content.
        :param int left: Left edge of bounding box.
        :param int right: Right edge of bounding box.
        :param int top: Top edge of bounding box.
        :param int bottom: Bottom edge of bounding box.
        :param Orientation orientation: Orientation of this element.
        :param WritingDirection writing_direction: Writing direction of this element.
        :param TextlineOrder textline_order: Text line order of this element.
        :param float deskew_angle: Angle required to make text upright in radians.
        :param float confidence: Mean confidence for the text in this element. Probability 0-100%.
        """
        super(TextElement, self).__init__()
        self.rect = Rect((top, left, bottom, right))
        self.text = text
        self.orientation = orientation
        self.writing_direction = writing_direction
        self.textline_order = textline_order
        self.deskew_angle = deskew_angle
        self.confidence = confidence

    @property
    def coords(self):
        return self.rect.coords

    def __repr__(self):
        return '<%s: %r>' % (self.__class__.__name__, self.text)

    def __str__(self):
        return '<%s: %r>' % (self.__class__.__name__, self.text)


class TextBlock(TextElement, collections.MutableSequence):
    """Text block."""

    def __init__(self, text, left, right, top, bottom, orientation, writing_direction, textline_order, deskew_angle,
                 confidence):
        """

        :param string text: Recognized text content.
        :param int left: Left edge of bounding box.
        :param int right: Right edge of bounding box.
        :param int top: Top edge of bounding box.
        :param int bottom: Bottom edge of bounding box.
        :param Orientation orientation: Orientation of this element.
        :param WritingDirection writing_direction: Writing direction of this element.
        :param TextlineOrder textline_order: Text line order of this element.
        :param float deskew_angle: Angle required to make text upright in radians.
        :param float confidence: Mean confidence for the text in this element. Probability 0-100%.
        """
        super(TextBlock, self).__init__(text, left, right, top, bottom, orientation, writing_direction, textline_order,
                                        deskew_angle, confidence)
        self.paragraphs = []

    def __getitem__(self, index):
        return self.paragraphs[index]

    def __setitem__(self, index, value):
        self.paragraphs[index] = value

    def __delitem__(self, index):
        del self.paragraphs[index]

    def __len__(self):
        return len(self.paragraphs)

    def insert(self, index, value):
        self.paragraphs.insert(index, value)


class TextParagraph(TextElement, collections.MutableSequence):
    """Text paragraph.

    :param string text: Recognized text content.
    :param int left: Left edge of bounding box.
    :param int right: Right edge of bounding box.
    :param int top: Top edge of bounding box.
    :param int bottom: Bottom edge of bounding box.
    :param Orientation orientation: Orientation of this element.
    :param WritingDirection writing_direction: Writing direction of this element.
    :param TextlineOrder textline_order: Text line order of this element.
    :param float deskew_angle: Angle required to make text upright in radians.
    :param float confidence: Mean confidence for the text in this element. Probability 0-100%.
    :param bool is_ltr: Whether this paragraph text is left to right.
    :param Justification justification: Paragraph justification.
    :param bool is_list_item: Whether this paragraph is part of a list.
    :param bool is_crown: Whether the first line is aligned with the subsequent lines yet other paragraphs are indented.
    :param int first_line_indent: Indent of first line in pixels.
    """

    def __init__(self, text, left, right, top, bottom, orientation, writing_direction, textline_order, deskew_angle,
                 confidence, is_ltr, justification, is_list_item, is_crown, first_line_indent):
        super(TextParagraph, self).__init__(text, left, right, top, bottom, orientation, writing_direction,
                                            textline_order, deskew_angle, confidence)
        self.lines = []
        self.is_ltr = is_ltr
        self.justification = justification
        self.is_list_item = is_list_item
        self.is_crown = is_crown
        self.first_line_indent = first_line_indent

    def __getitem__(self, index):
        return self.lines[index]

    def __setitem__(self, index, value):
        self.lines[index] = value

    def __delitem__(self, index):
        del self.lines[index]

    def __len__(self):
        return len(self.lines)

    def insert(self, index, value):
        self.lines.insert(index, value)


class TextLine(TextElement, collections.MutableSequence):
    """Text line."""

    def __init__(self, text, left, right, top, bottom, orientation, writing_direction, textline_order, deskew_angle,
                 confidence):
        """

        :param string text: Recognized text content.
        :param int left: Left edge of bounding box.
        :param int right: Right edge of bounding box.
        :param int top: Top edge of bounding box.
        :param int bottom: Bottom edge of bounding box.
        :param Orientation orientation: Orientation of this element.
        :param WritingDirection writing_direction: Writing direction of this element.
        :param TextlineOrder textline_order: Text line order of this element.
        :param float deskew_angle: Angle required to make text upright in radians.
        :param float confidence: Mean confidence for the text in this element. Probability 0-100%.
        """
        super(TextLine, self).__init__(text, left, right, top, bottom, orientation, writing_direction, textline_order,
                                       deskew_angle, confidence)
        self.words = []

    def __getitem__(self, index):
        return self.words[index]

    def __setitem__(self, index, value):
        self.words[index] = value

    def __delitem__(self, index):
        del self.words[index]

    def __len__(self):
        return len(self.words)

    def insert(self, index, value):
        self.words.insert(index, value)


class TextWord(TextElement, collections.MutableSequence):
    """Text word."""

    def __init__(self, text, left, right, top, bottom, orientation, writing_direction, textline_order, deskew_angle,
                 confidence, language, from_dictionary, numeric, font_name=None, bold=None, italic=None,
                 underlined=None, monospace=None, serif=None, smallcaps=None, pointsize=None, font_id=None):
        """

        :param string text: Recognized text content.
        :param int left: Left edge of bounding box.
        :param int right: Right edge of bounding box.
        :param int top: Top edge of bounding box.
        :param int bottom: Bottom edge of bounding box.
        :param Orientation orientation: Orientation of this element.
        :param WritingDirection writing_direction: Writing direction of this element.
        :param TextlineOrder textline_order: Text line order of this element.
        :param float deskew_angle: Angle required to make text upright in radians.
        :param float confidence: Mean confidence for the text in this element. Probability 0-100%.
        :param language: Language used to recognize this word.
        :param from_dictionary: Whether this word was found in a dictionary.
        :param numeric: Whether this word is numeric.
        :param string font_name: Font name.
        :param bool bold: Whether this word is bold.
        :param bool italic: Whether this word is italic.
        :param underlined: Whether this word is underlined.
        :param monospace: Whether this word is in a monospace font.
        :param serif: Whethet this word is in a serif font.
        :param smallcaps: Whether this word is in small caps.
        :param pointsize: Font size in points (1/72 inch).
        :param font_id: Font ID.
        """
        super(TextWord, self).__init__(text, left, right, top, bottom, orientation, writing_direction, textline_order,
                                       deskew_angle, confidence)
        self.symbols = []
        self.font_name = font_name
        self.bold = bold
        self.italic = italic
        self.underlined = underlined
        self.monospace = monospace
        self.serif = serif
        self.smallcaps = smallcaps
        self.pointsize = pointsize
        self.font_id = font_id
        self.language = language
        self.from_dictionary = from_dictionary
        self.numeric = numeric

    def __getitem__(self, index):
        return self.symbols[index]

    def __setitem__(self, index, value):
        self.symbols[index] = value

    def __delitem__(self, index):
        del self.symbols[index]

    def __len__(self):
        return len(self.symbols)

    def insert(self, index, value):
        self.symbols.insert(index, value)


class TextSymbol(TextElement):
    """Text symbol."""

    def __init__(self, text, left, right, top, bottom, orientation, writing_direction, textline_order, deskew_angle,
                 confidence, is_dropcap, is_subscript, is_superscript):
        """

        :param string text: Recognized text content.
        :param int left: Left edge of bounding box.
        :param int right: Right edge of bounding box.
        :param int top: Top edge of bounding box.
        :param int bottom: Bottom edge of bounding box.
        :param Orientation orientation: Orientation of this element.
        :param WritingDirection writing_direction: Writing direction of this element.
        :param TextlineOrder textline_order: Text line order of this element.
        :param float deskew_angle: Angle required to make text upright in radians.
        :param float confidence: Mean confidence for the text in this element. Probability 0-100%.
        :param bool is_dropcap: Whether this symbol is a dropcap.
        :param bool is_subscript: Whether this symbol is subscript.
        :param bool is_superscript: Whether this symbol is superscript.
        """
        super(TextSymbol, self).__init__(text, left, right, top, bottom, orientation, writing_direction, textline_order,
                                         deskew_angle, confidence)
        self.is_dropcap = is_dropcap
        self.is_subscript = is_subscript
        self.is_superscript = is_superscript


class TextParserAdapter:
    """Starting from any level in the parsed optically recognised text, allows getting individual paragraphs, lines,
    words, or symbols"""

    class ParsedLevelEnum(enum.Enum):
        """Each class is assigned a value indicating how many levels 'down' the data structure are the innermost elements"""
        TEXTSYMBOL = 0
        TEXTWORD = 1
        TEXTLINE = 2
        TEXTPARAGRAPH = 3
        TEXTBLOCK = 4

    level_dct = {TextSymbol: ParsedLevelEnum.TEXTSYMBOL,
                 TextWord: ParsedLevelEnum.TEXTWORD,
                 TextLine: ParsedLevelEnum.TEXTLINE,
                 TextParagraph: ParsedLevelEnum.TEXTPARAGRAPH,
                 TextBlock: ParsedLevelEnum.TEXTBLOCK
                 }

    def __init__(self, parsed):
        self.parsed = parsed
        self.level = self.level_dct[parsed.__class__]

    def get_all_elements(self, element_type):
        """Extracts all elements from a given level of the parsed data structure. `element_type` has to be an enum
        member from TextParserAdapter.ParsedLevelEnum
        :param element_type: Which elements (e.g. lines, words or symbols) should be extracted
        :type element_type: TextParserAdapter.ParsedLevelEnum"""

        max_level = self.level.value - element_type.value
        print(max_level)
        current_level = 0

        to_traverse = [self.parsed]
        current_level = 0
        while current_level < max_level:
            to_traverse = self._get_all_elements(to_traverse)
            current_level += 1
        return to_traverse

    def _get_all_elements(self, sequence):
        lst = [list(elem) for elem in sequence]
        lst = [subelem for elem in lst for subelem in elem]
        return lst


class OCRAnalyser:
    def __init__(self, img, ocr_output, conf_threshold):
        """Initialization is based on the initial output of the OCR engine. `ocr_output` must be a list/tuple, not
        an individual TextElement (or instance of its subclass)"""
        self.img = img
        self.ocr_output = ocr_output

        self.conf_threshold = conf_threshold
        self._text = []
        self.text_lines = []

        self.psm_dct = {TextWord: PSM.SINGLE_WORD,
                        TextSymbol: PSM.SINGLE_CHAR}

    def build_output(self):
        """Traverses the initial output from the OCR pipeline and adds words based on obtained confidence. If the
        confidence is below threshold, a given word is analysed on its own and only then appended to the list of found words"""
        for text_elem in self.ocr_output:
            words = TextParserAdapter(text_elem).get_all_elements(TextParserAdapter.ParsedLevelEnum.TEXTWORD)
            for w in words:
                if w.confidence > self.conf_threshold:
                    self._text.append(w.text)
                else:
                    self.analyse_word(w)

            return self.recover_textlines()

    def recover_textlines(self):
        """Puts the built output into appropriate text lines. This increases reliability of the parsing process
        by providing contextual information"""
        len_textlines = []
        for text_elem in self.ocr_output:
            lines = TextParserAdapter(text_elem).get_all_elements(TextParserAdapter.ParsedLevelEnum.TEXTLINE)
            len_textlines.extend([len(l) for l in lines])

        cum_len = [sum(len_textlines[:idx]) for idx in range(len(len_textlines))] + [len(self._text)]

        self.text_lines = [' '.join(self._text[cum_len[slice_idx]:cum_len[slice_idx + 1]]) for slice_idx in
                           range(len(cum_len) - 1)]

        return self.text_lines

    def analyse_word(self, w):
        confidence, text = self._analyse(w)
        if confidence > self.conf_threshold:
            self._text.append(text)
        else:
            word_piecewise = []
            for char in w:
                if char.rect.area > OCRConfig.PIECEWISE_OCR_THRESH_AREA:
                    _, text = self._analyse(char)
                    word_piecewise.append(text)
            self._text.append(''.join(word_piecewise))

    def _analyse(self, element):
        top, left, bottom, right = element.coords
        cropped = self.img[top:bottom, left:right]
        out = get_text(cropped, psm=self.psm_dct[element.__class__], whitelist=CONDITIONS_WHITELIST, pad_val=0)
        #         out = TextParserAdapter(out).get_all_elements(TextParserAdapter.level_dct[element.__class__])
        if not out:
            return 0, ''
        conf, text = zip(*[(elem.confidence, elem.text) for elem in out])
        conf = np.mean(conf)
        text = ' '.join(text)

        return conf, text.strip()
