# -*- coding: utf-8 -*-
"""
Optical Character Recognition
=============================

Extract text from images using Tesseract.
git
author: Damian Wilary
email: dmw51@cam.ac.uk


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import locale

locale.setlocale(locale.LC_ALL, 'C')
import collections
import enum
import logging
import numpy as np
from typing import Union

import cv2
from PIL import Image, ImageEnhance
import tesserocr

from configs.config import OCRConfig
from reactiondataextractor.models.segments import Rect

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
                        + HYPHEN + OTHER + '+'
CHAR_WHITELIST = DIGITS + '+' + ALPHABET_UPPER + ALPHABET_LOWER + "',\""

OCR_CONFIDENCE = 70

# api = tesserocr.PyTessBaseAPI(path=OCRConfig.TESSDATA_PATH, oem=tesserocr.OEM.TESSERACT_ONLY)
api = tesserocr.PyTessBaseAPI(init=False)
api.InitFull(
    path=OCRConfig.TESSDATA_PATH,
    variables={"load_system_dawg": "F",
               "load_freq_dawg": "F"},
    oem=tesserocr.OEM.TESSERACT_ONLY
)

class TextChar:
    """Class to represent an individual text character
    """
    def __init__(self, panel: 'Panel'):
        """Initializes the character by cropping the relevant image patch, preprocessing 
        and performing optical recognition.

        :param panel: panel containing the text character
        :type panel: Panel
        """
        self.panel = panel
        ocr_img = cv2.cvtColor(self.panel.crop.img_detectron, cv2.COLOR_RGB2GRAY)
        ocr_img = cv2.imread(pil_enhance(cv2_preprocess(ocr_img)), cv2.IMREAD_GRAYSCALE)

        text_blocks_char = get_text(ocr_img, whitelist=CHAR_WHITELIST, psm=PSM.SINGLE_CHAR)
        text_blocks_word = get_text(ocr_img, whitelist=CHAR_WHITELIST, psm=PSM.SINGLE_WORD)

        text_blocks = max([text_blocks_char, text_blocks_word], key=lambda output: output[0].confidence if output else 0)
        if text_blocks:
            text = text_blocks[0].text.strip()
            self.confidence = np.mean([block.confidence for block in text_blocks])
            self.text = text
        else:
            self.text = ''
            self.confidence = 0.0

    def __repr__(self):
        return f'TextChar({self.panel})'

    def __str__(self):
        return f"TextChar('{self.text}')"


def img_to_text(img: np.ndarray,
                whitelist: str,
                conf_threshold: int=70,
                psm: Union['PSM', None]=None):
    """High-level OCR function

    :param img: image to be fed to the OCR engine
    :type img: np.ndarray
    :param whitelist: list of allowed characters to be used by the OCR engine
    :type whitelist: str
    :param conf_threshold: confidence threshold, results below threshold are discarded, defaults to 70
    :type conf_threshold: int, optional
    :param psm: page segmentation mode used by the OCR engine, defaults to None
    :type psm: Union[PSM, None], optional
    :return: _description_
    :rtype: _type_
    """
 
    if psm is None:
        psm = PSM.SINGLE_BLOCK
    # top, left, bottom, right = region
    # img = crop.img
    img = cv2.imread(_pil_enhance(_cv2_preprocess(img)), cv2.IMREAD_GRAYSCALE)
    initial_ocr = get_text(img, psm=psm, whitelist=whitelist, pad_val=0)
    if initial_ocr:
        analyser = OCRAnalyser(img, initial_ocr, conf_threshold=conf_threshold)
        return analyser.build_output()
    return []


def _cv2_preprocess(img):

    img = cv2.resize(img, (0,0), fx=4, fy=4)
    kernel = np.ones((3, 3), np.uint8)

    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)

    img = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    cv2.imwrite('temp.png', img)
    return 'temp.png'


def _pil_enhance(image_path):
    image = Image.open(image_path)

    contrast = ImageEnhance.Contrast(image)
    contrast.enhance(4).save('temp.png')
    return 'temp.png'


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

    api.SetPageSegMode(psm)
    api.SetImage(Image.fromarray(img, mode='L'))
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


class TextElement:
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
        an individual TextElement (or instance of its subclass)
        :param img: image from which to recognise text
        :type img: np.ndarray
        :param ocr_output: initial output from the OCR engine
        :tyoe ocr_outputs: list[TextBlock]
        :param conf_threshold: threshold for detection confidence used to perform finer-grained analysis
        :type conf_threshold: float"""
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
        """Analyse a single word
        param w: recognised word to be further analysed
        type w: TextWord"""
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
