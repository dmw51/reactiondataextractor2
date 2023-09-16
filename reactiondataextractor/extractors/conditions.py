# -*- coding: utf-8 -*-
"""
Conditions
=======

This module contains classes and methods for extracting conditions, as well as directly related functions.

author: Damian Wilary
email: dmw51@cam.ac.uk

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import logging
import numpy as np
import re
from typing import List, Dict

from matplotlib.patches import Rectangle

from configs.config import ExtractorConfig
from reactiondataextractor.models.reaction import Conditions
from reactiondataextractor.models.base import BaseExtractor
from reactiondataextractor.ocr import img_to_text, CONDITIONS_WHITELIST

log = logging.getLogger('extract.conditions')

SPECIES_FILE = ExtractorConfig.CONDITIONS_SPECIES_PATH


class ConditionsExtractor(BaseExtractor):
    """Main class for extracting reaction conditions from images. Takes in the main figure and bounding panels of detected regions,
    recognises text and parses the data.
    """

    def __init__(self, fig: 'Figure', priors: List['Diagram']):
        """
        :param fig: main figure
        :type fig: Figure
        :param priors: Bounding panels of regions recognised as conditions
        :type priors: list[Panel]
        """
        super().__init__(fig)
        self.priors = priors
        self.ocr_fig = None
        self._extracted = None

    def extract(self):
        """Main extraction method.

        Delegates recognition to the OCR model, then parsing to the ConditionParser class. Returns the parsed
        Conditions.
        :return: parsed Conditions object
        :rtype: Conditions
        """
        assert self.ocr_fig
        conditions = []
        for cand in self.priors:
            # step_conditions = self.get_conditions(cand)
            crop = cand.panel.create_crop(self.ocr_fig)
            
            recognised = img_to_text(crop.img, whitelist=CONDITIONS_WHITELIST)
            if recognised:
                dct = ConditionParser(recognised).parse_conditions()
                step_conditions = Conditions(conditions_dct=dct, **cand.pass_attributes(), text=recognised)
                conditions.append(step_conditions)
            else:
                conditions.append(Conditions(conditions_dct={}, panel=cand.panel))
        self._extracted = conditions
        return self.extracted

    @property
    def extracted(self):
        """Returns extracted objects"""
        return self._extracted

    def plot_extracted(self, ax):
        """Adds extracted panels onto a canvas of ``ax``"""
        conditions = self._extracted
        params = {'facecolor': 'g', 'edgecolor': None, 'alpha': 0.3}

        for step_conditions in conditions:
            panel = step_conditions.panel
            rect_bbox = Rectangle((panel.left - 1, panel.top - 1), panel.right - panel.left,
                                  panel.bottom - panel.top, **params)
            ax.add_patch(rect_bbox)

    def add_diags_to_dicts(self, diags: List['Diagram']) -> None:
        """Adds SMILES representations of diagrams that had been assigned to conditions regions

        :param [Diagram,...] diags: iterable of extracted diagrams
        :return: None (mutates the conditions dictionary)
        :rtype: None"""
        conditions, _ = self.extracted
        for step_conditions in conditions:
            if step_conditions.structure_panels:
                cond_diags = [diag for diag in diags if diag.panel in step_conditions.structure_panels]
                step_conditions.diags = cond_diags
                try:
                    step_conditions.conditions_dct['other species'].extend(
                        [diag.smiles for diag in cond_diags if diag.smiles])
                except KeyError:
                    step_conditions.conditions_dct['other species'] = [diag.smiles for diag in cond_diags if
                                                                       diag.smiles]


class ConditionParser:
    """
    This class is used to parse conditions text. It is composed of several methods to facilitate parsing recognised text
    using formal grammars.

    The following strings define formal grammars to detect catalysts (cat) and coreactants (co) based on their units.
    Species which fulfill neither criterion can be parsed as `other_chemicals`. `default_values` is also defined to help 
    parse both integers and floating-point values.

    :param textlines: Sentence object retrieved from an OCR engine.
    :type textlines: chemdataextractor.Sentence
    """
    default_values = r'((?:\d\.)?\d{1,3})'
    cat_units = r'(mol\s?%|M|wt\s?%|%\s\w+)'
    # co_units = r'(eq\.?(?:uiv(?:alents?)?\.?)?|m?L)'
    co_units = r'(equivalents?|equiv\.?|eq\.?|m?L)'

    def __init__(self, textlines: List[str]):
        self.text_lines = textlines 

    def parse_conditions(self) -> Dict:
        """Parses conditions by first looking for chemical entities, and then selecting these entities
        by each parsing method one by one. Each entity selected by a parsing method is popped from the sequence,
        and not analysed further"""
        conditions_dct = {'catalysts': [], 'coreactants': [], 'other species': [], 'temperature': None,
                          'pressure': None, 'time': None, 'yield': None}

        textlines = [re.sub(r',(?!\d)', '#', textline) for textline in self.text_lines]
        entities = [entity for textline in textlines for entity in re.split(r'#.', textline)]
        entities = [entity.strip('#').strip() for entity in entities]
        
        #TODO: Now classify the entities into the subgroups, and then from leftovers pick the remaining chemicals
        remaining_entities = entities
        coreactants, remaining_entities = ConditionParser._parse_coreactants(remaining_entities)
        catalysts, remaining_entities = ConditionParser._parse_catalysis(remaining_entities)
        other_conditions, remaining_entities = ConditionParser._parse_other_conditions(remaining_entities)
        other_species, _ = ConditionParser._parse_other_species(remaining_entities)

        conditions_dct['coreactants'] = coreactants
        conditions_dct['catalysts'] = catalysts
        conditions_dct['other species'] = other_species
        conditions_dct.update(other_conditions)
        return conditions_dct
    
    @staticmethod
    def _identify_species(entities):

        with open(SPECIES_FILE, 'r') as file:
            species_list = file.read().strip().split('\n')

        # letters between which some lowercase letters and digits are allowed, optional brackets
        formulae_brackets = r'((?:[A-Z]*\d?[a-z]\d?)\((?:[A-Z]*\d?[a-z]?\d?)*\)?\d?[A-Z]*[a-z]*\d?)*'
        formulae_bracketless = r'(?<!°)\b(?<!\)|\()((?:[A-Z]+\d?[a-z]?\d?)+)(?!\(|\))\b'
        letter_upper_identifiers = r'((?<!°)\b[A-Z]{1,4}\b)(?!\)|\.)'  # Up to four capital letters
        letter_lower_identifiers = r'(\b[a-z]\b)(?!\)|\.)'  # Accept single lowercase letter subject to restrictions

        number_identifiers = r'(?:^| )(?<!\w)([1-9])(?!\w)(?!\))(?:$|[, ])(?![A-Za-z])'
        # number_identifiers matches the following:
        # "1, 2, 3": three numbers as chemical identifiers
        # "CH3OH, 5, 6 (5 equiv)"": 5 and 6 in the middle only
        # "5 5 equiv"  first 5 only
        # "A 5 equiv" - no matches

        captured_entities = []
        remaining_entities = []
        for entity in entities:
            if '/' in entity:
                captured_entities.append(entity)
            elif any(species in entity for species in species_list if species):
                captured_entities.append(entity)
            elif any([re.search(formulae_brackets, entity),
                    re.search(formulae_bracketless, entity),
                    re.search(letter_upper_identifiers, entity),
                    re.search(letter_lower_identifiers, entity),
                    re.search(number_identifiers, entity)]):
                captured_entities.append(entity)
            else:
                remaining_entities.append(entity)

        return captured_entities, remaining_entities
    
    @staticmethod
    def _parse_coreactants(entities):
        co_values = ConditionParser.default_values
        co_str = co_values + r'\s?' + ConditionParser.co_units
        remaining_entities = []
        captured_entities = []
        for entity in entities:
            if re.search(co_str, entity):
                captured_entities.append(entity)
            else:
                remaining_entities.append(entity)
        return captured_entities, remaining_entities
    
    @staticmethod
    def _parse_catalysis(entities):
        cat_values = ConditionParser.default_values
        cat_str = cat_values + r'\s?' + ConditionParser.cat_units
        remaining_entities = []
        captured_entities = []
        for entity in entities:
            if re.search(cat_str, entity):
                captured_entities.append(entity)
            else:
                remaining_entities.append(entity)
        return captured_entities, remaining_entities

    @staticmethod
    def _parse_other_species(sentence):
        cems = ConditionParser._identify_species(sentence)
        return [cem for cem in cems]

    @staticmethod
    def _parse_other_conditions(entities):
        other_dct = {}
        remaining_entities = entities
        temperature, remaining_entities = ConditionParser._parse_temperature(remaining_entities)
        time, remaining_entities = ConditionParser._parse_time(remaining_entities)
        pressure, remaining_entities = ConditionParser._parse_pressure(remaining_entities)
        yield_, remaining_entities = ConditionParser._parse_yield(remaining_entities)

        if temperature:
            other_dct['temperature'] = temperature  # Create the key only if temperature was parsed
        if time:
            other_dct['time'] = time
        if pressure:
            other_dct['pressure'] = pressure
        if yield_:
            other_dct['yield'] = yield_

        return other_dct, remaining_entities
  
    @staticmethod
    def _parse_time(entities):  # add conditions to add the parsed data
        t_values = ConditionParser.default_values
        t_units = r'(h(?:ours?)?|m(?:in)?|s(?:econds)?|days?)'
        time_str = re.compile(r'(?<!\w)' + t_values + r'\s?' + t_units + r'(?=$|\s?,)')
        remaining_entities = []
        captured_entities = []
        for entity in entities:
            if re.search(time_str, entity):
                captured_entities.append(re.search(time_str, entity))
            else:
                remaining_entities.append(entity)
        return [ConditionParser._form_dict_entry(t) for t in captured_entities], remaining_entities
    
    @staticmethod
    def _parse_temperature(entities):
        # The following formals grammars for temperature and pressure are quite complex, but allow to parse additional
        # generic descriptors like 'heat' or 'UHV' in `.group(1)'
        t_units = r'\s?(?:o|O|0|°)C|K'   # match 0C, oC and similar, as well as K

        t_value1 = r'-?\d{1,4}' + r'\s?(?=' + t_units + ')'  # capture numbers only if followed by units
        t_value2 = r'r\.?\s?t\.?'
        t_value3 = r'heat|reflux|room\s?temp'

        # Add greek delta?
        t_or = re.compile('(' + '|'.join((t_value1, t_value2, t_value3)) + ')' + '(' + t_units + ')' + '?', re.I)
        remaining_entities = []
        captured_entities = []
        for entity in entities:
            if re.search(t_or, entity):
                captured_entities.append(re.search(t_or, entity))
            else:
                remaining_entities.append(entity)
        return [ConditionParser._form_dict_entry(t) for t in captured_entities], remaining_entities

    @staticmethod
    def _form_dict_entry(match):
        if match:
            units = match.group(2) if match.group(2) else 'N/A'
            try:
                return {'Value': float(match.group(1)), 'Units': units}
            except ValueError:
                return {'Value': match.group(1), 'Units': units}   # if value is rt or heat, gram scale etc
    
    @staticmethod
    def _parse_pressure(entities):
        p_units = r'(?:m|h|k|M)?Pa|m?bar|atm'   # match bar, mbar, mPa, hPa, MPa and atm

        p_values1 = r'\d{1,4}' + r'\s?(?=' + p_units + ')'  # match numbers only if followed by units
        p_values2 = r'(?:U?HV)|vacuum'

        p_or = re.compile('(' + '|'.join((p_values1, p_values2)) + ')' + '(' + p_units + ')' + '?')
        remaining_entities = []
        captured_entities = []
        for entity in entities:
            if re.search(p_or, entity):
                captured_entities.append(re.search(p_or, entity))
            else:
                remaining_entities.append(entity)
        return [ConditionParser._form_dict_entry(t) for t in captured_entities], remaining_entities

    @staticmethod
    def _parse_yield(entities):
        y_units = r'%'   # match 0C, oC and similar, as well as K

        y_value1 = r'\d{1,2}' + r'\s?(?=' + y_units + ')'  # capture numbers only if followed by units
        y_value2 = r'gram scale'

        # Add greek delta?
        y_or = re.compile('(' + '|'.join((y_value1, y_value2)) + ')' + '(' + y_units + ')' + '?')
        remaining_entities = []
        captured_entities = []
        for entity in entities:
            if re.search(y_or, entity):
                captured_entities.append(re.search(y_or, entity))
            else:
                remaining_entities.append(entity)
        return [ConditionParser._form_dict_entry(t) for t in captured_entities], remaining_entities
