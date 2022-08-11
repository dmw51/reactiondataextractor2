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

from collections.abc import Sequence
from itertools import chain
import logging
from matplotlib.patches import Rectangle
import re

from configs.config import ExtractorConfig
from reactiondataextractor.models.reaction import Conditions
from reactiondataextractor.models.base import BaseExtractor, TextRegion
from reactiondataextractor.models.segments import FigureRoleEnum
from reactiondataextractor.ocr import img_to_text, CONDITIONS_WHITELIST
from reactiondataextractor.utils import DisabledNegativeIndices, erase_elements

log = logging.getLogger('extract.conditions')

SPECIES_FILE = ExtractorConfig.CONDITIONS_SPECIES_PATH


class ConditionsExtractor(BaseExtractor):
    """Main class for extracting reaction conditions from images. Takes in the bounding panels of recognised regions,
    recognises text and parses the data.
    """

    def __init__(self, fig, priors):
        """
        :param fig: main figure
        :type fig: Figure
        :param priors: Bounding panels of regions recognised as conditions
        :type priors: list[Panel]
        """
        super().__init__(fig)
        self.priors = priors
        self._extracted = None

    def extract(self):
        """Main extraction method.

        Delegates recognition to the OCR model, then parsing to the ConditionParser class. Returns the parsed
        Conditions.
        :return: parsed Conditions object
        :rtype: Conditions
        """
        conditions, conditions_structures = [], []
        for cand in self.priors:
            # step_conditions = self.get_conditions(cand)
            recognised = img_to_text(self.fig, cand, whitelist=CONDITIONS_WHITELIST)
            if recognised:
                dct = ConditionParser(recognised).parse_conditions()
                step_conditions = Conditions(conditions_dct=dct, **cand.pass_attributes(), text=recognised)
                conditions.append(step_conditions)
            else:
                conditions.append(Conditions(conditions_dct={}, panel=cand.panel))
            # #TODO: Implement diagram handling (whether a diagram is part of the conditions region) - This should be mostly
            # #handled by the unified model; is IoU enough?
            # if step_conditions:
            #     conditions.append(step_conditions)
            # else:
            #     conditions.append(Conditions(conditions_dct={}, panel=cand.panel))
            # conditions_structures.extend(step_structures)
        self._extracted = conditions#, conditions_structures
        return self.extracted

    @property
    def extracted(self):
        """Returns extracted objects"""
        return self._extracted

    def plot_extracted(self, ax):
        """Adds extracted panels onto a canvas of ``ax``"""
        conditions = self._extracted
        params = {'facecolor': 'g', 'edgecolor': None, 'alpha': 0.3}
        # for panel in conditions_structures:
        #     rect_bbox = Rectangle((panel.left - 1, panel.top - 1), panel.right - panel.left,
        #                           panel.bottom - panel.top, **params)
        #     ax.add_patch(rect_bbox)

        for step_conditions in conditions:
            panel = step_conditions.panel
            rect_bbox = Rectangle((panel.left - 1, panel.top - 1), panel.right - panel.left,
                                  panel.bottom - panel.top, **params)
            ax.add_patch(rect_bbox)

    # def get_conditions(self, candidate):
    #     """
    #     Recovers conditions of a single reaction step.
    #
    #     Analyses the region stored inside `conditions_candidate`. Passes text through an OCR engine, and parses
    #     the output. Forms a Conditions object containing all the collected information.
    #     :param TextCandidate candidate: A candidate object containing a relevant textual region
    #     :return Conditions: Conditions object containing found information.
    #     """
    #     recognised = img_to_text(self.fig, candidate, whitelist=CONDITIONS_WHITELIST)
    #     if recognised:
    #         dct = ConditionParser(recognised).parse_conditions()
    #         return Conditions(conditions_dct=dct, **candidate.pass_attributes(), text=recognised)

    def add_diags_to_dicts(self, diags):
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
    cat_units = r'(mol\s?%|M|wt\s?%)'
    # co_units = r'(eq\.?(?:uiv(?:alents?)?\.?)?|m?L)'
    co_units = r'(equivalents?|equiv\.?|eq\.?|m?L)'

    def __init__(self, textlines):

        self.text_lines = textlines  # sentences are ChemDataExtractor Sentence objects

    def parse_conditions(self):
        parse_fns = [ConditionParser._parse_coreactants, ConditionParser._parse_catalysis,
                     ConditionParser._parse_other_species, ConditionParser._parse_other_conditions]
        conditions_dct = {'catalysts': [], 'coreactants': [], 'other species': [], 'temperature': None,
                          'pressure': None, 'time': None, 'yield': None}

        coreactants_lst = []
        catalysis_lst = []
        other_species_lst = []
        for line in self.text_lines:
            parsed = [parse(line) for parse in parse_fns]

            coreactants_lst.extend(parsed[0])
            catalysis_lst.extend(parsed[1])
            other_species_lst.extend(ConditionParser._filter_species(parsed))
            conditions_dct.update(parsed[3])

        conditions_dct['coreactants'] = coreactants_lst
        conditions_dct['catalysts'] = catalysis_lst
        conditions_dct['other species'] = other_species_lst
        return conditions_dct

    @staticmethod
    def _identify_species(textline):

        with open(SPECIES_FILE, 'r') as file:
            species_list = file.read().strip().split('\n')

        # letters between which some lowercase letters and digits are allowed, optional brackets
        formulae_brackets = r'((?:[A-Z]*\d?[a-z]\d?)\((?:[A-Z]*\d?[a-z]?\d?)*\)?\d?[A-Z]*[a-z]*\d?)*'
        formulae_bracketless = r'(?<!°)\b(?<!\)|\()((?:[A-Z]+\d?[a-z]?\d?)+)(?!\(|\))\b'
        letter_upper_identifiers = r'((?<!°)\b[A-Z]{1,4}\b)(?!\)|\.)'  # Up to four capital letters? Just a single one?
        letter_lower_identifiers = r'(\b[a-z]\b)(?!\)|\.)'  # Accept single lowercase letter subject to restrictions

        number_identifiers = r'(?:^| )(?<!\w)([1-9])(?!\w)(?!\))(?:$|[, ])(?![A-Za-z])'
        # number_identifiers matches the following:
        # 1, 2, 3, three numbers as chemical identifiers
        # CH3OH, 5, 6 (5 equiv) 5 and 6 in the middle only
        # 5 5 equiv  first 5 only
        # A 5 equiv -no matches
        entity_mentions_brackets = re.finditer(formulae_brackets, textline)
        entity_mentions_bracketless = re.finditer(formulae_bracketless, textline)
        entity_mentions_letters_upper = re.finditer(letter_upper_identifiers, textline)
        entity_mentions_letters_lower = re.finditer(letter_lower_identifiers, textline)

        entity_mentions_numbers = re.finditer(number_identifiers, textline)

        cems = [e.group(1) for e in
                 chain(entity_mentions_brackets, entity_mentions_bracketless,
                       entity_mentions_numbers, entity_mentions_letters_upper,
                       entity_mentions_letters_lower) if e.group(1)]
        slashed_names = []
        for word in textline.split(' '):
            if '/' in word:
                slashed_names.append(word)

        all_mentions = ConditionParser._resolve_cems(cems+slashed_names)
        # Add species from the list, treat them as seeds - allow more complex names
        # eg. based on 'pentanol' on the list, allow '1-pentanol'
        species_from_list = [token for token in textline.split(' ')
                             if any(species in token.lower() for species in species_list if species)]  # except ''
        all_mentions += species_from_list
        return list(set(all_mentions))

    @staticmethod
    def _parse_coreactants(sentence):
        co_values = ConditionParser.default_values
        co_str = co_values + r'\s?' + ConditionParser.co_units

        return ConditionParser._find_closest_cem(sentence, co_str)

    @staticmethod
    def _parse_catalysis(sentence):
        cat_values = ConditionParser.default_values
        cat_str = cat_values + r'\s?' + ConditionParser.cat_units

        return ConditionParser._find_closest_cem(sentence, cat_str)

    @staticmethod
    def _parse_other_species(sentence):
        cems = ConditionParser._identify_species(sentence)
        return [cem for cem in cems]

    @staticmethod
    def _parse_other_conditions(sentence):
        other_dct = {}
        parsed = [ConditionParser._parse_temperature(sentence), ConditionParser._parse_time(sentence),
                  ConditionParser._parse_pressure(sentence), ConditionParser._parse_yield(sentence)]

        temperature, time, pressure, yield_ = parsed
        if temperature:
            other_dct['temperature'] = temperature  # Create the key only if temperature was parsed
        if time:
            other_dct['time'] = time
        if pressure:
            other_dct['pressure'] = pressure
        if yield_:
            other_dct['yield'] = yield_

        return other_dct


    @staticmethod
    def _find_closest_cem(textline, parse_str ):
        """Assign closest chemical species to found units (e.g. 'mol%' or 'eq')"""
        phrase = textline

        matches = []
        # cwt = ChemWordTokenizer()
        bracketed_units_pat = re.compile(r'\(\s*'+parse_str+r'\s*\)')
        bracketed_units = re.findall(bracketed_units_pat, phrase)
        if bracketed_units:   # remove brackets
            phrase = re.sub(bracketed_units_pat, ' '.join(bracketed_units[0]), phrase)
        for match in re.finditer(parse_str, phrase):
            match_tokens = match.group(0).split(' ')
            phrase_tokens = phrase.split(' ')
            match_start_idx = [idx for idx, token in enumerate(phrase_tokens) if match_tokens[0] in token][0]
            match_end_idx = [idx for idx, token in enumerate(phrase_tokens) if match_tokens[-1] in token][0]

            species = DisabledNegativeIndices(phrase_tokens)[match_start_idx-2:match_start_idx]
            species = ' '.join(token for token in species).strip('()., ')
            if not species:
                try:
                    species = DisabledNegativeIndices(phrase_tokens)[match_end_idx+1:match_start_idx+4]
                    # filter special signs and digits
                    species = map(lambda s: s.strip('., '), species)
                    species = filter(lambda token: token.isalpha(), species)
                    species = ' '.join(token for token in species)
                except IndexError:
                    log.debug('Closest CEM not found for a catalyst/coreactant key phrase')
                    species = ''

            if species:
                matches.append({'Species': species, 'Value': float(match.group(1)), 'Units': match.group(2)})

        return matches



    # @staticmethod
    # def _find_closest_cem(sentence, parse_str):
    #     """Assign closest chemical species to found units (e.g. 'mol%' or 'eq')"""
    #     phrase = sentence.text
    #     matches = []
    #     # cwt = ChemWordTokenizer()
    #     bracketed_units_pat = re.compile(r'\(\s*'+parse_str+r'\s*\)')
    #     bracketed_units = re.findall(bracketed_units_pat, sentence.text)
    #     if bracketed_units:   # remove brackets
    #         phrase = re.sub(bracketed_units_pat, ' '.join(bracketed_units[0]), phrase)
    #     for match in re.finditer(parse_str, phrase):
    #         match_tokens = cwt.tokenize(match.group(0))
    #         phrase_tokens = cwt.tokenize(phrase)
    #         match_start_idx = [idx for idx, token in enumerate(phrase_tokens) if match_tokens[0] in token][0]
    #         match_end_idx = [idx for idx, token in enumerate(phrase_tokens) if match_tokens[-1] in token][0]
    #         # To simplify syntax above, introduce a new tokeniser that splits full stops more consistently
    #         # Accept two tokens, strip commas and full stops, especially if one of the tokens
    #         species = DisabledNegativeIndices(phrase_tokens)[match_start_idx-2:match_start_idx]
    #         species = ' '.join(token for token in species).strip('()., ')
    #         if not species:
    #             try:
    #                 species = DisabledNegativeIndices(phrase_tokens)[match_end_idx+1:match_start_idx+4]
    #                 # filter special signs and digits
    #                 species = map(lambda s: s.strip('., '), species)
    #                 species = filter(lambda token: token.isalpha(), species)
    #                 species = ' '.join(token for token in species)
    #             except IndexError:
    #                 log.debug('Closest CEM not found for a catalyst/coreactant key phrase')
    #                 species = ''
    #
    #         if species:
    #             matches.append({'Species': species, 'Value': float(match.group(1)), 'Units': match.group(2)})
    #
    #     return matches

    @staticmethod
    def _filter_species(parsed):
        """ If a chemical species has been assigned as both catalyst or coreactant, and `other species`, remove if from
        the latter. Also remove special cases"""
        coreactants, catalysts, other_species, _ = parsed
        combined = [d['Species'] for d in coreactants] + [d['Species'] for d in catalysts]
        # if not coreactants or catalysts found, return unchanged
        if not combined:
            return other_species

        else:
            unaccounted = []
            combined = ' '.join(combined)
            for species in other_species:
                found = re.search(re.escape(species), combined)  # include individual tokens for multi-token names
                if not found and species != 'M':
                    unaccounted.append(species)
            return list(set(unaccounted))

    @staticmethod
    def _resolve_cems(cems):
        """Deletes partial extractions of chemical entity mentions"""
        cems_copy = cems.copy()
        # spans is ~10-15 elements long at most
        for cem1 in cems:
            for cem2 in cems:
                if cem1 != cem2:
                    if cem1 in cem2:
                        try:
                            cems_copy.remove(cem1)
                        except ValueError:
                            pass
                    elif cem2 in cem1:
                        try:
                            cems_copy.remove(cem2)
                        except ValueError:
                            pass

        return cems_copy

    @staticmethod
    def _parse_time(sentence):  # add conditions to add the parsed data
        t_values = ConditionParser.default_values
        t_units = r'(h(?:ours?)?|m(?:in)?|s(?:econds)?|days?)'
        time_str = re.compile(r'(?<!\w)' + t_values + r'\s?' + t_units + r'(?=$|\s?,)')
        time = re.search(time_str, sentence)
        if time:
            return {'Value': float(time.group(1)), 'Units': time.group(2)}

    @staticmethod
    def _parse_temperature(sentence):
        # The following formals grammars for temperature and pressure are quite complex, but allow to parse additional
        # generic descriptors like 'heat' or 'UHV' in `.group(1)'
        t_units = r'\s?(?:o|O|0|°)C|K'   # match 0C, oC and similar, as well as K

        t_value1 = r'-?\d{1,4}' + r'\s?(?=' + t_units + ')'  # capture numbers only if followed by units
        t_value2 = r'r\.?\s?t\.?'
        t_value3 = r'heat|reflux|room\s?temp'

        # Add greek delta?
        t_or = re.compile('(' + '|'.join((t_value1, t_value2, t_value3)) + ')' + '(' + t_units + ')' + '?', re.I)
        temperature = re.search(t_or, sentence)
        return ConditionParser._form_dict_entry(temperature)

    @staticmethod
    def _form_dict_entry(match):
        if match:
            units = match.group(2) if match.group(2) else 'N/A'
            try:
                return {'Value': float(match.group(1)), 'Units': units}
            except ValueError:
                return {'Value': match.group(1), 'Units': units}   # if value is rt or heat, gram scale etc

    @staticmethod
    def _parse_pressure(sentence):
        p_units = r'(?:m|h|k|M)?Pa|m?bar|atm'   # match bar, mbar, mPa, hPa, MPa and atm

        p_values1 = r'\d{1,4}' + r'\s?(?=' + p_units + ')'  # match numbers only if followed by units
        p_values2 = r'(?:U?HV)|vacuum'

        p_or = re.compile('(' + '|'.join((p_values1, p_values2)) + ')' + '(' + p_units + ')' + '?')
        pressure = re.search(p_or, sentence)
        if pressure:
            units = pressure.group(2) if pressure.group(2) else 'N/A'
            return {'Value': float(pressure.group(1)), 'Units': units}

    @staticmethod
    def _parse_yield(sentence):
        y_units = r'%'   # match 0C, oC and similar, as well as K

        y_value1 = r'\d{1,2}' + r'\s?(?=' + y_units + ')'  # capture numbers only if followed by units
        y_value2 = r'gram scale'

        # Add greek delta?
        y_or = re.compile('(' + '|'.join((y_value1, y_value2)) + ')' + '(' + y_units + ')' + '?')
        y = re.search(y_or, sentence)
        return ConditionParser._form_dict_entry(y)


def clear_conditions_region(fig):
    """Removes connected components belonging to conditions and denoises the figure afterwards

    :param Figure fig: Analysed figure
    :return: new Figure object with conditions regions erased"""

    fig_no_cond = erase_elements(fig, [cc for cc in fig.connected_components
                                       if cc.role == FigureRoleEnum.ARROW or cc.role == FigureRoleEnum.CONDITIONSCHAR])

    area_threshold = fig.get_bounding_box().area / 30000
    # width_threshold = fig.get_bounding_box().width / 200
    noise = [panel for panel in fig_no_cond.connected_components if panel.area < area_threshold]

    return erase_elements(fig_no_cond, noise)


# class Conditions(TextRegion):
#     """
#     This class describes conditions region and associated text
#
#     :param panel: extracted region containing conditions
#     :type panel: Panel
#     :param conditions_dct: dictionary with all parsed conditions
#     :type conditions_dct: dict
#     :param parent_panel: reaction arrow, around which the search for conditions is performed
#     :type parent_panel: SolidArrow
#     :param diags: bounding boxes of all chemical structures found in the region
#     :type diags: list[Panel]
#     """
#
#     def __init__(self, panel, conditions_dct, parent_panel=None, text=None, diags=None, _prior_class=None):
#         self.panel = panel
#         self.text = text
#         self.conditions_dct = conditions_dct
#
#         self._prior_class = _prior_class
#
#         if diags is None:
#             diags = []
#         self._diags = diags
#
#         self._parent_panel = parent_panel
#         # if parent_panel:
#         #     parent_panel.children.append(self)
#
#
#
#     @property
#     def arrow(self):
#         return self._parent_panel
#
#     def __repr__(self):
#         return f'Conditions({self.panel}, {self.conditions_dct}, {self.arrow})'
#
#     def __str__(self):
#         delimiter = '\n------\n'
#         return delimiter + 'Step conditions:' + \
#                '\n'.join(f'{key} : {value}' for key, value in self.conditions_dct.items() if value)  + delimiter
#
#     def __eq__(self, other):
#         if other.__class__ == self.__class__:
#             return self.panel == other.panel
#         else:
#             return False
#
#     def __hash__(self):
#         return hash(sum(self.panel.coords))
#
#     @property
#     def diags(self):
#         return self._diags
#
#     @property
#     def anchor(self):
#         a_pixels = self.arrow.pixels
#         return a_pixels[len(a_pixels)//2]
#
#     @property
#     def coreactants(self):
#         return self.conditions_dct['coreactants']
#
#     @property
#     def catalysts(self):
#         return self.conditions_dct['catalysts']
#
#     @property
#     def other_species(self):
#         return self.conditions_dct['other species']
#
#     @property
#     def temperature(self):
#         return self.conditions_dct['temperature']
#
#     @property
#     def time(self):
#         return self.conditions_dct['time']
#
#     @property
#     def pressure(self):
#         return self.conditions_dct['pressure']
#
#     @property
#     def yield_(self):
#         return self.conditions_dct['yield']
#
#     def merge_conditions_regions(self, other_region):
#         keys = self.conditions_dct.keys()
#         new_dict = {}
#         for k in keys:
#             if isinstance(self.conditions_dct[k], Sequence):
#                 new_value = self.conditions_dct[k] + other_region.conditions_dct[k]
#             else:
#                 val = self.conditions_dct[k]
#                 new_value = val if val else other_region.conditions_dct[k]
#             new_dict[k] = new_value
#         panel = self.panel.create_megapanel([self.panel, other_region.panel], fig=self.panel.fig)
#         text = self.text + other_region.text
#         diags = self._diags + other_region._diags
#
#         return Conditions(panel=panel, conditions_dct=new_dict, parent_panel=self._parent_panel, text=text,diags=diags,
#                           _prior_class=self._prior_class)