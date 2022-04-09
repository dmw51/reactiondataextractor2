# -*- coding: utf-8 -*-
"""
Output
=======

This module contains classes used for representing the output of extraction procedures.

author: Damian Wilary
email: dmw51@cam.ac.uk

"""
import copy
from abc import ABC, abstractmethod
from collections import Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json

from sklearn.cluster import DBSCAN

from .geometry import Line, Point
from .reaction import Diagram, ReactionStep
from .segments import ReactionRoleEnum
# from .utils import Point, PrettyFrozenSet, PrettyList
from ..actions import find_nearby_ccs #extend_line
from .reaction import Conditions
# from .. import settings

import matplotlib

from ..config import SchemeConfig
from ..utils import PrettyFrozenSet, find_points_on_line, euclidean_distance


class Graph(ABC):
    """
    Generic directed graph class

    :param graph_dict: underlying graph mapping
    :type graph_dict: dict
    """

    def __init__(self, graph_dict=None):
        if graph_dict is None:
            graph_dict = {}
        self._graph_dict = graph_dict

    @abstractmethod
    def _generate_edge(self, *args) :
        """
        This method should update the graph dict with a connection between vertices,
        possibly adding some edge annotation.
        """
        return NotImplemented

    @abstractmethod
    def edges(self):
        """
        This method should return all edges (partially via invoking the `_generate_edge` method).
        """
        return NotImplemented

    @abstractmethod
    def __str__(self):
        """
        A graph needs to have a __str__ method to constitute a valid output representation.
        """
    @property
    def nodes(self):
        return self._graph_dict.keys()

    def add_vertex(self, vertex):
        if vertex not in self._graph_dict:
            self._graph_dict[vertex] = []

    def find_isolated_vertices(self):
        """
        Returns all isolated vertices. Can be used for output validation
        :return: collection of isolated (unconnected) vertices
        """
        graph = self._graph_dict
        return [key for key in graph if graph[key] == []]

    def find_path(self, vertex1, vertex2, path=None):
        if path is None:
            path = []
        path += [vertex1]
        graph = self._graph_dict
        if vertex1 not in graph:
            return None
        if vertex2 in graph[vertex1]:
            return path + [vertex2]
        else:
            for value in graph[vertex1]:
                return self.find_path(value, vertex2, path)


class ReactionScheme(Graph):
    """Main class used for representing the output of an extraction process

    :param arrows: all extracted reaction arrows
    :type arrows: list[BaseArrow]
    :param diags: all extracted chemical diagrams
    :type diags: list[Diagram]
    :param fig: Analysed figure
    :type fig: Figure"""
    def __init__(self, arrows, diags, fig):
        self._conditions = arrows
        self._diags = diags
        super().__init__()
        self._reaction_steps = ([self._scan_form_reaction_step(arrow)
                                           for arrow in arrows])
        # self._pretty_reaction_steps = PrettyList(self._reaction_steps)
        self.create_graph()
        self._start = None  # start node(s) in a graph
        self._end = None   # end node(s) in a graph
        self._fig = fig
        graph = self._graph_dict
        self.set_start_end_nodes()

        self._single_path = True if len(self._start) == 1 and len(self._end) == 1 else False

    def edges(self):
        if not self._graph_dict:
            self.create_graph()

        return {k: v for k, v in self._graph_dict.items()}

    def _generate_edge(self, key, successor):

        self._graph_dict[key].append(successor)

    def __repr__(self):
        return f'ReactionScheme({self._reaction_steps})'

    def __str__(self):
        # if self._single_path:
        #     path = self.find_path(self.reactants, self.products)
        #     return '  --->  '.join((' + '.join(str(species) for species in group)) for group in path)
        # else:
        return '\n'.join([str(reaction_step) for reaction_step in self._reaction_steps])

    def __eq__(self, other):
        if isinstance(other, ReactionScheme):
            return other._graph_dict == self._graph_dict
        return False

    @property
    def reaction_steps(self):
        return self._reaction_steps

    @property
    def graph(self):
        return self._graph_dict

    @property
    def reactants(self):
        return self._start

    @property
    def products(self):
        return self._end

    def long_str(self):
        """Longer str method - contains more information (eg conditions)"""
        return f'{self._reaction_steps}'

    def draw_segmented(self, out=False):
        """Draw the segmented figure. If ``out`` is True, the figure is returned and can be saved"""
        y_size, x_size = self._fig.img.shape
        f, ax = plt.subplots(figsize=(x_size/100, y_size/100))
        ax.imshow(self._fig.img, cmap=plt.cm.binary)
        params = {'facecolor': 'g', 'edgecolor': None, 'alpha': 0.3}
        for step_conditions in self._conditions:
            panel = step_conditions.arrow.panel
            rect_bbox = Rectangle((panel.left, panel.top), panel.right - panel.left, panel.bottom - panel.top,
                                  facecolor='y', edgecolor=None, alpha=0.4)
            ax.add_patch(rect_bbox)

            for t in step_conditions.text_lines:
                panel = t.panel
                rect_bbox = Rectangle((panel.left - 1, panel.top - 1), panel.right - panel.left,
                                      panel.bottom - panel.top, **params)
                ax.add_patch(rect_bbox)
            # for panel in step_conditions.structure_panels:
            #     rect_bbox = Rectangle((panel.left - 1, panel.top - 1), panel.right - panel.left,
            #                           panel.bottom - panel.top, **params)
            #     ax.add_patch(rect_bbox)

        params = {'facecolor': (66 / 255, 93 / 255, 166 / 255),
                  'edgecolor': (6 / 255, 33 / 255, 106 / 255),
                  'alpha': 0.4}
        for diag in self._diags:
            panel = diag.panel
            rect_bbox = Rectangle((panel.left, panel.top), panel.right - panel.left, panel.bottom - panel.top,
                                  facecolor=(52/255, 0, 103/255), edgecolor=(6/255, 0, 99/255), alpha=0.4)
            ax.add_patch(rect_bbox)
            if diag.label:
                panel = diag.label.panel
                rect_bbox = Rectangle((panel.left - 1, panel.top - 1), panel.right - panel.left,
                                      panel.bottom - panel.top, **params)
                ax.add_patch(rect_bbox)
        ax.axis('off')
        if out:
            return f
        else:
            plt.show()

    def create_graph(self):
        """
        Unpack reaction steps to create a graph from individual steps
        :return: completed graph dictionary
        """
        graph = self._graph_dict
        for step in self._reaction_steps:
            [self.add_vertex(PrettyFrozenSet(species_group)) for species_group in step]
            self.add_vertex(step.conditions)

        for step in self._reaction_steps:
            self._generate_edge(step.reactants, step.conditions)
            self._generate_edge(step.conditions, step.products)

        return graph

    def set_start_end_nodes(self):
        """
        Finds and return the first vertex in a graph (group of reactants). Unpack all groups from ReactionSteps into
        a Counter. The first group is a group that is counted only once and exists as a key in the graph dictionary.
        Other groups (apart from the ultimate products) are counted twice (as a reactant in one step and a product in
        another).
        """
        group_count = Counter(group for step in self._reaction_steps for group in (step.reactants, step.products))
        self._start = [group for group, count in group_count.items() if count == 1 and
                       all(species.role == ReactionRoleEnum.STEP_REACTANT for species in group)]

        self._end = [group for group, count in group_count.items() if count == 1 and
                     all(species.role == ReactionRoleEnum.STEP_PRODUCT for species in group)]

    def find_path(self, group1, group2, path=None):
        """ Recursive routine for simple path finding between reactants and products"""
        graph = self._graph_dict
        if path is None:
            path = []
        path += [group1]
        if group1 not in graph:
            return None

        successors = graph[group1]
        if group2 in successors:
            return path+[group2]
        else:
            for prod in successors:
                return self.find_path(prod, group2, path=path)
        return None

    def to_json(self):
        # reactions = [self._json_generic_recursive(start_node) for start_node in self._start]
        json_dict = {}

        nodes = {label: node for label, node in zip(map(str, range(50)), self.nodes)}
        json_dict['node_labels'] = nodes
        adjacency = {}
        for node1, out_nodes in self.graph.items():
            node1_label = [label for label, node in nodes.items() if node == node1][0]
            out_nodes_labels = [label for label, node in nodes.items() if node in out_nodes]
            
            adjacency[node1_label] = out_nodes_labels
        json_dict['adjacency'] = adjacency

        for label, node in json_dict['node_labels'].items():
            if hasattr(node, '__iter__'):
                contents = []
                for diagram in node:
                    if diagram.label:
                        content = {'smiles': diagram.smiles, 'label': [sent.text.strip() for sent in diagram.label.text ]}
                    else:
                        content = {'smiles': diagram.smiles, 'label': None}
                    contents.append(content)
                json_dict['node_labels'][label] = contents
            elif isinstance(node, Conditions):
                contents = node.conditions_dct
                json_dict['node_labels'][label] = contents

        return json.dumps(json_dict, indent=4)

    # def _json_generic_recursive(self, start_key, json_obj=None):
    #     """
    #     Generic recursive json string generator. Takes in a single ``start_key`` node and builds up the ``json_obj`` by
    #     traverding the reaction graph
    #     :param start_key: node where the traversal begins (usually the 'first' group of reactants in the reactions)
    #     :param json_obj: a dictionary created in the recursive procedure (ready for json dumps)
    #     :return:  dict; the created ``json_obj``
    #     """
    #     graph = self._graph_dict
    #
    #     if json_obj is None:
    #         json_obj = {}
    #
    #     node = start_key
    #
    #     if hasattr(node, '__iter__'):
    #         contents = [{'smiles': species.smiles, 'label': str(species.label)} for species in node]
    #     else:
    #         contents = str(node)   # Convert the conditions_dct directly
    #
    #     json_obj['contents'] = contents
    #     successors = graph[node]
    #     if not successors:
    #         json_obj['successors'] = None
    #         return json_obj
    #     else:
    #         json_obj['successors'] = []
    #         for successor in successors:
    #             json_obj['successors'].append(self._json_generic_recursive(successor))
    #
    #     return json_obj

    def to_smirks(self, start_key=None, species_strings=None):
        """
        Converts the reaction graph into a SMIRKS (or more appropriately - reaction SMILES, its subset). Also outputs
        a string containing auxiliary information from the conditions' dictionary.
        :param start_key: node where the traversal begins (usually the 'first' group of reactants in the reactions)
        :param species_strings: list of found smiles strings (or chemical formulae) built up in the procedure and ready
        for joining into a single SMIRKS string.
        :return: (str, str) tuple containing a (reaction smiles, auxiliary info) pair
        """
        if not self._single_path:
            return NotImplemented  # SMIRKS only work for single-path reaction

        graph = self._graph_dict

        if start_key is None:
            start_key = self._start[0]

        if species_strings is None:
            species_strings = []

        node = start_key

        if hasattr(node, '__iter__'):  # frozenset of reactants or products
            species_str = '.'.join(species.smiles for species in node)
        else:  # Conditions object
            # The string is a sum of coreactants, catalysts (which have small dictionaries holding names and values/units)
            species_vals = '.'.join(species_dct['Species'] for group in iter((node.coreactants, node.catalysts))
                                    for species_dct in group)
            # and auxiliary species with simpler structures (no units)
            species_novals = '.'.join(group for group in node.other_species)
            species_str = '.'.join(filter(None, [species_vals, species_novals]))

        species_strings.append(species_str)

        successors = graph[node]
        if not successors:
            smirks ='>'.join(species_strings)
            return smirks
        else:
            return self.to_smirks(successors[0], species_strings)

        # return smirks, [node.conditions_dct for node in graph if isinstance(node, Conditions)]

    def _scan_form_reaction_step(self, arrow):
        """
        Scans an image around a single arrow to give reactants and products in a single reaction step
        :param BaseArrow arrow: BaseArrow object or a child class instance around which the scan is performed
        :return: arrows and diagrams packed inside a reaction step
        :rtype: ReactionStep
        """
        # arrow = step_conditions.arrow
        diags = self._diags

        # endpoint1, endpoint2 = extend_line(step_conditions.arrow.line,
        #                                    extension=arrow.pixels[0].separation(arrow.pixels[-1]) * 0.75)
        # react_side_point = step_conditions.arrow.react_side[0]
        # endpoint1_close_to_react_side = endpoint1.separation(react_side_point) < endpoint2.separation(react_side_point)
        # if endpoint1_close_to_react_side:
        #     react_endpoint, prod_endpoint = endpoint1, endpoint2
        # else:
        #     react_endpoint, prod_endpoint = endpoint2, endpoint1

        react_endpoint, prod_endpoint = arrow.center, arrow.reference_pt

        initial_distance = SchemeConfig.SEARCH_DISTANCE_FACTOR * np.sqrt(np.mean([panel.area for panel in diags]))
        # extended_distance = 4 * np.sqrt(np.mean([diag.panel.area for diag in diags]))
        extended_distance = initial_distance * 2
        distance_fn = lambda panel: initial_distance
        distances = initial_distance, distance_fn
        extended_distances = extended_distance, distance_fn
        nearby_diags = find_nearby_ccs(arrow.panel, diags, distances)

        # reactants = find_nearby_ccs(react_endpoint, diags, distances,
        #                             condition=lambda diag: diag.panel.role != ReactionRoleEnum.CONDITIONS)
        reactants, products = self.separate_ccs(nearby_diags, react_endpoint, prod_endpoint)
        # if not reactants:
        #     reactants = find_nearby_ccs(react_endpoint, diags, extended_distances,
        #                                 condition=lambda diag: diag.panel.role != ReactionRoleEnum.CONDITIONS)
        # if not reactants:
        #     reactants = self._search_elsewhere('up-right', step_conditions.arrow, distances)
        #
        # products = find_nearby_ccs(prod_endpoint, diags, distances,
        #                            condition=lambda diag: diag.panel.role != ReactionRoleEnum.CONDITIONS)
        # #
        # if not products:
        #     products = find_nearby_ccs(prod_endpoint, diags, extended_distances,
        #                                condition=lambda diag: diag.panel.role != ReactionRoleEnum.CONDITIONS)
        #
        # if not products:
        #     products = self._search_elsewhere('down-left', step_conditions.arrow, distances)
        #
        [setattr(reactant, 'role', ReactionRoleEnum.STEP_REACTANT) for reactant in reactants]
        [setattr(product, 'role', ReactionRoleEnum.STEP_PRODUCT) for product in products]

        return ReactionStep(reactants, products, conditions=arrow.conditions)

    def separate_ccs(self, ccs, point1, point2):
        """Separates Panel objects in `ccs` into two groups based on proximity to one of the two points `point1` and
        `point2. Distance between each point and the closest edge or corner of a panel is used (this normalises size
        of each panel)."""
        pts = [point1, point2]
        clusters = [[], []]
        for cc in ccs:
            dists = [(idx, cc.edge_separation(pts[idx])) for idx in range(2)]
            min_idx = min(dists, key=lambda x: x[1])[0]
            clusters[min_idx].append(cc)
        if any(c == [] for c in clusters):  # no reactants or products found
            pass
            # Come up with a solution - when such a situation happens? Anything other than when a scheme continues
            # in the next line? Could use old version for this

        return clusters



    def _search_elsewhere(self, where, arrow, distances):
        """
        Looks for structures in a different line of a multi-line reaction scheme.

        If a reaction scheme ends unexpectedly either on the left or right side of an arrows (no species found), then
        a search is performed in the previous or next line of a reaction scheme respectively (assumes multiple lines
        in a reaction scheme). Assumes left-to-right reaction scheme. Estimates the optimal alternative search point
        using arrow and diagrams' coordinates in a DBSCAN search.
        This gives clusters corresponding to the multiple lines in a reaction scheme. Performs a search in the new spot.
        :param str where: Allows either 'down-left' to look below and to the left of arrow, or 'up-right' (above to the right)
        :param Arrow arrow: Original arrow, around which the search failed
        :param (float, lambda) distances: pair containing initial search distance and a distance function (usually same as
        in the parent function)
        :return: Collection of found species
        :rtype: list[Diagram]
        """
        assert where in ['down-left', 'up-right']
        fig = settings.main_figure[0]
        diags = self._diags

        X = np.array([s.center[1] for s in diags] + [arrow.panel.center[1]]).reshape(-1, 1)  # the y-coordinate
        eps = np.mean([s.height for s in diags])
        dbscan = DBSCAN(eps=eps, min_samples=2)
        y = dbscan.fit_predict(X)
        num_labels = max(y) - min(y) + 1  # include outliers (label -1) if any
        arrow_label = y[-1]
        clustered = []
        for val in range(-1, num_labels):
            if val == arrow_label:
                continue  # discard this cluster - want to compare the arrow with other clusters only
            cluster = [centre for centre, label in zip(X, y) if label == val]
            if cluster:
                clustered.append(cluster)
        centres = [np.mean(cluster) for cluster in clustered]
        centres.sort()
        if where == 'down-left':
            move_to_vertical = [centre for centre in centres if centre > arrow.panel.center[1]][0]
            move_to_horizontal = np.mean([structure.width for structure in diags])
        elif where == 'up-right':
            move_to_vertical = [centre for centre in centres if centre < arrow.panel.center[1]][-1]
            move_to_horizontal = fig.img.shape[1] - np.mean([structure.width for structure in diags])
        else:
            raise ValueError("'where' takes in one of two values : ('down-left', 'up-right') only")
        species = find_nearby_ccs(Point(move_to_vertical, move_to_horizontal), diags, distances)

        return species

class RoleProbe:
    """This is a class used to probe reaction schemes around arrows to assign roles to diagrams and reconstruct
    the reaction in a machine-readable format"""

    def __init__(self, fig, arrows, diagrams):
        self.fig = fig
        self.arrows = arrows
        self.diagrams = diagrams
        self.stepsize = min([x for diag in self.diagrams for x in (diag.panel.width, diag.panel.height)]) #Step size should be related to width/height of the smallest diagram, whichever is smaller
        # Could also be a function depending on arrow direction, but might not be necessary
        self.segment_length = np.mean([(d.panel.width + d.panel.height) / 2 for d in self.diagrams])
        # This should be comparable to the largest dim of the largest diagrams, but might not be
                                # stable to outliers


    def probe_around_arrow(self, arrow):
        # center, direction_normal = self.find_normal_to_arrow(arrow)
        # center = np.asarray(center)
        x_one, y_one = arrow.center
        x_two, y_two = self.fig.img.shape[1] - arrow.center[0], self.fig.img.shape[0] - arrow.center[1]

        region_one_dims = (x_one, y_one)
        regions_two_dims = (x_two, y_two)
        diags_one = self._probe_around_arrow(arrow, region_one_dims, switch=-1)
        diags_two = self._probe_around_arrow(arrow, regions_two_dims, switch=+1)
        diags_react, diags_prod = self.assign_diags(diags_one, diags_two, arrow)
        return ReactionStep(arrow, reactants=diags_react, products=diags_prod)
        # return diags_one, diags_two


    def assign_diags(self, group1, group2, arrow):
        ref_point = arrow.reference_pt # arrow's center of mass which denoted the products' side
        groups = [group1, group2]

        def compute_ref_group_dist(group, pt):
            group_centre = np.mean([member.center for member in group], axis=0)
            return euclidean_distance(group_centre, pt)
        prod_group = min(groups, key=lambda x: compute_ref_group_dist(x, ref_point))
        groups.remove(prod_group)
        react_group = groups[0]
        return react_group, prod_group



        # (x, y), (MA, ma), angle = cv2.fitEllipse(arrow.contour)
        # angle = angle - 90  # Angle should be anti-clockwise relative to +ve x-axis
        # normal_angle = angle + 90
        # center = np.asarray([x, y])
        # direction = np.asarray([1, np.tan(np.radians(angle))])
        # direction_normal = np.asarray([1, np.tan(np.radians(normal_angle))])
        #
        # #Check both directions separately, and choose the one where image edge is reached quicker
        # # This should be done differently, the centers should lie along the arrow line (arrow direction) and hence the
        # # step in y direction is equal to stepsize * slope of arrow (should work in extreme cases too since we choose the lower number?)
        # stepsize_x = self.stepsize * direction[0] ## direction[0] is set to 1 to simplify reasoning
        # stepsize_y = self.stepsize * direction[1]
        # num_centers_one_x = center[0] // stepsize_x
        # num_centers_one_y = center[1] // stepsize_y
        # num_centers_one = int(min(num_centers_one_x, num_centers_one_y))
        # deltas = np.array([[stepsize_x * n, stepsize_y * n] for n in range(1, num_centers_one+1)])
        # centers_one = center - deltas
        # lines_one = [Line(find_points_on_line(center, direction_normal, distance=self.segment_length))
        #          for center in centers_one]
        #
        # ## Visualize lines ##
        # # import matplotlib.pyplot as plt
        # # plt.imshow(self.fig.img)
        # # for line in lines_one:
        # #     (x1, y1), (x2, y2) = line.endpoints
        # #     plt.plot([x1, x2], [y1, y2], c='r')
        # # plt.show()
        #
        # try:
        #     other_arrows = copy.copy(self.arrows)
        #     other_arrows.remove(arrow)
        #     arrow_overlap = [any(self._check_overlap(l, a.panel) for a in other_arrows) for l in lines_one].index(True)
        # except ValueError:
        #     arrow_overlap = None
        #
        # if arrow_overlap is not None:
        #     lines_one = lines_one[:arrow_overlap]
        # diags_one = []
        # for l in lines_one:
        #     for d in self.diagrams:
        #         if self.sufficient_overlap(l, d.panel):
        #             diags_one.append(d)
        # diags_one = list(set(diags_one))
        # ##### REVISION FINISHED - the conditions diag was not classified based on overlap.
        # ## It should be excluded but maybe the overlap criteria should be less restrictive in general?
        # ### Wrap the above code inside a function and duplicate for the other side
        #
        #
        # ## Check any overlap overlaps with arrows, discard all lines after the first overlap
        # ## Then check overlaps with diagrams and include if sufficient overlap
        #
        # num_centers_two_x = (self.fig.img.shape[1] - center[0]) // stepsize_x
        # num_centers_two_y = (self.fig.img.shape[0] - center[1]) // stepsize_y
        # num_centers_two = int(min(num_centers_two_x, num_centers_two_y))
        # deltas = np.array([[stepsize_x * n, stepsize_y * n] for n in range(1, num_centers_two+1)])
        # centers_two = center + deltas
        # lines_two = [Line(find_points_on_line(center, direction_normal, distance=self.segment_length))
        #          for center in centers_two]
        #
        # try:
        #     other_arrows = copy.copy(self.arrows)
        #     other_arrows.remove(arrow)
        #     arrow_overlap = [any(self._check_overlap(l, a.panel) for a in other_arrows) for l in lines_two].index(True)
        # except ValueError:
        #     arrow_overlap = None
        #
        # if arrow_overlap is not None:
        #     lines_two = lines_two[:arrow_overlap]
        #
        # diags_two = []
        # for l in lines_two:
        #     for d in self.diagrams:
        #         if self.sufficient_overlap(l, d.panel):
        #             diags_two.append(d)
        #
        # diags_two = list(set(diags_two))
        #
        # return diags_one, diags_two

    def find_normal_to_arrow(self, arrow):
        (x, y), (MA, ma), angle = cv2.fitEllipse(arrow.contour)
        angle = angle - 90  # Angle should be anti-clockwise relative to +ve x-axis
        normal_angle = angle + 90
        center = np.asarray([x, y])
        direction_normal = np.asarray([1, np.tan(np.radians(normal_angle))])
        return center, direction_normal
        p_n1, p_n2 = find_points_on_line(center, direction_normal, distance=self.segment_length)
        return Line.approximate_line(p_n1, p_n2)

    def sufficient_overlap(self, segment, panel):
        len_segment = euclidean_distance(*segment.endpoints)
        t, l, b, r = panel
        d_diag = euclidean_distance((t,l), (b, r))
        min_overlap = min(len_segment, d_diag) * SchemeConfig.MIN_PROBING_OVERLAP_FACTOR

        return self._check_overlap(segment, panel) > min_overlap


    def _check_overlap(self, segment, panel):
        """Checks overlap between a line segment and a panel. The overlap is defined as a common line segment between
        the bounding box and the probing segment

        This can be viewed as looking for an overlap between two rectangles, one defined by the diagram's panel,
        ane the other by the two endpoints of a segment (which can be seen generically as the top-left, and bottom-right
        corners)"""

        t1, l1, b1, r1 = panel
        p1, p2 = segment.endpoints
        # Define the second rectangle
        t2, b2 = min(p1[1], p2[1]), max(p1[1], p2[1])
        l2, r2 = min(p1[0], p2[0]), max(p1[0], p2[0])

        # Define intersection
        t = max(t1, t2)
        b = min(b1, b2)
        l = max(l1, l2)
        r = min(r1, r2)

        height = b - t
        width = r - l
        if height < 0 or width < 0:
            overlap = 0
        else:
            overlap = np.sqrt(height**2 + width**2)

        return overlap

    def _probe_around_arrow(self, arrow , region_dims, switch):
        """Finds diagrams around an arrow within a region.

        Each arrow divides an image into two regions: one in which potential reactants are located, and one where
        potential products are located (potential, because a reaction might involve multiple steps and not all species
        take part in a given step. We perform a line scan in both regions along the direction dictated by an arrow.
        To achieve this, we create equidistant lines and we control the direction of search propagation using a switch
        value of -1 or +1 to compute required differences in position from the arrow centre.
        #TODO: Document this fully
        """
        assert switch in [-1, 1]
        # center = arrow.center
        region_x_length, region_y_length = region_dims

        (x, y), (MA, ma), angle = cv2.fitEllipse(arrow.contour)
        angle = angle - 90  # Angle should be anti-clockwise relative to +ve x-axis
        normal_angle = angle + 90
        center = np.asarray([x, y])
        direction = np.asarray([1, np.tan(np.radians(angle))])
        direction = direction / np.linalg.norm(direction)
        direction_normal = np.asarray([1, np.tan(np.radians(normal_angle))])

        stepsize_x = self.stepsize * direction[0]
        stepsize_y = self.stepsize * direction[1]
        num_centers_x = abs(region_x_length // stepsize_x)
        num_centers_y = abs(region_y_length // stepsize_y)
        num_centers = int(min(num_centers_x, num_centers_y))
        deltas = switch * np.array([[stepsize_x * n, stepsize_y * n] for n in range(1, num_centers + 1)])
        centers = center + deltas
        lines = [Line(find_points_on_line(center, direction_normal, distance=self.segment_length))
                     for center in centers]

        ## Visualize lines ##
        # import matplotlib.pyplot as plt
        # plt.imshow(self.fig.img)
        # for line in lines_one:
        #     (x1, y1), (x2, y2) = line.endpoints
        #     plt.plot([x1, x2], [y1, y2], c='r')
        # plt.show()

        try:
            other_arrows = copy.copy(self.arrows)
            other_arrows.remove(arrow)
            arrow_overlap = [any(self._check_overlap(l, a.panel) for a in other_arrows) for l in lines].index(True)
        except ValueError:
            arrow_overlap = None

        if arrow_overlap is not None:
            lines = lines[:arrow_overlap]
        diags = []
        for l in lines:
            for d in self.diagrams:
                if self.sufficient_overlap(l, d.panel):
                    diags.append(d)
        diags = list(set(diags))

        return diags
