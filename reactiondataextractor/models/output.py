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
from collections import Counter, namedtuple
from collections.abc import Iterable
import numpy as np
import json
from typing import List, Tuple, Union

import cv2
from sklearn.cluster import DBSCAN

from reactiondataextractor.models.exceptions import SchemeReconstructionFailedException
from reactiondataextractor.models.geometry import Line
from reactiondataextractor.models.reaction import Diagram, ReactionStep, CurlyArrow, Conditions
from reactiondataextractor.models.segments import ReactionRoleEnum, Rect, Figure
from reactiondataextractor.configs.config import SchemeConfig
from reactiondataextractor.utils.utils import find_points_on_line, euclidean_distance, skeletonize

ConditionsPlaceholder = namedtuple('ConditionsPlaceholder', ['panel', 'text', 'conditions_dct'])
class Graph(ABC):
    """Generic directed graph class
    """

    def __init__(self):
        """:param nodes: a mapping from node index to contents of the node
        :type graph_dict: dict
        :param adjacency: a dict with a key for each node index with outgoing connections and a value specifying all connections from this node
        :param adjacency: dict"""
        self.nodes = {}
        self.adjacency = {}
        self._node_idx = 0

    @abstractmethod
    def _generate_edge(self, *args):
        """
        This method should update the graph dict with a connection between vertices,
        possibly adding some edge annotation.
        """
        return NotImplemented

    @abstractmethod
    def edges(self):
        """
        This method should return all edges.
        """
        return NotImplemented

    @abstractmethod
    def __str__(self):
        """
        A graph needs to have a __str__ method to constitute a valid output representation.
        """

    def add_node(self, node):
        if frozenset(node) not in self.nodes:
            self.nodes[frozenset(node)] = self._node_idx
            self._node_idx += 1

    def find_isolated_vertices(self):
        """
        Returns all isolated vertices. Can be used for output validation
        :return: collection of isolated (unconnected) vertices
        """
        nodes_indices = set(self.nodes.keys())

        # Isolated means no outgoing edges and no incoming edges
        no_out_nodes = set(nodes_indices.difference(set(self.adjacency.keys())))
        all_incoming_connections = set([v for connections in self.adjacency.values() for v in connections])
        unconnected = no_out_nodes.difference(all_incoming_connections)
        return unconnected

    def find_path(self, node1, node2, path=None):
        #TODO: Refactor this to match the new graph definition
        if path is None:
            path = []
        path += [node1]
        graph = self._graph_dict
        if node1 not in graph:
            return None
        if node2 in graph[node1]:
            return path + [node2]
        else:
            for value in graph[node1]:
                return self.find_path(value, node2, path)


class ReactionScheme(Graph):
    """Main class used for representing the output of an extraction process
    """
    def __init__(self, fig: 'Figure', reaction_steps: List['ReactionStep'], is_incomplete: bool):
        """
        :param fig: analysed figure
        :type fig: Figure
        :param reaction_steps: list of all detected reaction steps
        :type reaction_steps: List[ReactionStep]
        :param is_incomplete: whether all reaction steps have been successfully extracted (False if complete extraction)
        :type is_incomplete: bool
        """
        super().__init__()
        self.included_diags = []
        self._reaction_steps = reaction_steps
        self.is_incomplete = is_incomplete
        self.create_graph()
        self._fig = fig

    def edges(self):
        return self.adjacency

    def _generate_edge(self, key, successor):
        key = self.nodes[frozenset(key)]
        successor = self.nodes[frozenset(successor)]

        if self.adjacency.get(key):
            self.adjacency[key].append(successor)
        else:
            self.adjacency[key] = [successor]

    def __repr__(self):
        return f'ReactionScheme({self._reaction_steps})'

    def __str__(self):
        return '\n'.join([str(reaction_step) for reaction_step in self._reaction_steps])

    def __iter__(self):
        return self

    # TODO: Implement an interator protocol to probe the internal graph structur
    def __next__(self, next_node=None):
        pass

    def add_node(self, node):
        if frozenset(node) not in self.nodes:
            self.nodes[frozenset(node)] = self._node_idx
            self._node_idx += 1
            if isinstance(node, Iterable):
                for elem in node:
                    if isinstance(elem, Diagram):
                        self.included_diags.append(elem)


    @property
    def reaction_steps(self):
        return self._reaction_steps

    @property
    def reactants(self):
        return self._start

    @property
    def products(self):
        return self._end

    def long_str(self):
        """Longer str method - contains more information (eg conditions)"""
        return f'{self._reaction_steps}'

    def create_graph(self):
        """
        Unpack reaction steps to create a graph from individual steps
        :return: completed graph dictionary
        """

        for step in self._reaction_steps:
            conditions = step.conditions if step.conditions != [] else [ConditionsPlaceholder(panel=step.arrow.panel, text='', conditions_dct=None)]
            self.add_node(step.reactants)
            self.add_node(conditions)
            self.add_node(step.products)


        for step in self._reaction_steps:
            conditions = step.conditions if step.conditions != [] else [ConditionsPlaceholder(panel=step.arrow.panel, text='', conditions_dct=None)]
            self._generate_edge(step.reactants, conditions)
            self._generate_edge(conditions, step.products)

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
        adjacency = self.adjacency
        nodes = []
        for node, value in self.nodes.items():
            node_dcts = []
            for element in node:
                if isinstance(element, Diagram):  # Either a Diagram or Conditions
                    diag_dct = {'smiles': element.smiles, 'panel': str(element.panel.in_original_fig(as_str=False)),
                                'labels': [label.text for label in element.labels]}
                    node_dcts.append(diag_dct)
                else:
                    node_dcts.append(element.conditions_dct)
            nodes.append(node_dcts)
        json_dct = {'adjacency': adjacency,
                    'nodes': nodes,
                    'is_incomplete': self.is_incomplete}

        return json.dumps(json_dct, indent=4)


class RoleProbe:
    """This is a class used to probe reaction schemes around arrows to assign roles to diagrams and reconstruct
    the reaction in a machine-readable format. This class reconstructs individual reaction steps which are then passed
    to the ReactionScheme class"""

    def __init__(self, fig, arrows, diagrams):
        """:param fig: Analysed figure
        :type fig: Figure
        :param arrows: all extracted arrows
        :type arrows: list[BaseArrow]
        :param diagrams: all found diagrams
        :type diagrams: list[Diagram]
        """
        self.fig = fig
        self.arrows = arrows
        probed_diags = self.remove_reaction_conditions_diags(diagrams)
        self.diagrams = probed_diags

        self.stepsize = min([x for diag in self.diagrams for x in (diag.panel.width, diag.panel.height)]) * 0.2 #Step size should be related to width/height of the smallest diagram, whichever is smaller
        # Could also be a function depending on arrow direction, but might not be necessary
        self.segment_length = np.mean([(d.panel.width + d.panel.height) / 2 for d in self.diagrams]) // 2
        # This should be comparable to the largest dim of the largest diagrams, but might not be
                                # stable to outliers
        self.reaction_steps = []
        self.is_incomplete = False

    def remove_reaction_conditions_diags(self, diags):
        cond_diags = []
        for arrow in self.arrows:
            for child in arrow.children:
                if isinstance(child, Diagram):
                    cond_diags.append(child)
                elif isinstance(child, Conditions):
                    if child.diags:
                        cond_diags.extend(child.diags)
        return [diag for diag in diags if diag not in cond_diags]  

    def probe(self):
        unique_arrows = []
        for arrow1 in self.arrows:
            arrow_cluster = []
            for arrow2 in self.arrows:
                if arrow1.panel.edge_separation(arrow2.panel) < 30:
                    arrow_cluster.append(arrow2)
            arrow_cluster.sort(key=lambda arrow: arrow.panel.left)
            unique_arrows.append(arrow_cluster)
        arrows = [c[0] for c in unique_arrows]
        self.arrows = list(set(arrows))
        [self.probe_around_arrow(a) for a in self.arrows]
          
    def probe_around_arrow(self, arrow):
        """Main probing method.

        Establishes the direction of an arrow and probes its surroundings. Performs linear scanning in the two regions
        corresponding to step reactants and products to gather them. If one group is not found, assumes a step
        spread over multiple lines and performs the scanning accordingly. Finally, finds which region corresponds to
        reactants and which to products.
        :param arrow: arrow around which the probing is performed
        :type arrow: BaseArrow
        :return: an object containing step reactants, products, and the arrow
        :rtype: ReactionStep"""

        if isinstance(arrow, CurlyArrow):
            single_line = False
            biggest_diag = max(self.diagrams, key=lambda diag: diag.panel.area)
            scan_region_dims = biggest_diag.panel.width * 1.25, biggest_diag.panel.height * 1.25
            try:
                scan_params = self._compute_scan_direction_curly_arrow(arrow)
            except ValueError:
                return
            diags_react = self._perform_scan(arrow, scan_region_dims, scan_params[0][0], scan_params[0][1],
                                                switch=+1)
            diags_prod = self._perform_scan(arrow, scan_region_dims, scan_params[1][0], scan_params[1][1],
                                                switch=+1)
            
            if not (diags_react and diags_prod):
                self.is_incomplete = True
                return
            
        else:
            x_one, y_one = arrow.center
            x_two, y_two = self.fig.img.shape[1] - arrow.center[0], self.fig.img.shape[0] - arrow.center[1]

            region_one_dims = (x_one, y_one)
            regions_two_dims = (x_two, y_two)
            direction, direction_normal = self._compute_arrow_scan_params(arrow)
            diags_one = self._perform_scan(arrow, region_one_dims, arrow.center, direction,
                                                switch=-1)
            diags_two = self._perform_scan(arrow, regions_two_dims, arrow.center, direction,
                                                switch=+1)

            if diags_one and diags_two:
                single_line = True
                diags_react, diags_prod = self.assign_diags(diags_one, diags_two, arrow)

                
            else: ### If no diags were found on one side of an arrow, assume, they can be
                # found in the previous or next row of the reaction
                # Check whether the arrow is at the beginning or end of a given line
                closer_extreme = min([0, self.fig.img.shape[1]], key=lambda coord: (arrow.panel.center[0] - coord)**2)
                search_direction = 'up-right' if closer_extreme == 0 else 'down-left'
                try:
                    diags_one = diags_one if diags_one else self._search_elsewhere(where=search_direction, arrow=arrow,
                                                                                direction=direction,
                                                                                switch=-1)
                    diags_two = diags_two if diags_two else self._search_elsewhere(where=search_direction, arrow=arrow,
                                                                                direction=direction,
                                                                                switch=+1)
                    if diags_one and diags_two:
                        diags_react, diags_prod = self.assign_diags(diags_one, diags_two, arrow, multiline=True)
                    
                except IndexError or ValueError:
                    return
                single_line = False
                
                if not (diags_one and diags_two):
                    self.is_incomplete = True
                    return                
            
        self.reaction_steps.append(ReactionStep(arrow, reactants=diags_react, products=diags_prod, single_line=single_line))

    def assign_diags(self, group1, group2, arrow, multiline=False):
        """Assigns the two groups as either reactants or products based on their proximity to the arrow centre
        vs. their proximity to arrow's centre of mass.

        :param group1: a group of diagrams
        :type group1: list[Diagram]
        :param group2: a group of diagrams
        :type group2: list[Diagram]
        :param arrow: arrow with respect to which the groups are analysed
        :type arrow: BaseArrow
        :return: the same groups classified as either reactants or products
        :rtype: tuple[list[Diagram]]"""
        ref_point = arrow.reference_pt # arrow's center of mass which denoted the products' side
        arrow_center = arrow.panel.center
        groups = [group1, group2]

        def compute_arrow_group_dist_change(group):
            """Compute two distances: one between the midpoint and the group of bounding boxes,
            then betwen the reference point (cente of mass of an arrow). Thes subtract the two values.
            The distance in these two scenarios will decrease for the product group, and increase for the reactant group"""
            dist_center = min(bbox.edge_separation(arrow_center) for bbox in group)
            dist_ref_point = min(bbox.edge_separation(ref_point) for bbox in group)
            return dist_ref_point - dist_center
        
        def compute_arrow_group_dist(group):
            """Compute two distances: one between the midpoint and the group of bounding boxes,
            then betwen the reference point (cente of mass of an arrow). Thes subtract the two values.
            The distance in these two scenarios will decrease for the product group, and increase for the reactant group"""
            return  min(bbox.edge_separation(arrow_center) for bbox in group)
        
        if not multiline:
            prod_group = min(groups, key=compute_arrow_group_dist_change)
            groups.remove(prod_group)
            react_group = groups[0]
            return react_group, prod_group
        else:
            closer = min(groups, key=compute_arrow_group_dist)
            ref_center_dist_diff = compute_arrow_group_dist_change(closer)
            prod_group = None
            react_group = None
            if ref_center_dist_diff < 0:
                prod_group = closer
            else:
                react_group = closer
            groups.remove(closer)
            if react_group is None:
                react_group = groups[0]
            else:
                prod_group = groups[0]
            return react_group, prod_group
        
    def _compute_scan_direction_curly_arrow(self, curly_arrow):
        temp_fig = Figure(np.zeros_like(self.fig.img), self.fig.img)
        temp_fig.img[curly_arrow.panel.pixels] = 255
        temp_fig = curly_arrow.panel.create_crop(temp_fig)

        selected_px = self._select_arrow_ends(temp_fig)
        
        top, left = curly_arrow.panel.top, curly_arrow.panel.left
        selected_px = [(px[0]+left, px[1]+top) for px in selected_px]

        pairs = []
        for px in selected_px:
            x, y = px
            temp_rect = Rect((y,x,y,x)) 
            closest_diag = min(self.diagrams, key=lambda diag: diag.panel.edge_separation(temp_rect))
            if closest_diag.panel.edge_separation(temp_rect) < curly_arrow.panel.width * 0.3:
                pairs.append((px, closest_diag))
                
        reactant_ends = []
        product_ends = []
        EXTENSION = max(25, int(curly_arrow.panel.area*3*10e-5))
        for pair in pairs:
            px, _ = pair
            top = px[1] - EXTENSION
            bottom = px[1] + EXTENSION
            left = px[0] - EXTENSION
            right = px[0] + EXTENSION
            crop = self.fig.img[top:bottom, left:right]
            cnt, _ = cv2.findContours(crop, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE )
            cnt = max(cnt, key= lambda contour: cv2.contourArea(contour))
            [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
            direction_normal = np.concatenate((vy, -1*vx))
            x_step = vx
            y_step = vy
            x_deltas = np.arange(1,EXTENSION//2)
            
            deltas = np.stack((x_deltas*x_step, x_deltas*y_step), axis=1)
            line_pt = np.stack([x,y], axis=1)
            line_pts = np.concatenate((line_pt - deltas,line_pt,  line_pt + deltas), axis=0)
            num_pixels = []
            for pt in line_pts:
                l = Line(find_points_on_line(pt, direction_normal, distance=10))
                pixels = np.asarray([[p.row, p.col] for p in l.pixels])
                try:
                    pixels = crop[pixels[:,0], pixels[:,1]]
                    num_pixels.append(np.sum(pixels!=0))
                except:
                    continue

            num_pixels = list(filter(lambda x: x>0, num_pixels))
            if not num_pixels or max(num_pixels) - min(num_pixels) < 5:
                reactant_ends.append(pair)
            else:
                product_ends.append(pair)
        
        try:
            main_reactant_pair = max(reactant_ends, key=lambda pair: pair[1].area)
            main_product_pair = max(product_ends, key=lambda pair: pair[1].area)
        except ValueError:
            raise
        selected_ends= [main_reactant_pair, main_product_pair]

        scan_params = []
        for pair in selected_ends:
            point, diag = pair
            x1, y1  = point
            x2, y2 = diag.panel.center
            if x2 - x1 == 0:
                direction = [0, (y2-y1)/np.abs(y2-y1)]
            else:
                direction = np.asarray([(x2-x1)/np.abs(x2-x1), (x2-x1)/np.abs(x2-x1) * (y2-y1)/(x2-x1)])
                direction = direction / np.linalg.norm(direction)
                # point = (point[1], point[0])
            scan_params.append((point, direction))
            
        return scan_params

    def _search_elsewhere(self, where, arrow, direction, switch):
        """
        Looks for structures in a different line of a multi-line reaction scheme.

        If a reaction scheme ends unexpectedly either on the left or right side of an arrows (no species found), then
        a search is performed in the previous or next line of a reaction scheme respectively (assumes multiple lines
        in a reaction scheme). Assumes left-to-right reaction scheme. Estimates the optimal alternative search point
        using arrow and diagrams' coordinates in a DBSCAN search.
        This gives clusters corresponding to the multiple lines in a reaction scheme. Performs a search in the new spot.
        :param where: Allows either 'down-left' to look below and to the left of arrow, or 'up-right' (above to the right)
        :type where: str
        :param arrow: Original arrow, around which the search failed
        :type arrow: BaseArrow
        :param direction: direction of an arrow specified as a unit (x, y) vector
        :type direction: tuple[float, float]
        :return: Collection of found species
        :rtype: list[Diagram]
        """
        assert where in ['down-left', 'up-right']
        fig = self.fig
        diags = self.diagrams

        X = np.array([s.center[1] for s in diags] + [arrow.panel.center[1]]).reshape(-1, 1)  # the y-coordinate
        eps = np.mean([s.height for s in diags])*0.75
        dbscan = DBSCAN(eps=eps, min_samples=2)
        y = dbscan.fit_predict(X)
        num_labels = max(y) - min(y) + 1  # include outliers (labels -1) if any
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
            move_to_horizontal = 0
        elif where == 'up-right':
            move_to_vertical = [centre for centre in centres if centre < arrow.panel.center[1]][-1]
            move_to_horizontal = fig.img.shape[1]
        startpoint = (move_to_horizontal, move_to_vertical)
        species = self._perform_scan(arrow, self.fig.img.shape, startpoint, direction, switch)

        return species
    
    def _check_proximity(self, point, panel, prox_dist=50, edge=True):
        """Checks proximity between a point and a panel"""
        x, y = point
        temp_rect = Rect((y,x,y,x))
        if edge:
            return panel.edge_separation(temp_rect) < prox_dist
        else: 
            return panel.center_separation(temp_rect) < prox_dist
    
    def _compute_arrow_scan_params(self, arrow):
        min_rect = cv2.minAreaRect(arrow.contour[0])
        box_points = cv2.boxPoints(min_rect)
        diffs = [box_points[idx+1] - box_points[idx] for idx in range(3)] + [box_points[0] - box_points[-1]]
        box_segment_lengths = [np.sqrt(np.sum(np.power(x,2))) for x in diffs]
        largest_idx = np.argmax(box_segment_lengths)
        points = box_points[largest_idx], box_points[(largest_idx+1)%4]
        x_diff = points[1][0] - points[0][0]
        y_diff = points[1][1] - points[0][1]
        dir_array = np.array([x_diff, y_diff])
        direction_arrow = dir_array / np.linalg.norm(dir_array)

        direction_normal = np.asarray([-1*direction_arrow[1], direction_arrow[0]])

        return direction_arrow, direction_normal

    def _perform_scan(self, arrow, region_dims, start_point, direction, switch):
        # assert switch in [-1, 1]
        region_x_length, region_y_length = region_dims
        epsilon = 1e-5  # Avoid division by 0
        stepsize_x = max(self.stepsize * direction[0], epsilon, key=lambda x: abs(x))
        stepsize_y = max(self.stepsize * direction[1], epsilon, key=lambda x: abs(x))
        num_centers_x = abs(region_x_length // stepsize_x)
        num_centers_y = abs(region_y_length // stepsize_y)
        num_centers = int(min(num_centers_x, num_centers_y))
        if num_centers == 0:  # Handles a case where no line scan can be performed because the arrow lies close to
                              # image boundary (and no diagrams are present on this boundary)
            return []
        deltas = np.array([[stepsize_x * n, stepsize_y * n] for n in range(1, num_centers + 1)])
        deltas = deltas * switch
        start_offset_required = max(arrow.width, arrow.height) / 2
        num_steps_required = int(start_offset_required / max(stepsize_x, stepsize_y))
        try:
            start_point = start_point + deltas[num_steps_required+1]
        except IndexError:
            pass
        points = start_point + deltas
        # lines = [Line(find_points_on_line(center, direction_normal, distance=self.segment_length))
        #              for center in centers]

        # Visualize lines ##
        # import matplotlib.pyplot as plt
        # plt.imshow(self.fig.img)
        # for p in points:
        #     plt.scatter(*p, c='r')
        # plt.show()

        try:
            other_arrows = copy.copy(self.arrows)
            other_arrows.remove(arrow)
            prox_dist = min(max(stepsize_x, stepsize_y) * 4, 50)
            arrow_overlap = [any(self._check_proximity(p, a.panel, max(max(a.panel.width, a.panel.height) * 0.2, prox_dist)) for a in other_arrows) for p in points].index(True)
        except ValueError:
            arrow_overlap = None

        if arrow_overlap is not None and arrow_overlap >= 2: 
            points = points[:arrow_overlap]
        diags = []

        probe_dist = np.mean([max(a.panel.width, a.panel.height) for a in other_arrows]) * 0.3
        probe_dist = max(50, probe_dist)
        for p in points:
            for d in self.diagrams:
                if self._check_proximity(p, d.panel, probe_dist):
                    diags.append(d)
        diags = list(set(diags))

        return diags

    def _select_arrow_ends(self, fig):
        img = fig.img
        h, w = img.shape
        h_crop, w_crop = max(int(0.05*h),5), max(int(0.05*w),5)
        crop_1 = img[:h_crop, :]
        crop_2 = img[-1*h_crop:, :]
        crop_3 = img[:, :w_crop]
        crop_4 = img[:, -1*w_crop:]
        crops = [crop_1, crop_2, crop_3, crop_4]
        # padded_crops = list(map(lambda x: np.pad(x, 5, constant_values=0), crops))
        selected_px = []
        crops_on_px = []
        for crop in crops:
            y, x = np.nonzero(crop)
            if not (x.shape[0]== 0 or y.shape[0] == 0):
            
                crops_on_px.append(list(zip(*(y,x))))
            else:
                crops_on_px.append(None)
        if crops_on_px[1] is not None:
            crops_on_px[1] = crops_on_px[1][::-1]
            
        if crops_on_px[3] is not None:
            crops_on_px[3] = crops_on_px[3][::-1]
        #Select on_pixels closest to each image boundary
        selected_px = [on_px_list[0] for on_px_list in crops_on_px]
        selected_px = [[px[1], px[0]] for px in selected_px]
        selected_px[0] = selected_px[0] if selected_px[0] else None
        selected_px[1] = (selected_px[1][0],  selected_px[1][1] +  (h-h_crop)) if selected_px[1] else None
        selected_px[2] = (selected_px[2][0], selected_px[2][1]) if selected_px[2] else None
        selected_px[3] = (selected_px[3][0]+ (w-w_crop), selected_px[3][1] ) if selected_px[3] else None
        
        selected_px = [p for p in selected_px if p is not None]
        pruned_px = []
        for px1 in selected_px:
            if all([euclidean_distance(px1, p) > max(w,h)*0.3 for p in pruned_px]):
                pruned_px.append(px1)
        return pruned_px
