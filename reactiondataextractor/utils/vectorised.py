import cv2
import numpy as np

import potrace

from configs import config
from utils.utils import erase_elements, euclidean_distance, isolate_patches, HoughLinesP
from reactiondataextractor.configs import Config, ExtractorConfig


class DiagramVectoriser:
    def __init__(self):
        # self.corners = []
        # self.adjacency_matrix = []
        self.diag = None
        
    def create_vectorised_diagram_graph(self):
        # import matplotlib.pyplot as plt
        assert self.diag, "diagram for vectorisation has not been set"

        img = erase_elements(self.diag.crop, [char.panel for char in self.diag.text_chars]+
                             [r_group._lone_line for r_group in self.diag.r_group_placeholders if r_group.lone], copy_fig=False).img
        
        # Add corners for each heteroatom connecting carbon backbones inside the diagram
        heteroatom_corners_to_add = [char.panel.center for char in self.diag.text_chars if char.text in ['O', 'N', 'S', 'P', 'F'] ]
        corners = self.vectorise_image(img, self.diag.panel.fig.single_bond_length * 0.6, artificial_corners=heteroatom_corners_to_add)
        
        distance_matrix = np.array([[euclidean_distance(p1, p2) for p1 in corners] for p2 in corners])        
        adjacency_matrix = self._create_adjacency_matrix(distance_matrix)
        # import matplotlib.pyplot as plt
        # for idx,row in enumerate(adjacency_matrix):
        #     plt.close()
        #     if np.sum(row) > 1:
        #         plt.imshow(img)
        #         plt.scatter(*corners[idx], c='r')
        #         connected = np.where(row)
        #         for connected_idx in connected[0]:
        #             if connected_idx != idx:
        #                 plt.scatter(*corners[connected_idx], c='b')
        self.diag.corners = corners
        self.diag.adjacency_matrix = adjacency_matrix

    def vectorise_image(self, img, corner_prune_dist, artificial_corners=[]):
        img = cv2.ximgproc.thinning(img)
        data = potrace.Bitmap(img / 255)
        trace = data.trace(alphamax=0.4)
        corners = [segment.c for curve in trace for segment in curve if segment.is_corner]
        if artificial_corners:
            corners = corners + artificial_corners
        corners = self.remove_duplicate_corners(corners, corner_prune_dist, fixed_corners=artificial_corners) 
        return corners
        # import matplotlib.pyplot as plt
        # plt.imshow(img)   
        # for corner in corners:
        #     if corner:
        #         plt.scatter(*corner)

    def remove_duplicate_corners(self, corners, thresh_dist, fixed_corners=[]):
        duplicate_found = True
        
        while duplicate_found:
            duplicate_found = False
            for idx1 in range(len(corners)):
                if corners[idx1] is not None:
                    for idx2 in range(idx1+1, len(corners)):
                        if corners[idx2] is not None and euclidean_distance(corners[idx1], corners[idx2]) < thresh_dist:
                            duplicate_found = True
                            if corners[idx1] in fixed_corners:
                                new_x = corners[idx1][0]
                                new_y = corners[idx1][1]
                                corners[idx2] = None
                            elif corners[idx2] in fixed_corners:
                                new_x = corners[idx2][0]
                                new_y = corners[idx2][1]
                                corners[idx1] = None
                                break
                            else:
                                new_x = (corners[idx1][0] + corners[idx2][0]) / 2
                                new_y = (corners[idx1][1] + corners[idx2][1]) / 2
                                corners[idx2] = None
                            corners[idx1] = (new_x, new_y)
                  
        return [c for c in corners if c is not None]

    def _create_adjacency_matrix(self, dst_matrix):
        dst_matrix = dst_matrix / self.diag.panel.fig.single_bond_length * Config.SINGLE_BOND_LENGTH
        LIIMITING_DISTANCE = 2.12
        adjacency_matrix = np.zeros_like(dst_matrix)
        adjacency_matrix[dst_matrix < LIIMITING_DISTANCE] = 1
        return adjacency_matrix
    
    
def estimate_single_bond(fig):
    """Estimates length of a single bond in an image
    Uses a skeletonise image to find the number of lines of differing lengths. The single bond length is chosen using
    a graph of number of detected lines vs. the length of a line. The optimal value is where the change in number of
    lines as the length varies is greatest.
    :param Figure fig: analysed figure
    :return: approximate length of a single bond
    :rtype: int"""
    ccs = fig.connected_components
    # Get a rough bond length (line length) value from the two largest structures
    ccs = sorted(ccs, key=lambda cc: cc.area, reverse=True)
    estimation_ccs = ccs[:2]
    approx_line_lengths = []
    # estimation_fig = isolate_patches(fig, ccs[:3])
    biggest_cc = ccs[0]
    length_scan_param = 0.05 * min(biggest_cc.width, biggest_cc.height)
    pixel_masks = []
    for cc in estimation_ccs:
        pixel_mask = np.zeros_like(fig.img)
        pixel_mask[cc.pixels] = 1
        pixel_mask = pixel_mask[cc.top:cc.bottom+1, cc.left:cc.right+1]
        pixel_masks.append(pixel_mask)
        lines = HoughLinesP(pixel_mask, rho=1, theta=np.pi/180, minLineLength=int(length_scan_param), threshold=15)
        lengths = []
        for l in lines:
            x1, y1, x2, y2 = l.squeeze()
            x = x2 - x1
            y = y2 - y1
            length = np.sqrt(x**2 + y ** 2)
            lengths.append(length)
        approx_line_lengths.append(np.percentile(lengths, 85))
    approx_line_length = np.mean(approx_line_lengths)
    fig.single_bond_length = approx_line_length
    
    vectoriser = DiagramVectoriser()
    nearest_atom_dists_all = []
    for idx, cc in enumerate(estimation_ccs):
        pixel_mask = pixel_masks[idx] * 255
        corners = vectoriser.vectorise_image(pixel_mask, fig.single_bond_length * 0.6)
        dst_matrix = np.array([[euclidean_distance(p1, p2) for p1 in corners] for p2 in corners])        
        try:
            nearest_atom_dists =np.sort(dst_matrix)[:, 1]
            nearest_atom_dists_all.extend(nearest_atom_dists)
        except IndexError: # The cc is likely not an arrow
            continue 
    nearest_atom_dists_all = [dst for dst in nearest_atom_dists_all if dst != np.inf]
    
    single_bond = np.mean(nearest_atom_dists_all)
    # config.ExtractorConfig.SOLID_ARROW_MIN_LENGTH = int(single_bond // 4)
    # config.ExtractorConfig.SOLID_ARROW_THRESHOLD = int(single_bond // 4)
    fig.single_bond_length = single_bond
