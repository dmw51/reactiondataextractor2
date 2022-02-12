"""

"""

import os
import datetime
import re
import math
from collections import OrderedDict
from . import utils
import numpy as np

from skimage.color import rgb2gray, rgba2rgb
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.utils as KU
import tensorflow.keras.models as KM
from . import config


class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in config.BACKBONE_STRIDES])


############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(train_bn=True, input_shape=(512, 512)):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        train_bn: Boolean. Train or freeze Batch Norm layers
        input_shape: Shape of the resized input image
    """
    # assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    input_image = KL.Input(shape=(input_shape[0], input_shape[1], 1))
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    # x = AttentionBlock(32, 'embedded_gaussian', block_number=2)(x)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    # x = AttentionBlock(64, 'embedded_gaussian', block_number=3)(x)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 512], stage=4, block='a', train_bn=train_bn)
    # block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(5):
        x = identity_block(x, 3, [256, 256, 512], stage=4, block=chr(98 + i), train_bn=train_bn)
    # x = AttentionBlock(128, 'embedded_gaussian', 4)(x)
    C4 = x
    return KM.Model(inputs=[input_image], outputs=[C1, C2, C3, C4], name='feature_extractor')


############################################################
#  Proposal Layer
############################################################

def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result

def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    window = tf.cast(window, dtype=tf.float32)
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


class ProposalLayer(KL.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.

    Inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, proposal_count, nms_threshold, config=config.Config(), training=True, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.training = training

    def get_config(self):
        config = super(ProposalLayer, self).get_config()
        config["config"] = self.config.to_dict()
        config["proposal_count"] = self.proposal_count
        config["nms_threshold"] = self.nms_threshold
        return config

    def call(self, inputs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # Anchors
        anchors = inputs[2]

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(input=anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="top_anchors").indices
        # ix = tf.where(scores > 0.9)
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
                                    self.config.IMAGES_PER_GPU,
                                    names=["pre_nms_anchors"])

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = utils.batch_slice([pre_nms_anchors, deltas],
                                  lambda x, y: apply_box_deltas_graph(x, y),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors"])

        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(boxes,
                                  lambda x: clip_boxes_graph(x, window),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors_clipped"])

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Non-max suppression
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(
                boxes, scores, self.proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            scores = tf.gather(scores, indices)
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(input=proposals)[0], 0)
            proposals = tf.pad(tensor=proposals, paddings=[(0, padding), (0, 0)])
            scores = tf.pad(tensor=scores, paddings=[(0, padding)])
            return proposals, scores
        proposals, scores = utils.batch_slice([boxes, scores], nms,
                                      self.config.IMAGES_PER_GPU)
        if not self.training:
            return proposals, scores  # in prediction mode, scores are used to filter poor bboxes
        # if not tf.executing_eagerly():
        #     # Infer the static output shape:
        #     out_shape = self.compute_output_shape(None)
        #     proposals.set_shape(out_shape)
        return proposals

############################################################
#  Detection Target Layer
############################################################

def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeat() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(input=boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(input=boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(input=boxes1)[0], tf.shape(input=boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes , config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:
    proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [MAX_GT_INSTANCES] int class IDs
    gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(input=proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(tensor=gt_class_ids, mask=non_zeros,
                                   name="trim_gt_class_ids")


    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    # crowd_ix = tf.compat.v1.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.compat.v1.where(gt_class_ids > 0)[:, 0]
    # crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Compute overlaps with crowd boxes [proposals, crowd_boxes]
    # crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    # crowd_iou_max = tf.reduce_max(input_tensor=crowd_overlaps, axis=1)
    # no_crowd_bool = (crowd_iou_max < 0.001)

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(input_tensor=overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box.
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.compat.v1.where(positive_roi_bool)[:, 0]

    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    # negative_indices = tf.compat.v1.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]
    negative_indices = tf.compat.v1.where(roi_iou_max < 0.5)[:, 0]

    # Subsample ROIs. Aim for 33% positive. If no positives found, add one proposal with highest IoU to positives,
    # and one with lowest IoU to negatives
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
    npos = tf.shape(positive_indices)[-1] == 0
    positive_indices = tf.cond(npos, lambda: tf.argmax(tf.reduce_max(overlaps, axis=1))[tf.newaxis], lambda: positive_indices)

    positive_count = tf.shape(input=positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
    negative_indices = tf.cond(npos, lambda: tf.argmin(tf.reduce_min(overlaps, axis=1))[tf.newaxis], lambda: negative_indices)
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # # If no positives found, get one positive and one negative for triplet sampling based on IoU
    # def get_simple_positive_negative():
    #     """This function ensures that there is at least one positive and negative example to ensure that triplet
    #     sampling can be performed.
    #
    #     This function is executed conditionally, only if no real positive/negative rois
    #     have been found.
    #     :param proposals: [batch, num_proposals, 4] Tensor of generated proposals
    #     :param overlaps:[num_proposals, num_gt_boxes] Tensor of overlaps between the gt boxes and proposals.
    #     """
    #     positive = tf.reduce_max(overlaps, axis=1)
    #     positive = proposals[tf.argmax(positive)][tf.newaxis, :]
    #
    #     negative = tf.reduce_min(overlaps, axis=1)
    #     negative = proposals[tf.argmin(negative)][tf.newaxis, :]
    #     return positive, negative


    # positive_rois, negative_rois = tf.cond(empty_rois, get_simple_positive_negative, lambda:  positive_rois, negative_rois)
    # positive_indices = tf.cond(empty_rois, lambda:  tf.argmax(tf.reduce_max(overlaps, axis=1)), lambda: positive_indices)
    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        pred=tf.greater(tf.shape(input=positive_overlaps)[1], 0),
        true_fn=lambda: tf.argmax(input=positive_overlaps, axis=1),
        false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # # Compute mask targets
    # boxes = positive_rois
    # if config.USE_MINI_MASK:
    #     # Transform ROI coordinates from normalized image space
    #     # to normalized mini-mask space.
    #     y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
    #     gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
    #     gt_h = gt_y2 - gt_y1
    #     gt_w = gt_x2 - gt_x1
    #     y1 = (y1 - gt_y1) / gt_h
    #     x1 = (x1 - gt_x1) / gt_w
    #     y2 = (y2 - gt_y1) / gt_h
    #     x2 = (x2 - gt_x1) / gt_w
    #     boxes = tf.concat([y1, x1, y2, x2], 1)
    # box_ids = tf.range(0, tf.shape(input=positive_rois)[0])


    # Append negative ROIs and pad bbox deltas that
    # are not used for negative ROIs with zeros.


    rois = tf.concat([positive_rois, negative_rois], axis=0)
    # empty_rois = tf.shape(rois)[0] == 0
    # rois = tf.cond(empty_rois, get_simple_positive_negative, lambda:  rois)
    # roi_class_ids = tf.cond(empty_rois, lambda: tf.convert_to_tensor([1, 0]), lambda:  roi_class_ids)
    N = tf.shape(input=negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(input=rois)[0], 0)
    rois = tf.pad(tensor=rois, paddings=[(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(tensor=roi_gt_boxes, paddings=[(0, N + P), (0, 0)])
    roi_class_ids = tf.pad(tensor=roi_class_ids, paddings=[(0, N + P)])
    deltas = tf.pad(tensor=deltas, paddings=[(0, N + P), (0, 0)])

    # Additionally, output positive rois too
    positive_rois = tf.pad(tensor=positive_rois, paddings=[(0, N + P), (0, 0)])

    return positive_rois, rois, roi_class_ids, deltas


class DetectionTargetLayer(KL.Layer):
    """Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def get_config(self):
        config = super(DetectionTargetLayer, self).get_config()
        config["config"] = self.config.to_dict()
        return config

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]

        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes],
            lambda w, x, y: detection_targets_graph(
                w, x, y, self.config),
            self.config.IMAGES_PER_GPU, names=names)
        return outputs


class AttentionBlock(tf.keras.models.Model):
    """Attention block implementation as described by Wang et al. in 'Non-local Neural Networks' (2018).
     Contains multiple functions for computing the correlation between features. This implementation uses subsampling
     to reduce computation"""


    def __init__(self, bottleneck_size, instantiation, block_number):
        """
        :param int bottleneck_size: Number of features in the embedding space
        :param str instantiation: type of transformation performed on the input data as described in the original paper
        :param int block_number: number of a block (used in the layer name)
        :param bool pooled: boolean to indicate whether the data should be pooled prior to computing the correlation
        matrix (used to reduce computational cost)
        """
        assert instantiation in ['gaussian', 'embedded_gaussian',
                                 'dot_product', 'concatenation', ]

        instantiations_dict = {'gaussian': self.gaussian_inst,
                               'embedded_gaussian': self.embedded_gaussian_inst,
                               'dot_product': self.dot_product,}
                               # 'concatenation': self.concatenation}

        self.bottleneck_size = bottleneck_size
        self.instantiation = instantiations_dict.get(instantiation)
        self.block_number = block_number
        # self.pooled = pooled
        super().__init__(name=f'attn_{block_number}')

    def build(self, input_shape):
        self.g = KL.Conv2D(self.bottleneck_size, (1,1))
        self.conv_1 = KL.Conv2D(input_shape[-1], (1, 1))

    def call(self, inputs):
        batch = tf.shape(inputs)[0]
        h = tf.shape(inputs)[1]
        w = tf.shape(inputs)[2]
        filters = inputs.get_shape()[3]
        # print(tf.shape(inputs))
        # # batch, h, w , channels = inputs.get_shape().as_list()
        # batch = tf.cast(batch, tf.float32)
        f, normalizing_const = self.instantiation(inputs)
        g = self.g(inputs)
        g = KL.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(g)
        # print(g.get_shape())
        g = tf.reshape(g, (batch, -1, self.bottleneck_size))
        # print(g.get_shape())
        x = tf.matmul(f, g)/normalizing_const
        # x = tf.einsum('ijk, ijk->ik',g, f)/normalizing_const
        x = tf.reshape(x, (batch, h, w, self.bottleneck_size))
        x = self.conv_1(x)
        x += inputs
        return x


    def embedded_gaussian_inst(self, inputs):
        """Embedded Gaussian instantiation with dot product similarity"""
        # n = tf.shape(inputs)[3]
        batch = tf.shape(inputs)[0]
        theta = KL.Conv2D(self.bottleneck_size, (1,1))(inputs)
        phi = KL.Conv2D(self.bottleneck_size, (1, 1))(inputs)
        phi = KL.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(phi)

        theta = KL.Lambda(lambda t: tf.reshape(t, (batch, -1, self.bottleneck_size)))(theta)
        phi = KL.Lambda(lambda t: tf.reshape(t, (batch, -1, self.bottleneck_size)))(phi)

        x = tf.einsum('ijk,ilk->ijl', theta, phi)
        return tf.keras.activations.softmax(x), 1   # Softmax normalizes the output hence constant is 1

    def gaussian_inst(self, inputs):
        """Simple Gaussian instantiation with dot product similarity"""
        # n = tf.shape(inputs)[3]
        batch = inputs.get_shape()[0]
        x = KL.Lambda(lambda t: tf.reshape(t, (batch, -1, self.bottleneck_size)))(inputs)
        x = tf.einsum('ijk,ilk->ijl', x, x)
        return tf.keras.activations.softmax(x), 1   # Softmax normalizes the output hence constant is 1

    def dot_product(self, inputs):
        """Dot product similarity instantiation"""
        # n = tf.shape(inputs)[3]
        batch = inputs.get_shape()[0]
        theta = KL.Conv2D(self.bottleneck_size, (1,1))(inputs)
        phi = KL.Conv2D(self.bottleneck_size, (1, 1))(inputs)
        phi = KL.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(phi)

        theta =KL.Lambda(lambda t: tf.reshape(t, (batch, -1, self.bottleneck_size)))(theta)
        phi = KL.Lambda(lambda t: tf.reshape(t, (batch, -1, self.bottleneck_size)))(phi)

        # print('shapes:')
        # print(inputs.get_shape())
        # print(phi.get_shape())
        # print(theta.get_shape())
        x = tf.einsum('ijk,ilk->ijl', theta, phi)
        # print(x.get_shape())
        return x, tf.cast(tf.shape(x)[0], tf.float32)

    # def concatenation(self, inputs):
    #     """Alternative form of similarity metric using concatenation of embeddings"""
    #     n = tf.shape(inputs)[3]
    #     theta = KL.Conv2D(self.bottleneck_size, (1,1))(inputs)
    #     phi = KL.Conv2D(self.bottleneck_size, (1, 1))(inputs)
    #     concat = KL.concatenate((theta, phi))
    #
    #     theta = tf.reshape(theta, (-1, n))
    #     phi = tf.reshape(phi, (-1, n))
    #     x = tf.einsum('ij,kj->ik', theta, phi)
    #     return x, tf.cast(tf.shape(x)[0], tf.float32)

class TripletSampling(KL.Layer):

    def __init__(self, config, margin=0.4, num_filters=128):
        self.margin = margin
        self.config = config
        self.num_filters = num_filters
        super().__init__()


    def call(self, inputs):
        gt_features, gt_class_ids, rois, target_class_ids, roi_features = inputs
        batch, *_ = tf.shape(gt_features)

        # Use 1x1 convolutions to reduce the feature dimensionality
        shared = KL.Conv2D(self.num_filters, (1, 1))
        gt_features = shared(gt_features)
        roi_features = shared(roi_features)


        # Filter out padding in gt_features
        gts = tf.where(gt_class_ids > 0)
        gt_features = tf.gather_nd(gt_features, gts)
        num_gts = tf.shape(gt_features)[0]

        # Filter out padding in rois and roi features
        rois_bool = tf.reduce_sum(rois, axis=-1) > 0
        rois_idx = tf.where(rois_bool)
        # rois = tf.gather_nd(rois, rois_idx)
        rois_features_filtered = tf.gather_nd(roi_features, rois_idx)


        # Positive examples are rois with class > 0
        positive_indices = tf.where(target_class_ids > 0)
        positive_roi_features = tf.gather_nd(roi_features, positive_indices)
        num_positives = tf.shape(positive_roi_features)[0]

        # Negative examples are non-zero rois with class of 0
        cls_zero_idx = target_class_ids == 0
        negative_roi_features = tf.boolean_mask(roi_features, tf.logical_and(rois_bool, cls_zero_idx))
        num_negatives = tf.shape(negative_roi_features)[0]

        # Choose the optimal positives and negatives by selecting the closest negative and furthest positive for each anchor
        # Reshape the tensors to expose the feature maps. Add new axes to broadcast
        # Triplets are chosen from the whole batch (i.e. can be from different images) - this should help generalization
        # TODO: but does it?
        gt_features_reshaped = tf.math.l2_normalize(tf.reshape(gt_features, (num_gts, -1))[:, tf.newaxis, :], axis=-1)
        positive_rois_features_reshaped = tf.math.l2_normalize(tf.reshape(positive_roi_features, (num_positives, -1))[tf.newaxis, :, :], axis=-1)
        negative_rois_features_reshaped = tf.math.l2_normalize(tf.reshape(negative_roi_features, (num_negatives, -1))[tf.newaxis, :, :], axis=-1)

        dist_pos = tf.reduce_sum(tf.square(gt_features_reshaped - positive_rois_features_reshaped), axis=-1)
        dist_neg = tf.reduce_sum(tf.square(gt_features_reshaped - negative_rois_features_reshaped), axis=-1)

        # # Select the closest negative and furthest positive for each gt_features
        # max_pos = tf.argmax(dist_pos, axis=1)
        # min_neg = tf.argmax(dist_neg, axis=1)
        #
        # # Based on found distances, choose correct positives and negatives and stack them along a newly-created axis
        # # Expected_shape is [num_gt_boxes, 3, pooled_size, pooled_size, num_features]
        # return tf.stack((gt_features, tf.gather(positive_roi_features, max_pos), tf.gather(negative_roi_features, min_neg)), axis=1)

        # Return the max_pos and min_neg distances (rather than triplets themselves)
        # to avoid calculating these again in the triplet loss function
        # max_pos = tf.reduce_max(dist_pos, axis=1)
        # min_neg = tf.reduce_min(dist_neg, axis=1)

        ap_dist_unstacked = tf.unstack(dist_pos, axis=0)
        an_dist_unstacked = tf.unstack(dist_neg, axis=0)
        # Choose all ap and an distances for each anchor. Choose a distance per each ap pair so that d(ap) - d(an) < 0.
        # To simplify we choose the minimum value
        # (TODO: This might slow down convergence, those negatives should lie within the margin)
        triplet_diffs = [tf.reduce_min(ap[:, tf.newaxis] - an[tf.newaxis, :], axis=1) for ap, an in
                 zip(ap_dist_unstacked, an_dist_unstacked)]
        triplet_diffs = tf.stack(triplet_diffs, axis=0)

        # if d(ap) - d(an) > 0, eliminate this distance - this triplet will not contribute to the loss
        # (negative is not semi-hard. Similarly, remove whenever d(ap) - d(an) + margin < 0
        # These triplets bring no new information
        triplet_diffs = tf.boolean_mask(triplet_diffs, tf.logical_and(triplet_diffs < 0, triplet_diffs > -1.0 * self.margin))
        # print(triplet_diffs)
        # If no such triplets were found, suppress the loss
        # triplet_loss = K.switch(tf.size(triplet_diffs) > 0,
        #                         self.add_loss(tf.reduce_max([tf.reduce_mean(triplet_diffs), 0.0])),
        #                         tf.constant(0.0))
        triplet_loss = self.config.LOSS_WEIGHTS['triplet_loss'] * (tf.reduce_mean(triplet_diffs) + self.margin)
        self.add_loss(tf.reduce_max(tf.boolean_mask((triplet_loss, 0.0), tf.math.is_finite((triplet_loss, 0.0)))))
        # return tf.reduce_max((total_dist, 0))
        #
        # return [dist_pos, dist_neg]




def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""
    return tf.math.log(x) / tf.math.log(2.0)

class PyramidROIAlign(KL.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)


    def get_config(self):
        config = super(PyramidROIAlign, self).get_config()
        config['pool_shape'] = self.pool_shape
        return config


    def call(self, inputs, training):
        # TODO: Unify the two functions to reduce code redundancy
        if training:
            return self.training_call(inputs)

        else:
            return self.inference_call(inputs)

    def training_call(self, inputs):
        # For triplet sampling, we need to find the latent representation of ground truth boxes
        gt_boxes = inputs[0]

        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[1]

        #Concatenate gt_boxes and predicted boxes and process them together, then form separate outputs
        batch, num_gt_boxes, _ = tf.shape(gt_boxes)
        _, num_boxes, _ = tf.shape(boxes)
        boxes = tf.concat((gt_boxes, boxes), axis=1)

        # Image meta
        # Holds details about the image. See compose_image_meta()
        image_meta = inputs[2]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        # RDE: Only a single feature map is used in this implementation
        feature_maps = inputs[3:-1]
        feature_maps = list(feature_maps)
        FPN_LEVEL = 2

        # All RoIs (background/foreground) are RoI-aligned for triplet sampling. From these, we need to separate
        # the foreground RoIs, which will then be passed on to the (main) multi-class classification head
        target_class_ids = inputs[-1]

        # RDE: The box-to-FPNLevel assignment is not relevant here
        # # Assign each ROI to a level in the pyramid based on the ROI area.
        # y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        # h = y2 - y1
        # w = x2 - x1
        # # Use shape of first image. Images in a batch must have the same size.
        # image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
        # # Equation 1 in the Feature Pyramid Networks paper. Account for
        # # the fact that our coordinates are normalized here.
        # # e.g. a 224x224 ROI (in pixels) maps to P4
        #
        # image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        # roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        # roi_level = tf.minimum(5, tf.maximum(
        #     2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        # roi_level = tf.squeeze(roi_level, 2)
        #
        # # # Loop through levels and apply ROI pooling to each. P2 to P5.
        # pooled = []
        # box_to_level = []
        # # for i, level in enumerate(range(2, 6)):
        # ix = tf.compat.v1.where(tf.equal(roi_level, FPN_LEVEL))
        # level_boxes = tf.gather_nd(boxes, ix)
        #
        # Box indices for crop_and_resize.
        # box_indices = tf.cast(ix[:, 0], tf.int32)
        #
        # # Keep track of which box is mapped to which level
        # box_to_level.append(ix)
        #
        # # Stop gradient propogation to ROI proposals
        # level_boxes = tf.stop_gradient(level_boxes)
        # box_indices = tf.stop_gradient(box_indices)
        #
        # # Crop and Resize
        # # From Mask R-CNN paper: "We sample four regular locations, so
        # # that we can evaluate either max or average pooling. In fact,
        # # interpolating only a single value at each bin center (without
        # # pooling) is nearly as effective."
        # #
        # # Here we use the simplified approach of a single value per bin,
        # # which is how it's done in tf.crop_and_resize()
        # # Result: [batch * num_boxes, pool_height, pool_width, channels]

        # pooled.append(tf.image.crop_and_resize(
        #     feature_maps[0], level_boxes, box_indices, self.pool_shape,
        #     method="bilinear"))
        boxes_to_resize = tf.reshape(boxes, (-1, 4))
        boxes_to_resize = tf.stop_gradient(boxes_to_resize)

        # box_indices = tf.concat([[i]*num_gt_boxes.numpy() for i in range(batch)]+
        #                         [[j]*num_boxes.numpy() for j in range(batch)], axis=0)
        box_indices = tf.concat([[i]*(num_gt_boxes.numpy()+num_boxes.numpy()) for i in range(batch)], axis=0)
        box_indices = tf.stop_gradient(box_indices)
        pooled = tf.image.crop_and_resize(
            feature_maps[0], boxes_to_resize,
            box_indices=box_indices,
            crop_size=self.pool_shape,
            method="bilinear")

        # # Pack pooled features into one tensor
        # pooled = tf.concat(pooled, axis=0)
        #
        # # Pack box_to_level mapping into one array and add another
        # # column representing the order of pooled boxes
        # box_to_level = tf.concat(box_to_level, axis=0)
        # box_range = tf.expand_dims(tf.range(tf.shape(input=box_to_level)[0]), 1)
        # box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
        #                          axis=1)
        #
        # # Rearrange pooled features to match the order of the original boxes
        # # Sort box_to_level by batch then box index
        # # TF doesn't have a way to sort by two columns, so merge them and sort.
        # sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        # ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
        #     input=box_to_level)[0]).indices[::-1]
        # ix = tf.gather(box_to_level[:, 2], ix)
        # pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(input=boxes)[:2], tf.shape(input=pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        pooled_gt_features, pooled_features = tf.split(pooled, [num_gt_boxes, num_boxes], axis=1)

        # unstacked_pooled_features = tf.unstack(pooled_features, axis=0)
        # unstacked_target_class_ids = tf.unstack(target_class_ids, axis=0)
        # gathered = [tf.gather_nd(features, tf.where(class_ids > 0))
        #             for features, class_ids in zip(unstacked_pooled_features, unstacked_target_class_ids)]
        # gathered = [tf.pad(g, ((0, num_boxes-tf.shape(g)[0]), (0,0), (0, 0), (0, 0))) for g in gathered]
        # pooled_positive_features = tf.stack(gathered, axis=0)

        return pooled_gt_features, pooled_features#, pooled_positive_features

    def inference_call(self, inputs):
        """Simplified version of `call` used for inference. This does not rely on any ground truth inputs and does
        not output boxes in a format suitable for triplet sampling (since this is not used at inference time)"""

        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]
        batch, num_boxes, _ = tf.shape(boxes)


        # Image meta
        # Holds details about the image. See compose_image_meta()
        image_meta = inputs[1]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        # RDE: Only a single feature map is used in this implementation
        feature_maps = inputs[2:]
        feature_maps = list(feature_maps)
        FPN_LEVEL = 2


        # RDE: The box-to-FPNLevel assignment is not relevant here
        # # Assign each ROI to a level in the pyramid based on the ROI area.
        # y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        # h = y2 - y1
        # w = x2 - x1
        # # Use shape of first image. Images in a batch must have the same size.
        # image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
        # # Equation 1 in the Feature Pyramid Networks paper. Account for
        # # the fact that our coordinates are normalized here.
        # # e.g. a 224x224 ROI (in pixels) maps to P4
        #
        # image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        # roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        # roi_level = tf.minimum(5, tf.maximum(
        #     2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        # roi_level = tf.squeeze(roi_level, 2)
        #
        # # # Loop through levels and apply ROI pooling to each. P2 to P5.
        # pooled = []
        # box_to_level = []
        # # for i, level in enumerate(range(2, 6)):
        # ix = tf.compat.v1.where(tf.equal(roi_level, FPN_LEVEL))
        # level_boxes = tf.gather_nd(boxes, ix)
        #
        # Box indices for crop_and_resize.
        # box_indices = tf.cast(ix[:, 0], tf.int32)
        #
        # # Keep track of which box is mapped to which level
        # box_to_level.append(ix)
        #
        # # Stop gradient propogation to ROI proposals
        # level_boxes = tf.stop_gradient(level_boxes)
        # box_indices = tf.stop_gradient(box_indices)
        #
        # # Crop and Resize
        # # From Mask R-CNN paper: "We sample four regular locations, so
        # # that we can evaluate either max or average pooling. In fact,
        # # interpolating only a single value at each bin center (without
        # # pooling) is nearly as effective."
        # #
        # # Here we use the simplified approach of a single value per bin,
        # # which is how it's done in tf.crop_and_resize()
        # # Result: [batch * num_boxes, pool_height, pool_width, channels]

        # pooled.append(tf.image.crop_and_resize(
        #     feature_maps[0], level_boxes, box_indices, self.pool_shape,
        #     method="bilinear"))
        boxes_to_resize = tf.reshape(boxes, (-1, 4))
        boxes_to_resize = tf.stop_gradient(boxes_to_resize)

        # box_indices = tf.concat([[i]*num_gt_boxes.numpy() for i in range(batch)]+
        #                         [[j]*num_boxes.numpy() for j in range(batch)], axis=0)
        box_indices = tf.concat([[i]*(num_boxes.numpy()) for i in range(batch)], axis=0)
        box_indices = tf.stop_gradient(box_indices)
        pooled = tf.image.crop_and_resize(
            feature_maps[0], boxes_to_resize,
            box_indices=box_indices,
            crop_size=self.pool_shape,
            method="bilinear")

        # # Pack pooled features into one tensor
        # pooled = tf.concat(pooled, axis=0)
        #
        # # Pack box_to_level mapping into one array and add another
        # # column representing the order of pooled boxes
        # box_to_level = tf.concat(box_to_level, axis=0)
        # box_range = tf.expand_dims(tf.range(tf.shape(input=box_to_level)[0]), 1)
        # box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
        #                          axis=1)
        #
        # # Rearrange pooled features to match the order of the original boxes
        # # Sort box_to_level by batch then box index
        # # TF doesn't have a way to sort by two columns, so merge them and sort.
        # sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        # ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
        #     input=box_to_level)[0]).indices[::-1]
        # ix = tf.gather(box_to_level[:, 2], ix)
        # pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(input=boxes)[:2], tf.shape(input=pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)


        # unstacked_pooled_features = tf.unstack(pooled_features, axis=0)
        # unstacked_target_class_ids = tf.unstack(target_class_ids, axis=0)
        # gathered = [tf.gather_nd(features, tf.where(class_ids > 0))
        #             for features, class_ids in zip(unstacked_pooled_features, unstacked_target_class_ids)]
        # gathered = [tf.pad(g, ((0, num_boxes-tf.shape(g)[0]), (0,0), (0, 0), (0, 0))) for g in gathered]
        # pooled_positive_features = tf.stack(gathered, axis=0)

        return pooled


# def build_rpn_model(anchor_stride, anchors_per_location, depth):
#     """Builds a Keras model of the Region Proposal Network.
#     It wraps the RPN graph so it can be used multiple times with shared
#     weights.
#
#     anchors_per_location: number of anchors per pixel in the feature map
#     anchor_stride: Controls the density of anchors. Typically 1 (anchors for
#                    every pixel in the feature map), or 2 (every other pixel).
#     depth: Depth of the backbone feature map.
#
#     Returns a Keras Model object. The model outputs, when called, are:
#     rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
#     rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
#     rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
#                 applied to anchors.
#     """
#     input_feature_map = KL.Input(shape=[None, None, depth],
#                                  name="input_rpn_feature_map")
#     outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
#     return KM.Model([input_feature_map], outputs, name="rpn_model")


def rpn_graph(feature_map_shape, anchors_per_location, anchor_stride):
    """Builds the computation graph of Region Proposal Network.

    feature_map: backbone features [batch, height, width, depth]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """
    # TODO: check if stride of 2 causes alignment issues if the feature map
    # is not even.
    # Shared convolutional base of the RPN
    feature_map = KL.Input(feature_map_shape+(512,))
    shared = KL.Conv2D(1024, (3, 3), padding='same', activation='relu',
                       strides=anchor_stride,
                       name='rpn_conv_shared')(feature_map)

    # Anchor Score. [batch, height, width, anchors per location * 2].
    x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(input=t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation(
        "softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(input=t)[0], -1, 4]))(x)
    rpn_model = KM.Model(inputs=feature_map, outputs=[rpn_class_logits, rpn_probs, rpn_bbox], name='region_proposal')
    return rpn_model

def classifier_head(num_classes, train_bn=True,
                         fc_layers_size=1024, ):
    """Builds the computation model of the network classifier

    rois_features: [batch, num_rois, pool_size, pool_size, num_channels] Proposal boxes in normalized
          coordinates.

    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    fc_layers_size: Size of the 2 FC layers

    Returns:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                     proposal boxes
    """

    inputs = KL.Input((None, 7, 7, 512))
    # pool_size = tf.shape(inputs)[-2]


    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (7, 7), padding="valid"),
                           name="mrcnn_class_conv1")(inputs)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)),
                           name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                       name="pool_squeeze")(x)

    # Classifier head
    class_logits = KL.Dense(num_classes, name='mrcnn_class_logits')(shared)
    probs = KL.Activation("softmax", name="mrcnn_class")(class_logits)

    # BBox head
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'),
                           name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
    s = K.int_shape(x)
    if s[1] is None:
        mrcnn_bbox = KL.Reshape((-1, num_classes, 4), name="cnn_bbox")(x)
    else:
        mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="rcnn_bbox")(x)

    return KM.Model(inputs=inputs, outputs=[class_logits, probs, mrcnn_bbox], name='classifier_head')



class RDEModel(KM.Model):

    def __init__(self, training, config):
        super().__init__()
        self.training = training
        proposal_count = config.POST_NMS_ROIS_TRAINING if self.training else config.POST_NMS_ROIS_INFERENCE
        self.proposal_count = proposal_count

        self.config = config
        self.resnet = resnet_graph(train_bn=True, input_shape=self.config.IMAGE_SHAPE[:2])
        feature_map_shape = self.resnet.output_shape[-1][1:3]
        anchors_per_location = len(config.RPN_ANCHOR_SCALES) * len(config.RPN_ANCHOR_RATIOS)
        self.rpn_model = rpn_graph(feature_map_shape, anchors_per_location=anchors_per_location, anchor_stride=1)

        self.anchors = generate_anchors(self.config,
                                        self.config.RPN_ANCHOR_SCALES, self.config.RPN_ANCHOR_RATIOS, self.config.FEATURE_MAP_SHAPE,
                                        self.config.BACKBONE_STRIDE, self.config.RPN_ANCHOR_STRIDE, self.config.IMAGES_PER_GPU)
        self.anchors = utils.norm_boxes(self.anchors, config.IMAGE_SHAPE[:2])
        self.proposal_layer = ProposalLayer(self.proposal_count, self.config.RPN_NMS_THRESHOLD, config=config, training=self.training)
        self.detection_target_layer = DetectionTargetLayer(config=self.config)
        self.roi_align = PyramidROIAlign(pool_shape=(7, 7))
        self.triplet_sampling = TripletSampling(config=self.config, num_filters=self.config.TRIPLETS_NUM_FILTERS)
        self.classifier_head = classifier_head(self.config.NUM_CLASSES, self.config.TRAIN_BN, self.config.FPN_CLASSIF_FC_LAYERS_SIZE)

    def call(self, inputs):
        if self.training:
            input_image, gt_class_ids, gt_boxes, image_meta = inputs
            gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[1:3]))(gt_boxes)
            *_, x = self.resnet(input_image)
            rpn_class_logits, rpn_probs, rpn_deltas = self.rpn_model(x)

            rpn_rois = self.proposal_layer([rpn_probs, rpn_deltas, self.anchors])
            positive_rois_out, rois, target_class_ids, target_bbox = self.detection_target_layer([rpn_rois, gt_class_ids, gt_boxes])

            gt_features, roi_features = self.roi_align((gt_boxes, rois, image_meta, x, target_class_ids), training=self.training)


            self.triplet_sampling([gt_features, gt_class_ids, rois,
                                                   target_class_ids, roi_features])


            cnn_class_logits, cnn_probs, pred_box = self.classifier_head(roi_features)

            cnn_class_loss = cnn_class_loss_graph(target_class_ids, cnn_class_logits)
            cnn_class_loss = self.config.LOSS_WEIGHTS['cnn_class_loss'] * cnn_class_loss
            self.add_loss(cnn_class_loss)

            cnn_bbox_loss = cnn_bbox_loss_graph(target_bbox, target_class_ids, pred_box)
            self.add_loss(cnn_bbox_loss)

            # return rpn_class_logits, rpn_deltas, x, rois, tf.argmax(cnn_probs, axis=-1), cnn_class_logits, target_class_ids
            return rpn_class_logits, rpn_probs, rpn_deltas, positive_rois_out, tf.argmax(cnn_probs,
                                                                    axis=-1), target_class_ids
        else:
            input_image, image_meta = inputs
            assert input_image.shape[0] == 1, "Currently only batches of 1 are accepted in prediction mode"
            *_, x = self.resnet(input_image)
            rpn_class_logits, rpn_probs, rpn_deltas = self.rpn_model(x)
            rpn_class = tf.nn.softmax(rpn_class_logits)
            rpn_rois, rpn_scores = self.proposal_layer([rpn_probs, rpn_deltas, self.anchors])

            #Build the unused layers for completeness
            self.detection_target_layer.build(input_image.shape[1:])
            self.triplet_sampling.build(input_image.shape[1:])

            roi_features = self.roi_align((rpn_rois, image_meta, x), training=False)

            cnn_class_logits, cnn_probs, pred_bbox = self.classifier_head(roi_features)
            # Filter out background boxes
            fg_score_idx = tf.where(rpn_scores > self.config.DETECTION_MIN_CONFIDENCE)
            rpn_rois = tf.gather_nd(rpn_rois, fg_score_idx)
            cnn_probs = tf.gather_nd(cnn_probs, fg_score_idx)
            pred_bbox = tf.gather_nd(pred_bbox, fg_score_idx)
            window = norm_boxes_graph(image_meta[0, 7:11], self.config.IMAGE_SHAPE[:2])

            refined_rois, class_ids, scores = refine_detections_graph(rpn_rois, cnn_probs, pred_bbox, window, self.config)
            # cnn_probs = cnn_probs[0, ...]
            # pred_bbox = pred_bbox[0, ...]
            # class_ids = tf.argmax(input=cnn_probs, axis=1, output_type=tf.int32)
            # # Class probability of the top class of each ROI
            # indices = tf.stack([tf.range(cnn_probs.shape[0]), class_ids], axis=1)
            # class_scores = tf.gather_nd(cnn_probs, indices)
            # # Class-specific bounding box deltas
            # deltas_specific = tf.gather_nd(pred_bbox, indices)
            # # Apply bounding box deltas
            # # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
            # refined_rois = apply_box_deltas_graph(
            #     rpn_rois[0, ...], deltas_specific * self.config.BBOX_STD_DEV)
            # # Clip boxes to image window
            #
            # #add dummy axis to conform to the required shape
            # refined_rois = refined_rois[tf.newaxis, ...]
            # cnn_probs = cnn_probs[tf.newaxis, ...]

            return refined_rois[tf.newaxis,...], rpn_scores, class_ids[tf.newaxis,...], scores[tf.newaxis,...]

    def predict(self, x):
        """Predicts object bounding boxes and classes in input images `x`. 'x' should be a single image or a list of images"""
        assert self.training == False, 'You must initialise the model with `training=False` to make predictions'
        if not isinstance(x, list):
            x = [x]
        # images = tf.unstack(x, axis=0)
        images, metas = zip(*[preprocess_image(
            image, config=self.config, image_id=-1, active_class_ids=range(self.config.NUM_CLASSES)) for image in x])

        boxes, rpn_scores, classes, scores = self.call((tf.stack(images, axis=0), tf.stack(metas, axis=0)))
        windows = [m[7:11] for m in metas]
        boxes = [postprocess_detected(b, m ) for b, m in zip(boxes, metas)]
        # boxes = [denorm_boxes_graph(b, shape=image.shape[:2]) for b, image in zip(tf.unstack(boxes, axis=0), x)]
        scores = tf.squeeze(scores, axis=-1)
        classes = tf.squeeze(classes, axis=-1)
        classes, scores = [tf.unstack(x, axis=0) for x in (classes, scores)]
        classes = tf.cast(classes, tf.uint8)
        # conf_masks = [s> self.config.DETECTION_MIN_CONFIDENCE for s in scores]
        class_filter = [s > self.config.DETECTION_MIN_CONFIDENCE for s in scores]
        # foreground_filter = [s > self.config.DETECTION_MIN_CONFIDENCE for s in rpn_scores]
        out_lst = [[], [], []]
        for i in range(len(x)):
          for idx, output in enumerate([boxes, classes, scores]):
              # out_lst[idx].append(tf.gather_nd(output[i], filtered[i]))
                out_lst[idx].append(tf.boolean_mask(output[i], class_filter[i]))

        return out_lst

def refine_detections_graph(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in normalized coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where
        coordinates are normalized.
    """
    # Class IDs per ROI
    class_ids = tf.argmax(input=probs, axis=1, output_type=tf.int32)
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = apply_box_deltas_graph(
        rois, deltas_specific * config.BBOX_STD_DEV)
    # Clip boxes to image window
    refined_rois = clip_boxes_graph(refined_rois, window)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = tf.compat.v1.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    # if config.DETECTION_MIN_CONFIDENCE:
    #     conf_keep = tf.compat.v1.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
    #     keep = tf.sets.intersection(tf.expand_dims(keep, 0),
    #                                     tf.expand_dims(conf_keep, 0))
    #     keep = tf.sparse.to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois,   keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.compat.v1.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
                tf.gather(pre_nms_rois, ixs),
                tf.gather(pre_nms_scores, ixs),
                max_output_size=config.DETECTION_MAX_INSTANCES,
                iou_threshold=config.DETECTION_NMS_THRESHOLD)
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(input=class_keep)[0]
        class_keep = tf.pad(tensor=class_keep, paddings=[(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.compat.v1.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse.to_dense(keep)[0]
    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(input=class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.dtypes.cast(tf.gather(class_ids, keep), tf.float32)[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)

    refined_rois = tf.gather(refined_rois, keep)
    class_ids = tf.dtypes.cast(tf.gather(class_ids, keep), tf.float32)[..., tf.newaxis]
    class_scores = tf.gather(class_scores, keep)[..., tf.newaxis]
    return refined_rois, class_ids, class_scores


def postprocess_detected(boxes, meta):
    """
    boxes: np.array[num_boxes, 4] array of detected bounding boxes (in normalized coordinates of the
    preprocessed - padded/scaled image)
    meta: np.array[20] array containing meta information about the image
    returns: boxes - np.array[num_boxes, 4] array of bounding boxes in the coordinate system of the initial image.

    The input image is rescaled and padded with zeros. This function takes in the boxes and changes their coordinates
    to represent their true positions in the original image.
    """
    window = meta[7:11]
    original_image_shape = meta[1:4]
    image_shape = meta[4:7]
    # Translate normalized coordinates in the resized image to pixel
    # coordinates in the original image before resizing
    window = utils.norm_boxes(window, image_shape[:2])
    wy1, wx1, wy2, wx2 = window
    shift = np.array([wy1, wx1, wy1, wx1])
    wh = wy2 - wy1  # window height
    ww = wx2 - wx1  # window width
    scale = np.array([wh, ww, wh, ww])
    # Convert boxes to normalized coordinates on the window
    boxes = np.divide(boxes - shift, scale)
    # Convert boxes to pixel coordinates on the original image
    boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

    return boxes
############################################################
#  Miscellenous Graph Functions
############################################################

def generate_anchors(config, scales, ratios, shape, feature_stride, anchor_stride, batch_size):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    batch_size: Number of images per GPU
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)


    boxes = np.broadcast_to(boxes, (batch_size,) + boxes.shape)

    return boxes


def trim_zeros_graph(boxes, name='trim_zeros'):
    """Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(input_tensor=tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(tensor=boxes, mask=non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    # x = tf.cast(x, tf.float32)
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])

    return tf.concat(outputs, axis=0)


def norm_boxes_graph(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
    """
    boxes = tf.cast(boxes, dtype=tf.float32)
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.math.maximum(tf.divide(boxes - shift, scale), 0)


def denorm_boxes_graph(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [..., (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)


#### Losses
def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for BG/FG.
    """
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.compat.v1.where(K.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Cross entropy loss
    loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    loss = K.switch(tf.size(input=loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss

def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    config: the model config object.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.compat.v1.where(K.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts,
                                   config.IMAGES_PER_GPU)

    loss = smooth_l1_loss(target_bbox, rpn_bbox)

    loss = K.switch(tf.size(input=loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss

def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    y_true = tf.cast(y_true, tf.float32)
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def triplet_loss(distances, margin=0.4):
    """Computes triplet loss given Euclidean space distances between anchors and positive examples (ap), and anchors and
    negative examples (an). Uses all anchor-positive pairs and assigns a negative to each pair based of anchor-negative
    distance. Only negatives such that d(ap) < d(an) are chosen to contribute to the loss (semi-hard negatives)

    Input:
    :param distances: a list of anchor-positive, and anchor-negative distances for each possible pair
    :param margin: integer specifying the required triplet loss margin
    """
    #After unpacking, unstack the distances so that per-gt-box computation is possible
    ap_dist, an_dist = distances
    ap_dist_unstacked = tf.unstack(ap_dist, axis=0)
    an_dist_unstacked = tf.unstack(an_dist, axis=0)
    # Choose all ap and an distances for each anchor. Choose a distance per each ap pair so that d(ap) - d(an) < 0.
    # To simplify we choose the minimum value
    # (TODO: This might slow down convergence, those negatives should lie within the margin)
    dists = [tf.reduce_min(ap[:, tf.newaxis] - an[tf.newaxis, :], axis=1) for ap, an in zip(ap_dist_unstacked, an_dist_unstacked)]
    dists = tf.stack(dists, axis=0)

    # if d(ap) - d(an) > 0, eliminate this distance - this triplet will not contribute to the loss
    # (negative is not semi-hard. Similarly, remove whenever d(ap) - d(an) + margin < 0
    # These triplets bring no new information
    dists = tf.boolean_mask(dists, tf.logical_and(dists < 0, dists > -1.0 * margin))
    total_dist = tf.reduce_mean(dists) + margin
    return tf.reduce_max((total_dist, 0))

def cnn_class_loss_graph(target_class_ids, pred_class_logits):
    """Loss for the classifier head of the network.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    # active_class_ids: [batch, num_classes]. Has a value of 1 for
    #     classes that are in the dataset of the image, and 0
    #     for classes that are not in the dataset.
    """
    # During model building, Keras calls this function with
    # target_class_ids of type float32. Unclear why. Cast it
    # to int to get around it.
    target_class_ids = tf.cast(target_class_ids, 'int64')

    # Find predictions of classes that are not in the dataset.
    # pred_class_ids = tf.argmax(input=pred_class_logits, axis=2)
    # pred_active = tf.gather(active_class_ids[0], pred_class_ids)

    # RDE: Remove class 0 - this classification head focuses solely on differentiating foreground objects
    non_zero_class_ids = tf.where(target_class_ids > 0)
    target_class_ids = tf.gather_nd(target_class_ids, non_zero_class_ids)
    pred_class_logits = tf.gather_nd(pred_class_logits, non_zero_class_ids)



    # Loss
    loss = K.sparse_categorical_crossentropy(target=target_class_ids,
                                             output=pred_class_logits,
                                             from_logits=True)

    # Only take into account non-background classes?
    # loss = tf.gather_nd(loss, tf.where(target_class_ids > 0))
    loss = K.switch(tf.size(input=loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss

def cnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Loss for final bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 4))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    positive_roi_ix = tf.compat.v1.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = K.switch(tf.size(input=target_bbox) > 0,
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


# @tf.function
def train_one_epoch(model, train_dataset, optimizer):
    losses = []
    i = 0
    ids = []
    partial_losses = [[], [], [], [], []]
    train_generator = DataGenerator(train_dataset, model.config, shuffle=True )
    for batch in train_generator:
        with tf.GradientTape() as tape:
            inputs, _ = batch
            inputs = [tf.convert_to_tensor(i) for i in inputs]
            batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox, batch_gt_class_ids, batch_gt_boxes = inputs
            ids.append(batch_image_meta[:,0])

            rpn_class_logits, rpn_probs, rpn_deltas, positive_rois_out, pred_classes, target_class_ids = model((batch_images, batch_gt_class_ids,
                                                                                  batch_gt_boxes, batch_image_meta ))
            loss_weights = model.config.LOSS_WEIGHTS
            loss = loss_weights['rpn_class_loss'] * rpn_class_loss_graph(batch_rpn_match, rpn_class_logits)
            rpn_bbox_loss = loss_weights['rpn_bbox_loss'] * rpn_bbox_loss_graph(model.config, batch_rpn_bbox, batch_rpn_match, rpn_deltas)

            partial_losses[0].append(loss)
            partial_losses[1].append(rpn_bbox_loss)
            loss = loss + rpn_bbox_loss

            partial_losses[2].append(model.losses[0]) # triplet loss
            partial_losses[3].append(model.losses[1]) # cnn_class loss
            partial_losses[4].append(model.losses[2]) # cnn bbox loss
            loss = loss + tf.reduce_sum(model.losses)
            losses.append(loss)

            # print(f'Training loss at step {i}: {loss}')

            grad = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grad, model.trainable_weights))
        i += 1
    print(f'Training loss at end of epoch: {np.mean(losses)}, rpn class loss: {tf.reduce_mean(partial_losses[0])}, rpn bbox loss: {tf.reduce_mean(partial_losses[1])},'
          f'triplet loss: {tf.reduce_mean(partial_losses[2])},  cnn class loss: {tf.reduce_mean(partial_losses[3])}, cnn bbox loss: {tf.reduce_mean(partial_losses[4])}')
    print(f'Losses: {[l.numpy() for l in losses]}')
    print(f'Ids: {[i.numpy().tolist() for i in ids]}')

def validate_model(model, val_dataset):
    losses = []
    i = 0
    val_dataset = DataGenerator(val_dataset, model.config)
    for batch in val_dataset:
        with tf.GradientTape() as tape:
            inputs, _ = batch
            inputs = [tf.convert_to_tensor(i) for i in inputs]
            batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox, batch_gt_class_ids, batch_gt_boxes = inputs

            rpn_class_logits, rpn_probs, rpn_deltas, positive_rois_out, pred_classes, target_class_ids = model((batch_images, batch_gt_class_ids,
                                                                                  batch_gt_boxes, batch_image_meta ))
            loss = rpn_class_loss_graph(batch_rpn_match, rpn_class_logits)
            loss = loss + (rpn_bbox_loss_graph(model.config, batch_rpn_bbox, batch_rpn_match, rpn_deltas))

            loss = loss + tf.reduce_sum(model.losses)
            losses.append(loss)
            # print(f'Validation loss at step {i}: {loss}')
            i += 1
    print(f'Validation loss at end of epoch: {np.mean(losses)}')

class DataGenerator(KU.Sequence):
    """An iterable that returns images and corresponding target class ids,
        bounding box deltas, and masks. It inherits from keras.utils.Sequence to avoid data redundancy
        when multiprocessing=True.

        dataset: The Dataset object to pick data from
        config: The model config object
        shuffle: If True, shuffles the samples before every epoch
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
            For example, passing imgaug.augmenters.Fliplr(0.5) flips images
            right/left 50% of the time.
        random_rois: If > 0 then generate proposals to be used to train the
                     network classifier and mask heads. Useful if training
                     the Mask RCNN part without the RPN.
        detection_targets: If True, generate detection targets (class IDs, bbox
            deltas, and masks). Typically for debugging or visualizations because
            in trainig detection targets are generated by DetectionTargetLayer.

        Returns a Python iterable. Upon calling __getitem__() on it, the
        iterable returns two lists, inputs and outputs. The contents
        of the lists differ depending on the received arguments:
        inputs list:
        - images: [batch, H, W, C]
        - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
        - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
        - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
        - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
        - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                    are those of the image unless use_mini_mask is True, in which
                    case they are defined in MINI_MASK_SHAPE.

        outputs list: Usually empty in regular training. But if detection_targets
            is True then the outputs list contains target class_ids, bbox deltas,
            and masks.
        """

    def __init__(self, dataset, config, shuffle=True, augmentation=None,
                 random_rois=0, detection_targets=False):

        self.image_ids = np.copy(dataset.image_ids)
        self.dataset = dataset
        self.config = config

        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]
        self.backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
        # self.anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
        #                                               config.RPN_ANCHOR_RATIOS,
        #                                               self.backbone_shapes,
        #                                               config.BACKBONE_STRIDES,
        #                                               config.RPN_ANCHOR_STRIDE)
        #Generate_anchors returns anchors of shape [batch_size, num_anchors, 4], but generator expects [num_anchors,4]
        #As a temporary work-around, set batch_size = 1 and squeeze out the 0th dimension
        self.anchors = generate_anchors(self.config,
                self.config.RPN_ANCHOR_SCALES, self.config.RPN_ANCHOR_RATIOS, self.config.FEATURE_MAP_SHAPE,
                self.config.BACKBONE_STRIDE, self.config.RPN_ANCHOR_STRIDE, batch_size=1)
        self.anchors = np.squeeze(self.anchors, axis=0)

        self.shuffle = shuffle
        self.augmentation = augmentation
        self.random_rois = random_rois
        self.batch_size = self.config.BATCH_SIZE
        self.detection_targets = detection_targets

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / float(self.batch_size)))

    def __iter__(self):
        self._index = -1
        return self

    def __next__(self):

        #Shuffle at the beginning of an epoch
        if self.shuffle and self._index == -1:
            np.random.shuffle(self.image_ids)
        b = 0

        if self._index >= (len(self.image_ids) - self.config.BATCH_SIZE): # skip less than full batches
            raise StopIteration
        else:
            while b < self.batch_size:

                # Get GT bounding boxes and masks for image.
                image_id = self.image_ids[self._index]
                self._index = self._index + 1
                image, image_meta, gt_class_ids, gt_boxes = \
                    load_image_gt(self.dataset, self.config, image_id,
                                  augmentation=self.augmentation)

                # Skip images that have no instances. This can happen in cases
                # where we train on a subset of classes and the image doesn't
                # have any of the classes we care about.
                if not np.any(gt_class_ids > 0):
                    continue

                # RPN Targets
                rpn_match, rpn_bbox = build_rpn_targets(image.shape, self.anchors,
                                                        gt_class_ids, gt_boxes, self.config)

                # Mask R-CNN Targets
                if self.random_rois:
                    rpn_rois = generate_random_rois(
                        image.shape, self.random_rois, gt_class_ids, gt_boxes)
                    if self.detection_targets:
                        rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask = \
                            build_detection_targets(
                                rpn_rois, gt_class_ids, gt_boxes, gt_masks, self.config)

                # Init batch arrays
                if b == 0:
                    batch_image_meta = np.zeros(
                        (self.batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                    batch_rpn_match = np.zeros(
                        [self.batch_size, self.anchors.shape[0], 1], dtype=rpn_match.dtype)
                    batch_rpn_bbox = np.zeros(
                        [self.batch_size, self.config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                    batch_images = np.zeros(
                        (self.batch_size,) + image.shape, dtype=np.float32)
                    batch_gt_class_ids = np.zeros(
                        (self.batch_size, self.config.MAX_GT_INSTANCES), dtype=np.int32)
                    batch_gt_boxes = np.zeros(
                        (self.batch_size, self.config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                    # batch_gt_masks = np.zeros(
                    #     (self.batch_size, gt_masks.shape[0], gt_masks.shape[1],
                    #      self.config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)
                    if self.random_rois:
                        batch_rpn_rois = np.zeros(
                            (self.batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                        if self.detection_targets:
                            batch_rois = np.zeros(
                                (self.batch_size,) + rois.shape, dtype=rois.dtype)
                            batch_mrcnn_class_ids = np.zeros(
                                (self.batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
                            batch_mrcnn_bbox = np.zeros(
                                (self.batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                            # batch_mrcnn_mask = np.zeros(
                            #     (self.batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)

                # If more instances than fits in the array, sub-sample from them.
                if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
                    ids = np.random.choice(
                        np.arange(gt_boxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
                    gt_class_ids = gt_class_ids[ids]
                    gt_boxes = gt_boxes[ids]
                    # gt_masks = gt_masks[:, :, ids]

                # Add to batch
                batch_image_meta[b] = image_meta
                batch_rpn_match[b] = rpn_match[:, np.newaxis]
                batch_rpn_bbox[b] = rpn_bbox
                # batch_images[b] = (image.astype(np.float32) - image.mean())/image.std()
                batch_images[b] = image.astype(np.float32)
                batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
                batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
                # batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
                if self.random_rois:
                    batch_rpn_rois[b] = rpn_rois
                    if self.detection_targets:
                        batch_rois[b] = rois
                        batch_mrcnn_class_ids[b] = mrcnn_class_ids
                        batch_mrcnn_bbox[b] = mrcnn_bbox
                        # batch_mrcnn_mask[b] = mrcnn_mask
                b += 1

            inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                      batch_gt_class_ids, batch_gt_boxes]
            outputs = []

            if self.random_rois:
                inputs.extend([batch_rpn_rois])
                if self.detection_targets:
                    inputs.extend([batch_rois])
                    # Keras requires that output and targets have the same number of dimensions
                    batch_mrcnn_class_ids = np.expand_dims(
                        batch_mrcnn_class_ids, -1)
                    outputs.extend(
                        [batch_mrcnn_class_ids, batch_mrcnn_bbox])#, batch_mrcnn_mask])

            return inputs, outputs


def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:,0]
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinement() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox

def generate_random_rois(image_shape, count, gt_class_ids, gt_boxes):
    """Generates ROI proposals similar to what a region proposal network
    would generate.

    image_shape: [Height, Width, Depth]
    count: Number of ROIs to generate
    gt_class_ids: [N] Integer ground truth class IDs
    gt_boxes: [N, (y1, x1, y2, x2)] Ground truth boxes in pixels.

    Returns: [count, (y1, x1, y2, x2)] ROI boxes in pixels.
    """
    # placeholder
    rois = np.zeros((count, 4), dtype=np.int32)

    # Generate random ROIs around GT boxes (90% of count)
    rois_per_box = int(0.9 * count / gt_boxes.shape[0])
    for i in range(gt_boxes.shape[0]):
        gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
        h = gt_y2 - gt_y1
        w = gt_x2 - gt_x1
        # random boundaries
        r_y1 = max(gt_y1 - h, 0)
        r_y2 = min(gt_y2 + h, image_shape[0])
        r_x1 = max(gt_x1 - w, 0)
        r_x2 = min(gt_x2 + w, image_shape[1])

        # To avoid generating boxes with zero area, we generate double what
        # we need and filter out the extra. If we get fewer valid boxes
        # than we need, we loop and try again.
        while True:
            y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box * 2, 2))
            x1x2 = np.random.randint(r_x1, r_x2, (rois_per_box * 2, 2))
            # Filter out zero area boxes
            threshold = 1
            y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                        threshold][:rois_per_box]
            x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                        threshold][:rois_per_box]
            if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
                break

        # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
        # into x1, y1, x2, y2 order
        x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
        y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
        box_rois = np.hstack([y1, x1, y2, x2])
        rois[rois_per_box * i:rois_per_box * (i + 1)] = box_rois

    # Generate random ROIs anywhere in the image (10% of count)
    remaining_count = count - (rois_per_box * gt_boxes.shape[0])
    # To avoid generating boxes with zero area, we generate double what
    # we need and filter out the extra. If we get fewer valid boxes
    # than we need, we loop and try again.
    while True:
        y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
        x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
        # Filter out zero area boxes
        threshold = 1
        y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                    threshold][:remaining_count]
        x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                    threshold][:remaining_count]
        if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
            break

    # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
    # into x1, y1, x2, y2 order
    x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
    y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
    global_rois = np.hstack([y1, x1, y2, x2])
    rois[-remaining_count:] = global_rois
    return rois

def build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config):
    """Generate targets for training Stage 2 classifier and mask heads.
    This is not used in normal training. It's useful for debugging or to train
    the Mask RCNN heads without using the RPN head.

    Inputs:
    rpn_rois: [N, (y1, x1, y2, x2)] proposal boxes.
    gt_class_ids: [instance count] Integer class IDs
    gt_boxes: [instance count, (y1, x1, y2, x2)]
    gt_masks: [height, width, instance count] Ground truth masks. Can be full
              size or mini-masks.

    Returns:
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    bboxes: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (y, x, log(h), log(w))]. Class-specific
            bbox refinements.
    masks: [TRAIN_ROIS_PER_IMAGE, height, width, NUM_CLASSES). Class specific masks cropped
           to bbox boundaries and resized to neural network output size.
    """
    assert rpn_rois.shape[0] > 0
    assert gt_class_ids.dtype == np.int32, "Expected int but got {}".format(
        gt_class_ids.dtype)
    assert gt_boxes.dtype == np.int32, "Expected int but got {}".format(
        gt_boxes.dtype)
    assert gt_masks.dtype == np.bool_, "Expected bool but got {}".format(
        gt_masks.dtype)

    # It's common to add GT Boxes to ROIs but we don't do that here because
    # according to XinLei Chen's paper, it doesn't help.

    # Trim empty padding in gt_boxes and gt_masks parts
    instance_ids = np.where(gt_class_ids > 0)[0]
    assert instance_ids.shape[0] > 0, "Image must contain instances."
    gt_class_ids = gt_class_ids[instance_ids]
    gt_boxes = gt_boxes[instance_ids]
    gt_masks = gt_masks[:, :, instance_ids]

    # Compute areas of ROIs and ground truth boxes.
    rpn_roi_area = (rpn_rois[:, 2] - rpn_rois[:, 0]) * \
        (rpn_rois[:, 3] - rpn_rois[:, 1])
    gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
        (gt_boxes[:, 3] - gt_boxes[:, 1])

    # Compute overlaps [rpn_rois, gt_boxes]
    overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
    for i in range(overlaps.shape[1]):
        gt = gt_boxes[i]
        overlaps[:, i] = utils.compute_iou(
            gt, rpn_rois, gt_box_area[i], rpn_roi_area)

    # Assign ROIs to GT boxes
    rpn_roi_iou_argmax = np.argmax(overlaps, axis=1)
    rpn_roi_iou_max = overlaps[np.arange(
        overlaps.shape[0]), rpn_roi_iou_argmax]
    # GT box assigned to each ROI
    rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]
    rpn_roi_gt_class_ids = gt_class_ids[rpn_roi_iou_argmax]

    # Positive ROIs are those with >= 0.5 IoU with a GT box.
    fg_ids = np.where(rpn_roi_iou_max > 0.5)[0]

    # Negative ROIs are those with max IoU 0.1-0.5 (hard example mining)
    # TODO: To hard example mine or not to hard example mine, that's the question
    # bg_ids = np.where((rpn_roi_iou_max >= 0.1) & (rpn_roi_iou_max < 0.5))[0]
    bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]

    # Subsample ROIs. Aim for 33% foreground.
    # FG
    fg_roi_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    if fg_ids.shape[0] > fg_roi_count:
        keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False)
    else:
        keep_fg_ids = fg_ids
    # BG
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep_fg_ids.shape[0]
    if bg_ids.shape[0] > remaining:
        keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
    else:
        keep_bg_ids = bg_ids
    # Combine indices of ROIs to keep
    keep = np.concatenate([keep_fg_ids, keep_bg_ids])
    # Need more?
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep.shape[0]
    if remaining > 0:
        # Looks like we don't have enough samples to maintain the desired
        # balance. Reduce requirements and fill in the rest. This is
        # likely different from the Mask RCNN paper.

        # There is a small chance we have neither fg nor bg samples.
        if keep.shape[0] == 0:
            # Pick bg regions with easier IoU threshold
            bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
            assert bg_ids.shape[0] >= remaining
            keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
            assert keep_bg_ids.shape[0] == remaining
            keep = np.concatenate([keep, keep_bg_ids])
        else:
            # Fill the rest with repeated bg rois.
            keep_extra_ids = np.random.choice(
                keep_bg_ids, remaining, replace=True)
            keep = np.concatenate([keep, keep_extra_ids])
    assert keep.shape[0] == config.TRAIN_ROIS_PER_IMAGE, \
        "keep doesn't match ROI batch size {}, {}".format(
            keep.shape[0], config.TRAIN_ROIS_PER_IMAGE)

    # Reset the gt boxes assigned to BG ROIs.
    rpn_roi_gt_boxes[keep_bg_ids, :] = 0
    rpn_roi_gt_class_ids[keep_bg_ids] = 0

    # For each kept ROI, assign a class_id, and for FG ROIs also add bbox refinement.
    rois = rpn_rois[keep]
    roi_gt_boxes = rpn_roi_gt_boxes[keep]
    roi_gt_class_ids = rpn_roi_gt_class_ids[keep]
    roi_gt_assignment = rpn_roi_iou_argmax[keep]

    # Class-aware bbox deltas. [y, x, log(h), log(w)]
    bboxes = np.zeros((config.TRAIN_ROIS_PER_IMAGE,
                       config.NUM_CLASSES, 4), dtype=np.float32)
    pos_ids = np.where(roi_gt_class_ids > 0)[0]
    bboxes[pos_ids, roi_gt_class_ids[pos_ids]] = utils.box_refinement(
        rois[pos_ids], roi_gt_boxes[pos_ids, :4])
    # Normalize bbox refinements
    bboxes /= config.BBOX_STD_DEV

    # Generate class-specific target masks
    masks = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0], config.MASK_SHAPE[1], config.NUM_CLASSES),
                     dtype=np.float32)
    for i in pos_ids:
        class_id = roi_gt_class_ids[i]
        assert class_id > 0, "class id must be greater than 0"
        gt_id = roi_gt_assignment[i]
        class_mask = gt_masks[:, :, gt_id]

        if config.USE_MINI_MASK:
            # Create a mask placeholder, the size of the image
            placeholder = np.zeros(config.IMAGE_SHAPE[:2], dtype=bool)
            # GT box
            gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[gt_id]
            gt_w = gt_x2 - gt_x1
            gt_h = gt_y2 - gt_y1
            # Resize mini mask to size of GT box
            placeholder[gt_y1:gt_y2, gt_x1:gt_x2] = \
                np.round(utils.resize(class_mask, (gt_h, gt_w))).astype(bool)
            # Place the mini batch in the placeholder
            class_mask = placeholder

        # Pick part of the mask and resize it
        y1, x1, y2, x2 = rois[i].astype(np.int32)
        m = class_mask[y1:y2, x1:x2]
        mask = utils.resize(m, config.MASK_SHAPE)
        masks[i, :, :, class_id] = mask

    return rois, roi_gt_class_ids, bboxes, masks


def load_image_gt(dataset, config, image_id, augmentation=None):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    RDEModel: A binary image is returned (1 colour channel), augmentation will be added in future releases
    Returns:
    image: [height, width, 1]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    # mask, class_ids = dataset.load_mask(image_id)
    bbox, class_ids = dataset.load_bboxes(image_id)

    original_shape = image.shape
    image, window, scale, padding, crop = utils.resize_image(
        image,
        # min_dim=config.IMAGE_MIN_DIM,
        min_dim=config.IMAGE_SHAPE[0],
        min_scale=config.IMAGE_MIN_SCALE,
        # max_dim=config.IMAGE_MAX_DIM,
        max_dim=config.IMAGE_SHAPE[0],
        # mode=config.IMAGE_RESIZE_MODE)
        mode='square')
    # mask = utils.resize_mask(mask, scale, padding, crop)
    bbox = utils.resize_bbox(bbox, scale, padding, crop )
    # # Augmentation
    # # This requires the imgaug lib (https://github.com/aleju/imgaug)
    # if augmentation:
    #     import imgaug
    #
    #     # Augmenters that are safe to apply to masks
    #     # Some, such as Affine, have settings that make them unsafe, so always
    #     # test your augmentation on masks
    #     MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
    #                        "Fliplr", "Flipud", "CropAndPad",
    #                        "Affine", "PiecewiseAffine"]
    #
    #     def hook(images, augmenter, parents, default):
    #         """Determines which augmenters to apply to masks."""
    #         return augmenter.__class__.__name__ in MASK_AUGMENTERS
    #
    #     # Store shapes before augmentation to compare
    #     image_shape = image.shape
    #     # mask_shape = mask.shape
    #     # Make augmenters deterministic to apply similarly to images and masks
    #     det = augmentation.to_deterministic()
    #     image = det.augment_image(image)
    #     # Change mask to np.uint8 because imgaug doesn't support np.bool
    #     # mask = det.augment_image(mask.astype(np.uint8),
    #     #                          hooks=imgaug.HooksImages(activator=hook))
    #     # Verify that shapes didn't change
    #     assert image.shape == image_shape, "Augmentation shouldn't change image size"
    #     # assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
    #     # Change mask back to bool
    #     # mask = mask.astype(np.bool)

    # # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # # and here is to filter them out
    # _idx = np.sum(mask, axis=(0, 1)) > 0
    # # mask = mask[:, :, _idx]
    # class_ids = class_ids[_idx]
    # # Bounding boxes. Note that some boxes might be all zeros
    # # if the corresponding mask got cropped out.
    # # bbox: [num_instances, (y1, x1, y2, x2)]
    # # bbox = utils.extract_bboxes(mask)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # # Resize masks to smaller size to reduce memory usage
    # if config.USE_MINI_MASK:
    #     mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # Image meta data
    image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox#, mask


def preprocess_image(image, config, image_id, active_class_ids):
    """Preprocesses an image prior to prediction.
    Similar to `load_image_gt`, but does not manipulate classes or bboxes.

    Takes in the raw image and appropriate config with the `image_id` usually set to a dummy variable of -1.
    Resizes the image, and composes image meta. Returns the resized and padded image."""

    original_shape = image.shape
    # Convert to grayscale
    if original_shape[-1] == 4:
        image = rgba2rgb(image)
    if len(original_shape) == 3:
        image = rgb2gray(image)

    assert len(image.shape) == 2
    image = (image - image.min()) / (image.max() - image.min())
    print(image.max())
    print(image.min())
    dist_0 = np.abs(image.mean() - 0)
    dist_1 = np.abs(image.mean() - 1)
    if dist_1 > dist_0:  # background is 0
        image = image > 0.15
        sum_px = np.sum(image)
        num_px = image.shape[0] * image.shape[1]
        num_fg_px = sum_px
        num_bg_px = num_px - sum_px
    elif dist_1 < dist_0:  # background is 1
        image = image < 0.85
        sum_px = np.sum(image)
        num_px = image.shape[0] * image.shape[1]

        num_bg_px = sum_px
        num_fg_px = num_px - sum_px

    if num_fg_px > num_bg_px:  # background is white
        image = np.abs(1 - image)

    image = image[:, :, np.newaxis]

    image, window, scale, padding, crop = utils.resize_image(
        image,
        # min_dim=config.IMAGE_MIN_DIM,
        min_dim=config.IMAGE_SHAPE[0],
        min_scale=config.IMAGE_MIN_SCALE,
        # max_dim=config.IMAGE_MAX_DIM,
        max_dim=config.IMAGE_SHAPE[0],
        # mode=config.IMAGE_RESIZE_MODE)
        mode='square')

    # Image meta data
    if len(original_shape) == 2:
        original_shape += (1,)
    image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                               window, scale, active_class_ids)

    return image.astype(np.float64), image_meta

def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    assert len(original_image_shape) == 3, "Input image needs to have 3 dimensions. For 2D image, add a dummy axis"
    #for debugging
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +           # size=3
        list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +                     # size=1
        list(active_class_ids)        # size=num_classes
    )
    return meta

def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed tensors.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }
