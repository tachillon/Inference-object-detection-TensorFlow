#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# License: © 2020 Achille-Tâm GUILCHARD All Rights Reserved
# Author: Achille-Tâm GUILCHARD

from object_detection.utils import label_map_util
import sys
import os
import time
import tensorflow as tf
import numpy as np
import cv2
import shutil
from os import listdir
from os.path import isfile, join

sys.path.append('/home/models/research')
sys.path.append('/home/models/research/object_detection')
sys.path.append('/home/models/research/slim')

# Constants
ALPHA          = 0.75
FONT           = cv2.FONT_HERSHEY_PLAIN
TEXT_SCALE     = 1
TEXT_THICKNESS = 1
BLACK          = (0, 0, 0)
WHITE          = (255, 255, 255)


def list_files1(directory, extension):
    res_list = []
    dirFiles = os.listdir(directory)
    sorted(dirFiles)  # sort numerically in ascending order
    for f in dirFiles:
        if f.endswith('.' + extension) or f.endswith('.' + 'jpg') or f.endswith(
                '.' + 'jpeg') or f.endswith('.' + 'tif') or f.endswith('.' + 'png'):
            res_list.append(directory + "/" + f)
    return res_list


def draw_boxed_text(img, text, topleft, color):
    """Draw a transluent boxed text in white, overlayed on top of a
    colored patch surrounded by a black border. FONT, TEXT_SCALE,
    TEXT_THICKNESS and ALPHA values are constants (fixed) as defined
    on top.

    # Arguments
      img: the input image as a numpy array.
      text: the text to be drawn.
      topleft: XY coordinate of the topleft corner of the boxed text.
      color: color of the patch, i.e. background of the text.

    # Output
      img: note the original image is modified inplace.
    """
    assert img.dtype == np.uint8
    img_h, img_w, _ = img.shape
    if topleft[0] >= img_w or topleft[1] >= img_h:
        return img
    margin = 3
    size = cv2.getTextSize(text, FONT, TEXT_SCALE, TEXT_THICKNESS)
    w = size[0][0] + margin * 2
    h = size[0][1] + margin * 2
    # the patch is used to draw boxed text
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    patch[...] = color
    cv2.putText(patch, text, (margin + 1, h - margin - 2), FONT, TEXT_SCALE,
                WHITE, thickness=TEXT_THICKNESS, lineType=cv2.LINE_8)
    cv2.rectangle(patch, (0, 0), (w - 1, h - 1), BLACK, thickness=1)
    w = min(w, img_w - topleft[0])  # clip overlay at image boundary
    h = min(h, img_h - topleft[1])
    # Overlay the boxed text onto region of interest (roi) in img
    roi = img[topleft[1]:topleft[1] + h, topleft[0]:topleft[0] + w, :]
    cv2.addWeighted(patch[0:h, 0:w, :], ALPHA, roi, 1 - ALPHA, 0, roi)
    return img


def draw_bboxes(img, bb, confs, clss):
    """Draw detected bounding boxes on the original image."""
    x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
    color = (0, 255, 0)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
    txt_loc = (max(x_min, 0), max(y_min - 18, 0))
    txt = '{} {:.2f}'.format(clss, confs)
    img = draw_boxed_text(img, txt, txt_loc, color)
    return img


class ImageDetection():
    def __init__(self, PATH_TO_FROZEN_GRAPH, PATH_TO_LABELS):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            self.od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as self.fid:
                self.serialized_graph = self.fid.read()
                self.od_graph_def.ParseFromString(self.serialized_graph)
                tf.import_graph_def(self.od_graph_def, name='')

                self.category_index = label_map_util.create_category_index_from_labelmap(
                    PATH_TO_LABELS, use_display_name=True)

                # Get handles to input and output tensors
                self.ops = tf.compat.v1.get_default_graph().get_operations()
                self.all_tensor_names = {
                    self.output.name for self.op in self.ops for self.output in self.op.outputs}
                self.tensor_dict = {}
                for self.key in [
                    'num_detections',
                    'detection_boxes',
                    'detection_scores',
                    'detection_classes',
                        'detection_masks']:
                    self.tensor_name = self.key + ':0'
                    if self.tensor_name in self.all_tensor_names:
                        self.tensor_dict[self.key] = tf.compat.v1.get_default_graph(
                        ).get_tensor_by_name(self.tensor_name)
                self.image_tensor = tf.compat.v1.get_default_graph(
                ).get_tensor_by_name('image_tensor:0')

                self.sess = tf.compat.v1.Session()

    def detect(self, img):
        # Actual detection.
        output_dict = self.sess.run(
            self.tensor_dict, feed_dict={
                self.image_tensor: np.expand_dims(
                    img, 0)})
        return output_dict


def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        #overlap = (w * h) / area[idxs[:last]]
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(
            ([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def main():
    PATH_TO_FROZEN_GRAPH = "/tmp/model/frozen_inference_graph.pb"
    PATH_TO_LABELS       = "/tmp/model/label.pbtxt"
    IMG_DIR              = "/tmp/images"
    img_detector         = ImageDetection(PATH_TO_FROZEN_GRAPH, PATH_TO_LABELS)
    category_index       = label_map_util.create_category_index_from_labelmap(
        PATH_TO_LABELS, use_display_name=True)
    max_boxes_to_draw = 20
    min_score_thresh  = 0.5
    shutil.rmtree("/tmp/resultats", ignore_errors=True)
    os.makedirs("/tmp/resultats", exist_ok=True)

    listFiles = list_files1(IMG_DIR, 'png')
    listFiles.sort()
    
    for img in listFiles:
        basename = os.path.basename(img) 
        print(">Processing file: " + basename)
        bb_pp_c = []

        inputImg     = cv2.imread(img)
        initHeight   = inputImg.shape[0]
        initWidth    = inputImg.shape[1]
        initChannels = inputImg.shape[2]

        im_rgb = cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB)
        width  = 1217
        height = 684
        dim    = (width, height)
        resized     = cv2.resize(im_rgb, dim, interpolation=cv2.INTER_CUBIC)
        tic = time.perf_counter()
        output_dict = img_detector.detect(resized)
        toc = time.perf_counter()
        resized     = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(
            np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]

        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]

        for j in range(len(output_dict['detection_boxes'])):
            output_dict['detection_boxes'][j][0] = output_dict['detection_boxes'][j][0] * initHeight
            output_dict['detection_boxes'][j][1] = output_dict['detection_boxes'][j][1] * initWidth
            output_dict['detection_boxes'][j][2] = output_dict['detection_boxes'][j][2] * initHeight
            output_dict['detection_boxes'][j][3] = output_dict['detection_boxes'][j][3] * initWidth

            bouding_box_ = []
            bouding_box_.append(output_dict['detection_boxes'][j][0])
            bouding_box_.append(output_dict['detection_boxes'][j][1])
            bouding_box_.append(output_dict['detection_boxes'][j][2])
            bouding_box_.append(output_dict['detection_boxes'][j][3])
            bouding_box_.append(output_dict['detection_scores'][j])

            if output_dict['detection_scores'][j] > min_score_thresh:
                bb_pp_c.append(bouding_box_)

        bb_pp_c_np = np.array(bb_pp_c, dtype=np.float32)

        # NMS
        bb_pp_c_np = non_max_suppression_fast(bb_pp_c_np, 0.45)

        number_of_repetition = 0
        for j in range(len(bb_pp_c_np)):
            ymin = bb_pp_c_np[j][0]
            xmin = bb_pp_c_np[j][1]
            ymax = bb_pp_c_np[j][2]
            xmax = bb_pp_c_np[j][3]
            h = int(ymax) - int(ymin)
            w = int(xmax) - int(xmin)
            bbox = [xmin, ymin, xmax, ymax]
            if (h != 0 and w != 0):
                inputImg = draw_bboxes(inputImg, bbox, 100, "item")
        print(f"    Inference done in {toc - tic:0.4f} seconds")
        cv2.imwrite("/tmp/resultats/" + basename, inputImg)


if __name__ == "__main__":
    # execute only if run as a script
    main()
