"""
Process [GRID x GRID x BOXES x (4 + 1 + CLASSES)]. Filter low confidence
boxes, apply NMS and return boxes, scores, classes.
"""

import tensorflow as tf
from keras import backend as K
import numpy as np 

from .netparams import YoloParams


def process_outs(b, s, c):
    
    b_p = b
    # Expand dims of scores and classes so we can concat them 
    # with the boxes and have the output of NMS as an added layer of YOLO.
    # Have to do another expand_dims this time on the first dim of the result
    # since NMS doesn't know about BATCH_SIZE (operates on 2D, see 
    # https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression) 
    # but keras needs this dimension in the output.
    s_p = K.expand_dims(s, axis=-1)
    c_p = K.expand_dims(c, axis=-1)
    
    output_stack = K.concatenate([b_p, s_p, c_p], axis=1)
    return K.expand_dims(output_stack, axis=0)


class YoloOutProcess(object):


    def __init__(self):
        # thresholds
        self.max_boxes = YoloParams.TRUE_BOX_BUFFER
        self.nms_threshold = YoloParams.NMS_THRESHOLD
        self.detection_threshold = YoloParams.DETECTION_THRESHOLD

        self.num_classes = YoloParams.NUM_CLASSES

    def __call__(self, y_sing_pred):

        # need to convert b's from GRID_SIZE units into IMG coords. Divide by grid here. 
        b_xy = (K.sigmoid(y_sing_pred[..., 0:2]) + YoloParams.c_grid[0]) / YoloParams.GRID_SIZE
        b_wh = (K.exp(y_sing_pred[..., 2:4])*YoloParams.anchors[0]) / YoloParams.GRID_SIZE
        b_xy1 = b_xy - b_wh / 2.
        b_xy2 = b_xy + b_wh / 2.
        boxes = K.concatenate([b_xy1, b_xy2], axis=-1)
        
        # filter out scores below detection threshold
        scores_all = K.sigmoid(y_sing_pred[..., 4:5]) * K.softmax(y_sing_pred[...,5:])
        indicator_detection = scores_all > self.detection_threshold
        scores_all = scores_all * K.cast(indicator_detection, np.float32)

        # compute detected classes and scores
        classes = K.argmax(scores_all, axis=-1)
        scores = K.max(scores_all, axis=-1)

        # flattened tensor length
        S2B = YoloParams.GRID_SIZE*YoloParams.GRID_SIZE*YoloParams.NUM_BOUNDING_BOXES

        # flatten boxes, scores for NMS
        flatten_boxes = K.reshape(boxes, shape=(S2B, 4))
        flatten_scores = K.reshape(scores, shape=(S2B, ))
        flatten_classes = K.reshape(classes, shape=(S2B, ))

        inds = []

        # apply multiclass NMS 
        for c in range(self.num_classes):

            # only include boxes of the current class, with > 0 confidence
            class_mask = K.cast(K.equal(flatten_classes, c), np.float32)
            score_mask = K.cast(flatten_scores > 0, np.float32) 
            mask = class_mask * score_mask
            
            # compute class NMS
            nms_inds = tf.image.non_max_suppression(
                    flatten_boxes, 
                    flatten_scores*mask, 
                    max_output_size=self.max_boxes, 
                    iou_threshold=self.nms_threshold,
                    score_threshold=0.
                )
            
            inds.append(nms_inds)

        # combine winning box indices of all classes 
        selected_indices = K.concatenate(inds, axis=-1)
        
        # gather corresponding boxes, scores, class indices
        selected_boxes = K.gather(flatten_boxes, selected_indices)
        selected_scores = K.gather(flatten_scores, selected_indices)
        selected_classes = K.gather(flatten_classes, selected_indices)

        return process_outs(selected_boxes, selected_scores, K.cast(selected_classes, np.float32))




class YoloOutProcessOther(object):
    """
    [UNUSED] Ignore.
    """

    def __init__(self):

        self.max_boxes = YoloParams.TRUE_BOX_BUFFER
        self.nms_threshold = YoloParams.NMS_THRESHOLD
        self.detection_threshold = YoloParams.DETECTION_THRESHOLD

        self.num_classes = YoloParams.NUM_CLASSES


    def _class_nms(self, boxes, scores, c_mask):
        #c_mask = K.equal(classes, i)
        c_mask = c_mask*K.cast(scores > 0, np.float32)
        c_boxes = boxes * K.expand_dims(c_mask, axis=-1)
        c_scores = scores * c_mask
        inds = tf.image.non_max_suppression(c_boxes, c_scores, max_output_size=10, iou_threshold=0.2)
        # tf.pad(inds, tf.Variable([[0,10-tf.shape(inds)[0]]]), "CONSTANT")
        return self._pad_tensor(inds, 10, value=-1)
        

    def _pad_tensor(self, t, length, value=0):
        """Pads the input tensor with 0s along the first dimension up to the length.
        Args:
        t: the input tensor, assuming the rank is at least 1.
        length: a tensor of shape [1]  or an integer, indicating the first dimension
          of the input tensor t after padding, assuming length <= t.shape[0].
        Returns:
        padded_t: the padded tensor, whose first dimension is length. If the length
          is an integer, the first dimension of padded_t is set to length
          statically.
        """
        t_rank = tf.rank(t)
        t_shape = tf.shape(t)
        t_d0 = t_shape[0]
        pad_d0 = tf.expand_dims(length - t_d0, 0)
        pad_shape = tf.cond(
          tf.greater(t_rank, 1), lambda: tf.concat([pad_d0, t_shape[1:]], 0),
          lambda: tf.expand_dims(length - t_d0, 0))
        padded_t = tf.concat([t, value+tf.zeros(pad_shape, dtype=t.dtype)], 0)

        t_shape = padded_t.get_shape().as_list()
        t_shape[0] = length
        padded_t.set_shape(t_shape)

        return padded_t

    def __call__(self, y_sing_pred):

        # need to convert b's from GRID_SIZE units into IMG coords. Divide by grid here. 
        b_xy = (K.sigmoid(y_sing_pred[..., 0:2]) + YoloParams.c_grid[0]) / YoloParams.GRID_SIZE
        b_wh = (K.exp(y_sing_pred[..., 2:4])*YoloParams.anchors[0]) / YoloParams.GRID_SIZE
        b_xy1 = b_xy - b_wh / 2.
        b_xy2 = b_xy + b_wh / 2.
        boxes = K.concatenate([b_xy1, b_xy2], axis=-1)
        
        scores_all = K.expand_dims(K.sigmoid(y_sing_pred[..., 4]), axis=-1) * K.softmax(y_sing_pred[...,5:])
        indicator_detection = scores_all > self.detection_threshold
        scores_all = scores_all * K.cast(indicator_detection, np.float32)

        classes = K.argmax(scores_all, axis=-1)
        scores = K.max(scores_all, axis=-1)

        S2B = YoloParams.GRID_SIZE*YoloParams.GRID_SIZE*YoloParams.NUM_BOUNDING_BOXES

        flatten_boxes = K.reshape(boxes, shape=(S2B, 4))
        flatten_scores = K.reshape(scores, shape=(S2B, ))
        flatten_classes = K.reshape(classes, shape=(S2B, ))


        c_masks = K.map_fn(lambda c: K.cast(K.equal(flatten_classes, c), np.float32), np.arange(self.num_classes), dtype=np.float32)
        resu_stacked = tf.map_fn(
            lambda c: self._class_nms(flatten_boxes, flatten_scores, c), 
            c_masks, 
            dtype=np.int32, 
            infer_shape=True)

        resu_flat = K.reshape(resu_stacked, shape=(-1,))
        selected_indices = tf.boolean_mask(resu_flat, ~K.equal(resu_flat, -1))

        selected_boxes = K.gather(flatten_boxes, selected_indices)
        selected_scores = K.gather(flatten_scores, selected_indices)
        selected_classes = K.gather(flatten_classes, selected_indices)

        # Exclude padding boxes left behind by tensorflow NMS
        score_mask = selected_scores>0.
        selected_boxes = tf.boolean_mask(selected_boxes, score_mask)  
        selected_scores = tf.boolean_mask(selected_scores, score_mask)  
        selected_classes = tf.boolean_mask(selected_classes, score_mask)  
        
        return process_outs(selected_boxes, selected_scores, K.cast(selected_classes, np.float32))





if __name__ == '__main__':

    tf.InteractiveSession()

    a = tf.convert_to_tensor(np.load('ocell.npy'), np.float32)
    
    yolo_out = YoloOutProcess()

    resu = yolo_out(a).eval()[0]

    b = resu[:,:4]
    s = resu[:,4]
    c = resu[:,5]

    print('---------------------')

    print(c)
    print(s)
    print(b)
