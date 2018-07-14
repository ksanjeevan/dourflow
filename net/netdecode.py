import tensorflow as tf
from keras import backend as K
import numpy as np 

from net.netparams import YoloParams


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

        self.max_boxes = YoloParams.TRUE_BOX_BUFFER
        self.nms_threshold = YoloParams.NMS_THRESHOLD
        self.detection_threshold = YoloParams.DETECTION_THRESHOLD


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

        selected_indices = tf.image.non_max_suppression(
                                        flatten_boxes, 
                                        flatten_scores, 
                                        max_output_size=self.max_boxes, 
                                        iou_threshold=self.nms_threshold)
        
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

    sess = tf.InteractiveSession()

    max_boxes = 10
    nms_threshold = 0.1
    boxes = tf.convert_to_tensor(np.random.rand(10,4), np.float32)
    scores = tf.convert_to_tensor(np.random.rand(10,), np.float32)

    classes = tf.convert_to_tensor((10.*np.random.rand(10,)%3).astype(int), np.float32)

    s,b,c = yolo_non_max_suppression(scores, boxes, classes, max_boxes, nms_threshold)

    print(boxes.eval().shape)
    print(scores.eval().shape)
    print(classes.eval().shape)

    print('-----------------------')

    print(b.eval().shape)
    print(s.eval().shape)
    print(c.eval().shape)

