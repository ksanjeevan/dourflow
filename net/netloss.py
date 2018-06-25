
import tensorflow as tf
import numpy as np

from net.netparams import YoloParams

EPSILON = 1e-6


def calculate_ious(A1, A2, use_iou=True):

    if not use_iou: 
        return 1.

    A1_xy = A1[..., 0:2]
    A1_wh = A1[..., 2:4]

    A2_xy = A2[..., 0:2]
    A2_wh = A2[..., 2:4]
    
    A1_wh_half = A1_wh / 2.
    A1_mins    = A1_xy - A1_wh_half
    A1_maxes   = A1_xy + A1_wh_half
    
    A2_wh_half = A2_wh / 2.
    A2_mins = A2_xy - A2_wh_half
    A2_maxes   = A2_xy + A2_wh_half

    intersect_mins  = tf.maximum(A2_mins,  A1_mins)
    intersect_maxes = tf.minimum(A2_maxes, A1_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    true_areas = A1_wh[..., 0] * A1_wh[..., 1]
    pred_areas = A2_wh[..., 0] * A2_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)

    return iou_scores


class YoloLoss(object):
    # ADD WARM UP CONDITIONS

    def __init__(self):

        self.__name__ = 'yolo_loss'
        self.iou_threshold = YoloParams.IOU_THRESHOLD
        self.readjust_obj_score = True

        self.lambda_coord = YoloParams.COORD_SCALE
        self.lambda_noobj = YoloParams.NO_OBJECT_SCALE
        self.lambda_obj = YoloParams.OBJECT_SCALE
        self.lambda_class = YoloParams.CLASS_SCALE

        self.norm = False

    def coord_loss(self, y_true, y_pred):
        
        b_xy_pred = y_pred[..., :2]
        b_wh_pred = y_pred[..., 2:4]
        
        b_xy = y_true[..., 0:2]
        b_wh = y_true[..., 2:4]

        indicator_coord = tf.expand_dims(y_true[..., 4], axis=-1) * self.lambda_coord

        norm_coord = 1
        if self.norm:
            norm_coord = tf.reduce_sum(tf.to_float(indicator_coord > 0.0))


        loss_xy = tf.reduce_sum(tf.square(b_xy - b_xy_pred) * indicator_coord, axis=[1,2,3,4])
        #loss_wh = tf.reduce_sum(tf.square(b_wh - b_wh_pred) * indicator_coord, axis=[1,2,3,4])
        loss_wh = tf.reduce_sum(tf.square(tf.sqrt(b_wh) - tf.sqrt(b_wh_pred)) * indicator_coord, axis=[1,2,3,4])

        return (loss_wh + loss_xy) / (norm_coord + EPSILON) / 2


    def obj_loss(self, y_true, y_pred):

        b_o = calculate_ious(y_true, y_pred, use_iou=self.readjust_obj_score) * y_true[..., 4]
        b_o_pred = y_pred[..., 4]

        num_true_labels = YoloParams.GRID_SIZE*YoloParams.GRID_SIZE*YoloParams.NUM_BOUNDING_BOXES
        y_true_p = tf.reshape(y_true[..., :4], shape=(YoloParams.BATCH_SIZE, 1, 1, 1, num_true_labels, 4))
        iou_scores_buff = calculate_ious(y_true_p, tf.expand_dims(y_pred, axis=4))
        best_ious = tf.reduce_max(iou_scores_buff, axis=4)

        indicator_noobj = tf.to_float(best_ious < self.iou_threshold) * (1 - y_true[..., 4]) * self.lambda_noobj
        indicator_obj = y_true[..., 4] * self.lambda_obj


        norm_conf = 1
        if self.norm:
            norm_conf = tf.reduce_sum(tf.to_float((indicator_obj + indicator_noobj)  > 0.0))

        loss_obj = tf.reduce_sum(tf.square(b_o-b_o_pred) * (indicator_obj + indicator_noobj), axis=[1,2,3])

        return loss_obj / (norm_conf + EPSILON) / 2


    def class_loss(self, y_true, y_pred):

        b_class = tf.argmax(y_true[..., 5:], axis=-1)
        b_class_pred = y_pred[..., 5:]

        indicator_class = y_true[..., 4] * tf.gather(
            YoloParams.CLASS_WEIGHTS, b_class) * self.lambda_class

        norm_class = 1
        if self.norm:
            norm_class = tf.reduce_sum(tf.to_float(indicator_class > 0.0))

        loss_class_arg = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=b_class, logits=b_class_pred)

        loss_class = tf.reduce_sum(loss_class_arg * indicator_class, axis=[1,2,3])

        return loss_class / (norm_class + EPSILON)


    def _transform_netout(self, y_pred_raw):
        y_pred_xy = tf.sigmoid(y_pred_raw[..., :2]) + YoloParams.c_grid
        y_pred_wh = tf.exp(y_pred_raw[..., 2:4]) * YoloParams.anchors
        y_pred_conf = tf.sigmoid(y_pred_raw[..., 4:5])
        y_pred_class = y_pred_raw[...,5:]

        return tf.concat([y_pred_xy, y_pred_wh, y_pred_conf, y_pred_class], axis=-1)



    def __call__(self, y_true, y_pred_raw):

        y_pred = self._transform_netout(y_pred_raw)
        
        total_coord_loss = self.coord_loss(y_true, y_pred)
        total_obj_loss = self.obj_loss(y_true, y_pred)
        total_class_loss = self.class_loss(y_true, y_pred)

        loss = total_coord_loss + total_obj_loss + total_class_loss

        #loss = tf.Print(loss, [total_coord_loss], message='\nCoord Loss \t', summarize=1000)
        #loss = tf.Print(loss, [total_obj_loss], message='Conf Loss \t', summarize=1000)
        #oss = tf.Print(loss, [total_class_loss], message='Class Loss \t', summarize=1000)
        #oss = loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)

        return  loss




if __name__ == '__main__':
    
    sess = tf.InteractiveSession()

    y_pred = tf.convert_to_tensor(np.random.rand(16,13,13,5,85), np.float32)
    y_true = tf.convert_to_tensor(np.random.rand(16,13,13,5,85), np.float32)

    var = YoloLoss()

    print( var(y_true, y_pred).eval() )






















