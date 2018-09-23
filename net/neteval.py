"""
Evaluate results. mAP validation, tensorboard metrics:
mAP callback, recall.
"""

from .netparams import YoloParams
from .utils import draw_boxes, compute_iou, mkdir_p, handle_empty_indexing
from .netloss import _transform_netout, calculate_ious

import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2, os
import keras

from keras import backend as K






class YoloEvaluate(object):


    def __init__(self, generator, model):

        self.inf_model = model
        self.generator = generator
        self.class_labels = np.array(YoloParams.CLASS_LABELS)

        self.iou_detection_threshold = YoloParams.IOU_THRESHOLD

        self.val_out_path = YoloParams.VALIDATION_OUT_PATH
        self.debug_plots = True if self.val_out_path else False
    
        if self.debug_plots: mkdir_p(self.val_out_path)        


    def _find_detection(self, q_box, boxes, global_index):

        if boxes.size == 0:
            #print('EMPTY BOXES')
            return -1

        ious = list(map(lambda x: compute_iou(q_box, x), boxes))

        max_iou_index = np.argmax( ious )

        if ious[max_iou_index] > self.iou_detection_threshold:
            return global_index[max_iou_index]

        return -1


    def _plot_preds(self, image, pred_info, true_info, image_index):

        image_out = draw_boxes(image, pred_info)
        image_out = draw_boxes(image_out, true_info)
        image_name = os.path.basename( self.generator.load_image_name(image_index) )
    
        outfile = os.path.join(self.val_out_path, image_name)
        cv2.imwrite(outfile, image_out)



    def _process_image(self, i):
        
        true_boxes, true_labels = self.generator.load_annotation(i)

        image = self.generator.load_image(i)

        pred_boxes, conf, pred_labels, _ = self.inf_model.predict(image.copy())

        if self.debug_plots:

            # np.array(YoloParams.CLASS_LABELS)[pred_labels]
            label_names_pred = handle_empty_indexing(self.class_labels, pred_labels)
            label_names_true = self.class_labels[true_labels]

            pred_info = (pred_boxes, conf, label_names_pred)
            true_info = (true_boxes, None, label_names_true)

            self._plot_preds(image.copy(), pred_info=pred_info, true_info=true_info, image_index=i)


        sorted_inds = np.argsort(-conf)

        repeat_mask = [True]*len(true_boxes)
        matched_labels = []
        global_index = np.arange(len(true_labels))


        image_results = []
        image_labels = [0]*YoloParams.NUM_CLASSES

        for tl in true_labels:
            image_labels[tl] += 1


        for i in sorted_inds:

            label_mask = (pred_labels[i] == true_labels)
            index_subset = global_index[(repeat_mask)&(label_mask)]
            true_boxes_subset = true_boxes[(repeat_mask)&(label_mask)]

            idx = self._find_detection(pred_boxes[i], true_boxes_subset, index_subset)

            if idx != -1: 
                matched_labels.append(idx)
                repeat_mask[idx] = False

            image_results.append([pred_labels[i], conf[i], 1 if idx != -1 else 0])

        return image_results, image_labels


    def _interp_ap(self, precision, recall):

        if precision.size == 0 or recall.size == 0:
            return 0.

        iap = 0
        for r in np.arange(0.,1.1, 0.1):
            recall_mask = (recall >= r)
            p_max = precision[recall_mask]
            
            iap += np.max( p_max if p_max.size > 0 else [0] )

        return iap / 11


    def compute_ap(self, detections, num_gts):

        detections_sort_indx = np.argsort(-detections[:,1])
        detections = detections[detections_sort_indx]

        precision = []
        recall = []

        if num_gts == 0:
            return 0.

        for i in range(1, len(detections) + 1):

            precision.append( np.sum(detections[:i][:,2]) / i )
            recall.append( np.sum(detections[:i][:,2]) / num_gts )

        return self._interp_ap(np.array(precision), np.array(recall))

    def comp_map(self):

        detection_results = []
        detection_labels = np.array([0]*YoloParams.NUM_CLASSES) 

        for i in tqdm(range(len(self.generator.images)), desc='Batch Processed'):

            image_name = os.path.basename( self.generator.load_image_name(i) )

            #if image_name == '2011_003285.jpg':

            image_results, image_labels = self._process_image(i)

            detection_results.extend(image_results)
            detection_labels += np.array(image_labels)


        detection_results = np.array(detection_results)

        ap_dic = {}
        for class_ind, num_gts in enumerate(detection_labels):
            
            class_detections = detection_results[detection_results[:,0]==class_ind]            
            
            ap = self.compute_ap(class_detections, num_gts)

            ap_dic[self.class_labels[class_ind]] = ap


        return ap_dic



class Callback_MAP(keras.callbacks.Callback):

    def __init__(self, generator, model, tensorboard):

        self.yolo_eval = YoloEvaluate(generator=generator, model=model)
        self.tensorboard = tensorboard

    def on_epoch_end(self, epoch, logs={}):
        
        mAP_dict = self.yolo_eval.comp_map()

        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = np.mean(list(mAP_dict.values()))
        summary_value.tag = "mAP"
        #self.tensorboard.writer.add_summary(summary, epoch)

        self.tensorboard.val_writer.add_summary(summary, epoch)

        self.tensorboard.val_writer.flush()



def yolo_recall(y_true, y_pred_raw):
    truth = y_true[...,4]

    y_pred = _transform_netout(y_pred_raw)
    ious = calculate_ious(y_true, y_pred, use_iou=True)
    pred_ious = K.cast(ious > YoloParams.IOU_THRESHOLD, np.float32)

    scores = y_pred[..., 4:5] * y_pred[...,5:]
    pred_scores = K.cast(K.max(scores, axis=-1) > YoloParams.DETECTION_THRESHOLD, np.float32)

    tp = K.sum(pred_ious * pred_scores) 
    tpfn = K.sum(truth)

    return tp / (tpfn + 1e-8)



# https://stackoverflow.com/questions/47877475/keras-tensorboard-plot-train-and-validation-scalars-in-a-same-figure?rq=1

def in_loss_decmop(k):
    return any([term in k for term in ['coord','obj','class']])

class YoloTensorBoard(keras.callbacks.TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(YoloTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')


        self.loss_dir = {
                'training':{
                    'coord':os.path.join(training_log_dir, 'coordinate'),
                    'obj':os.path.join(training_log_dir, 'confidence'),
                    'class':os.path.join(training_log_dir, 'class')
                },
                'validation':{
                    'coord':os.path.join(self.val_log_dir, 'coordinate'),
                    'obj':os.path.join(self.val_log_dir, 'confidence'),
                    'class':os.path.join(self.val_log_dir, 'class')
                }
        }

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)

        self.loss_writer = {}
        for k,v in self.loss_dir.items():
            self.loss_writer[k] = {}
            for l,m in v.items():
                self.loss_writer[k][l] = tf.summary.FileWriter(m) 

        super(YoloTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        loss_logs = {k:v for k, v in logs.items() if in_loss_decmop(k)}
        logs = {k:v for k, v in logs.items() if not in_loss_decmop(k)}
        
        for name, value in loss_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            
            decomp_part = name.replace('val_', '').replace('l_', '')

            key = ('val', 'validation') if name.startswith('val_') else ('train', 'training')
        
            summary_value.tag = key[0] + '_loss_decomp'    
            self.loss_writer[key[1]][decomp_part].add_summary(summary, epoch)
            self.loss_writer[key[1]][decomp_part].flush()


        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(YoloTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(YoloTensorBoard, self).on_train_end(logs)
        self.val_writer.close()



