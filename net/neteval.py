


from net.netparams import YoloParams
from net.netdecode import YoloOutProcess

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2, os
import keras
from net.utils import draw_boxes, compute_iou, mkdir_p, \
yolo_normalize, mkdir_p, handle_empty_indexing, parse_annotation




'''
def exrtract_wh(img):
    result = []
    pixel_height = img['height']
    pixel_width = img['width']

    fact_pixel_grid_h = YoloParams.GRID_SIZE / pixel_height
    fact_pixel_grid_w = YoloParams.GRID_SIZE / pixel_width

    for obj in img['object']:
        grid_h = (obj['ymax'] - obj['ymin']) *  fact_pixel_grid_h
        grid_w = (obj['xmax'] - obj['xmin']) *  fact_pixel_grid_w
        
        result.append( np.array(grid_h, grid_w) )

    return result

def gen_anchors(fname):

    imgs, _ = parse_annotation(ann_dir, img_dir)

    data_wh = []
    for img in imgs:
        data_wh += exrtract_wh(img)

    c = AgglomerativeClustering(self.num_clusters, affinity='precomputed', linkage=self.c_type)

'''




class YoloDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, images, shuffle=True):

        self.images = self._prune_ann_labels(images)
        self.input_size = YoloParams.INPUT_SIZE
        self.anchors = YoloParams.anchors

        self.generator = None

        self.batch_size = YoloParams.BATCH_SIZE
        
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        bound_l = index*self.batch_size
        bound_r = (index+1)*self.batch_size

        return self._data_to_yolo_output(self.images[bound_l:bound_r])

    def load_image_name(self, i):
        return self.images[i]['filename']


    def load_image(self, i):
        return cv2.imread(self.images[i]['filename'])

    def load_annotation(self, i):
        labels = []
        bboxes = []
        
        height = self.images[i]['height']
        width = self.images[i]['width']

        for obj in self.images[i]['object']:
            #if obj['name'] in YoloParams.CLASS_LABELS:
            labels.append( obj['name'] )
            bboxes.append( 
                [obj['xmin'] / width, obj['ymin'] / height, obj['xmax'] / width, obj['ymax'] / height] )


        class_inds = [YoloParams.CLASS_TO_INDEX[l] for l in labels]

        return np.array(bboxes), np.array(class_inds)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle: np.random.shuffle(self.images)

    def _prune_ann_labels(self, images):
        clean_images = []
        for im in images:
            clean_im = im.copy()
            clean_objs = []
            for obj in clean_im['object']:
                if obj['name'] in YoloParams.CLASS_LABELS:
                    clean_objs.append( obj )

            clean_im.update({'object' : clean_objs})
            clean_images.append(clean_im)

        return clean_images


    def _data_to_yolo_output(self, batch_images):

        # INPUT IMAGES READY FOR TRAINING
        x_batch = np.zeros((len(batch_images), self.input_size, self.input_size, 3))

        # GET DESIRED NETWORK OUTPUT
        y_batch = np.zeros((len(batch_images), YoloParams.GRID_SIZE,  
            YoloParams.GRID_SIZE, YoloParams.NUM_BOUNDING_BOXES, 4+1+len(YoloParams.CLASS_LABELS)))

        grid_factor = YoloParams.GRID_SIZE / self.input_size

        for j, train_instance in enumerate(batch_images):
            
            img_raw = cv2.imread(train_instance['filename'])

            h_factor_resize = img_raw.shape[0] / self.input_size
            w_factor_resize = img_raw.shape[1] / self.input_size 

            img = cv2.resize(img_raw, (self.input_size, self.input_size))

            for obj_box_idx, label in enumerate(train_instance['object']):

                xmin_resized = int(round(label['xmin'] / w_factor_resize))
                xmax_resized = int(round(label['xmax'] / w_factor_resize))
                ymin_resized = int(round(label['ymin'] / h_factor_resize))
                ymax_resized = int(round(label['ymax'] / h_factor_resize))

                bbox_center_x = .5*(xmin_resized + xmax_resized) * grid_factor
                grid_x = int(bbox_center_x)
                
                bbox_center_y = .5*(ymin_resized + ymax_resized) * grid_factor 
                grid_y = int(bbox_center_y)
                
                obj_indx  = YoloParams.CLASS_LABELS.index(label['name'])
                
                bbox_w = (xmax_resized - xmin_resized) * grid_factor
                bbox_h = (ymax_resized - ymin_resized) * grid_factor
                
                shifted_wh = np.array([0,0,bbox_w, bbox_h])

                func = lambda prior: compute_iou((0,0,prior[0],prior[1]), shifted_wh)

                anchor_winner = np.argmax(np.apply_along_axis(func, -1, self.anchors))
                        
                # assign ground truth x, y, w, h, confidence and class probs to y_batch

                # ASSIGN CLASS CONFIDENCE
                y_batch[j, grid_y, grid_x, anchor_winner, 0:4] = [bbox_center_x, bbox_center_y, bbox_w, bbox_h]

                # ASSIGN OBJECTNESS CONF
                y_batch[j, grid_y, grid_x, anchor_winner, 4  ] = 1.

                # ASSIGN CORRECT CLASS TO
                y_batch[j, grid_y, grid_x, anchor_winner, 4+1+obj_indx] = 1
                
                # number of labels per instance !> than true_box_buffer, add check in processing (?)
            x_batch[j] = yolo_normalize(img)

        ############################################################
        # x_batch -> list of input images
        # y_batch -> list of network ouput gt values for each image
        ############################################################
        return x_batch, y_batch




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




    def __call__(self):

        detection_results = []
        detection_labels = np.array([0]*YoloParams.NUM_CLASSES)

        num_annotations = 0 
        counter = 0

        for i in tqdm(range(len(self.generator.images)), desc='Batch Processed'):
            counter += 1

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




