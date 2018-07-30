"""
Data generator augmentation.
"""

from .netparams import YoloParams
from .utils import compute_iou

import numpy as np
import cv2, os
import keras
import copy

PERC_LIMIT = 0.2
HSV_FACT = 1.5

class YoloDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, images, shuffle=True, augment=False):

        self.images = self._prune_ann_labels(images)
        self.input_size = YoloParams.INPUT_SIZE
        self.anchors = YoloParams.anchors

        self.generator = None

        self.batch_size = YoloParams.BATCH_SIZE
        
        self.shuffle = shuffle
        self.perc = PERC_LIMIT
        self.hsvf = HSV_FACT
        self.augment = augment

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
        
        for j, inst in enumerate(batch_images):
            
            img_raw, new_inst = data_augmentation(inst, self.perc, self.hsvf, self.augment)

            h_factor_resize = img_raw.shape[0] / self.input_size
            w_factor_resize = img_raw.shape[1] / self.input_size 

            img = cv2.resize(img_raw, (self.input_size, self.input_size))

            for label in new_inst['object']:

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
            
            x_batch[j] = img / 255.

        ############################################################
        # x_batch -> list of input images
        # y_batch -> list of network ouput gt values for each image
        ############################################################
        return x_batch, y_batch




def _scale_translation(inst, fact):

    height, width = inst['height'], inst['width']
    # what % from the increased height will 
    # contribute to the offset position
    pos_fact = fact * np.random.rand()
    off_x = int(round(pos_fact * width))
    off_y = int(round(pos_fact * height))

    fields = {
    'xmin':(off_x, width), 
    'xmax':(off_x, width), 
    'ymin':(off_y, height),
    'ymax':(off_y, height)}

    final_objs = []
    for label in inst['object']:

        for coord,v in fields.items():
            offset, lim = v
            label[coord] = label[coord] * (1+fact)
            label[coord] = max(min(int(round(label[coord]-offset)), lim),0)

        # if a an object was left out of the transform don't include it
        # for amin, amax in [('xmin', 'xmax'),('ymin','ymax')]:
        xcond = label['xmax'] - label['xmin'] > 5 
        ycond = label['ymax'] - label['ymin'] > 5

        if xcond and ycond: 
            final_objs.append(label)

    return off_x, off_y, final_objs

def _exposure_saturation(img, hsvf):
    sfact = np.random.uniform(1,hsvf)
    vfact = np.random.uniform(1,hsvf)
    
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    s = (hsv[...,1]*sfact).astype(np.int)
    v = (hsv[...,2]*vfact).astype(np.int)
    
    hsv[...,1] = np.where(s < 255, s, 255)
    hsv[...,2] = np.where(v < 255, v, 255)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)



def data_augmentation(inst, perc, hsvf, augment):
    img_raw = cv2.imread(inst['filename']).copy()
    new_inst = copy.deepcopy(inst)
    
    if not augment: return img_raw, new_inst


    fact = perc * np.random.rand()

    off_x, off_y, objs = _scale_translation(new_inst, fact=fact)

    if len(objs) == 0:
        return img_raw, inst        

    new_inst['object'] = objs

    img_resized = cv2.resize(img_raw, (0,0), fx=(1+fact), fy=(1+fact))
    # preserve original size after scaling & translating
    img_off = img_resized[off_y:off_y+img_raw.shape[0], off_x:off_x+img_raw.shape[1]]

    img_final = _exposure_saturation(img_off, hsvf=hsvf)

    return img_final, new_inst