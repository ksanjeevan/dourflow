import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, errno
import xml.etree.ElementTree as ET

import tensorflow as tf
import copy
import cv2



def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def compute_iou(bb_1, bb_2):

    xa0, ya0, xa1, ya1 = bb_1
    xb0, yb0, xb1, yb1 = bb_2

    intersec = (min([xa1, xb1]) - max([xa0, xb0]))*(min([ya1, yb1]) - max([ya0, yb0]))

    union = (xa1 - xa0)*(ya1 - ya0) + (xb1 - xb0)*(yb1 - yb0) - intersec

    return intersec / union


def benchmark_timings(data, path=''):

    fig = plt.figure(figsize=(10,15))
    ax = plt.gca()
    df = pd.DataFrame(data)
    df.plot(ax=ax, kind='area', subplots=True)
    plt.savefig(path + 'timings.png', format='png')
    plt.close()

    df2 = df.apply(lambda x: x/df['total'], axis=0)[['decode', 'prediction', 'prepro']]

    fig = plt.figure(figsize=(20,13))
    ax = fig.add_subplot(111)
    df2.plot(ax=ax)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
    plt.savefig(path + 'timings_combined.png', format='png')
    plt.close()




def space_to_depth_x2(x):
    """Thin wrapper for Tensorflow space_to_depth with block_size=2."""
    # Import currently required to make Lambda work.
    # See: https://github.com/fchollet/keras/issues/5088#issuecomment-273851273
    
    # the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)


    # tf.space_to_depth:
    # Input: [batch, height, width, depth]
    # Output: [batch, height/block_size, width/block_size, depth*block_size*block_size]
    # Example: [1,4,4,1] -> [1,2,2,4] or in this case [?,38,38,64] -> [?,19,19,256]
    # This operation is useful for resizing the activations between convolutions (but keeping all data),
    # e.g. instead of pooling. It is also useful for training purely convolutional models.

    # space_to_depth_x2 is just tf.space_to_depth wrapped with block_size=2


    # Example
    """
    input shape = (4,4,1)
    
    [
        [[1], [2], [3], [4]],
        [[5], [6], [7], [8]],
        [[9], [10], [11], [12]],
        [[13], [14], [15], [16]]
    ]
    
    is divided into the following chunks (block_size, block_size, channels):
    
    [[[1], [2]],       [[[3], [4]],
     [[5], [6]]]        [[7], [8]]]
    
    [[[9], [10],]      [[[11], [12]],
     [[13], [14]]]      [[15], [16]]]
     
     flatten each chunk to a single array:

    [[1, 2, 5, 6]],      [[3, 4, 7, 8]]
    [[9, 10, 13, 14]],    [[11, 12, 15, 16]]


    spatially rearrange chunks according to their initial position:
    
    [
        [[1, 2, 5, 6]], [[3, 4, 7, 8]],
        [[9 10, 13, 14]], [[11, 12, 15, 16]]
    ]
    
    output shape = (2,2,4)             
    """
    import tensorflow as tf
    return tf.space_to_depth(x, block_size=2)


def draw_boxes(image, info):
    image_h, image_w, _ = image.shape

    boxes, scores, labels = info
    color_mod = 255

    for i in range(len(boxes)):
        xmin = int(boxes[i][0]*image_w)
        ymin = int(boxes[i][1]*image_h)
        xmax = int(boxes[i][2]*image_w)
        ymax = int(boxes[i][3]*image_h)  

        if scores is None:
            #text = "%s"%(labels[i])
            text = ''
            color_mod = 0
        else:
            text = "%s (%.1f%%)"%(labels[i], 100*scores[i])

        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (color_mod,255,0), 2)

        cv2.putText(image, 
                    text, 
                    (xmin, ymin - 15), 
                    cv2.FONT_HERSHEY_COMPLEX, 
                    1e-3 * image_h, 
                    (color_mod,255,0), 1)
    return image          
        

def parse_annotation(ann_dir, img_dir, labels=[]):
    # from https://github.com/experiencor/keras-yolo2/blob/master/preprocessing.py
    all_imgs = []
    seen_labels = {}
    # go through annotations by sorted filename
    for ann in sorted(os.listdir(ann_dir)):
        img = {'object':[]}
        tree = ET.parse(os.path.join(ann_dir, ann))
        
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = os.path.join(img_dir, elem.text)
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1
                        
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]
                            
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs += [img]
    
    # all_imgs: [img1, img2, img3, ..]
    # 
    """
    img: 
        {'object' : [{'name': 'class1', 'xmin': , 'ymin': , 'xmax': , 'ymax': }, # object 1
                    {'name': 'class1', 'xmin': , 'ymin': , 'xmax': , 'ymax': },  # object 2
                    {'name': 'class2', 'xmin': , 'ymin': , 'xmax': , 'ymax': }]  # object 3
         'filename' : <where the image file is stored>,
         'width':, 
         'height': 
            }
    """
    # seen_labels: {'classname': count}
    return all_imgs, seen_labels







def setup_logging(logging_path='logs'):

    log_path = os.path.join(os.getcwd(),logging_path)
    mkdir_p(log_path)

    check_names = lambda y: y if y.isdigit() else -1
    get_ind = lambda x: int(check_names(x.split('_')[1]))
    
    run_counter = max(map(get_ind, os.listdir(log_path)), default=-1) + 1

    run_path = os.path.join(log_path, 'run_%s'%run_counter)
    mkdir_p(run_path)

    print('Logging set up, to monitor training run:\n'
        '\t\'tensorboard --logdir=%s\'\n'%run_path)

    return run_path


def handle_empty_indexing(arr, idx):
    if idx.size > 0:
        return arr[idx]
    return []



if __name__ == '__main__':

    imgs, cnts = parse_annotation('/home/kiran/Downloads/VOCdevkit/VOC2012/Annotations/','/home/kiran/Downloads/VOCdevkit/VOC2012/JPEGImages/')
    imgs, cnts = parse_annotation('/home/kiran/Downloads/VOCdevkit2007/VOC2007/Annotations/','/home/kiran/Downloads/VOCdevkit2007/VOC2007/JPEGImages/')













