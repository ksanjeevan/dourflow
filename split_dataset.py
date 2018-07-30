
import os
from shutil import copy2
from random import shuffle

import argparse

from tqdm import tqdm
from net.utils import mkdir_p


argparser = argparse.ArgumentParser(
    description='Process training / validation data.')

argparser.add_argument(
    '-p',
    help='fraction of dataset for training split.',
    default=0.75)

argparser.add_argument(
    '--in_ann',
    help='path to input annotations file.')


argparser.add_argument(
    '--in_img',
    help='path to input images file.')


argparser.add_argument(
    '--output',
    help='path to output dir.')


#path = '/home/kiran/Downloads/VOCdevkit/VOC2012/JPEGImages/'


def sample_from_dir(paths, train_p):
    
    img_path, ann_path, out_path = paths

    imgs = os.listdir(img_path)
    
    total_num = len(imgs)
    train_num = int(len(imgs)*float(train_p))

    img_fmt = '.' + imgs[0].split('.')[1]
    fns = [im.split('.')[0] for im in imgs] 

    fn_train = fns[:train_num]
    fn_val = fns[train_num:]

    out_paths = []

    for subf, file_names in tqdm([('train', fn_train), ('valid', fn_val)], desc='Train/Val', leave=False):

        out_img_path = os.path.join(out_path, subf, 'imgs')
        out_ann_path = os.path.join(out_path, subf, 'anns')

        out_paths.extend([out_img_path, out_ann_path])        

        mkdir_p(out_img_path)
        mkdir_p(out_ann_path)

        for f in tqdm(file_names, desc='Files', leave=False):
            
            img_fnm = os.path.join(img_path, f + img_fmt)#'/home/kiran/Downloads/VOCdevkit/VOC2012/JPEGImages/' + f + '.jpg'
            ann_fnm = os.path.join(ann_path, f + '.xml')#'/home/kiran/Downloads/VOCdevkit/VOC2012/Annotations/' + f + '.xml' 

            if os.path.isfile(img_fnm) and os.path.isfile(ann_fnm):
                copy2(ann_fnm, out_ann_path)
                copy2(img_fnm, out_img_path)
            else:
                print(img_fnm)
                pritn(ann_fnm)
                raise ValueError('\nAL TANTO\n')

    print('------------------------------------')
    print('Train / Validation Splits stored.\nPaths:')
    print('\n\tTrain:')
    print('\n\t\tImages:      %s'%out_paths[0])
    print('\n\t\tAnnotations: %s'%out_paths[1])
    
    print('\n\t Validate:')
    print('\n\t\tImages:      %s'%out_paths[2])
    print('\n\t\tAnnotations: %s'%out_paths[3])

    print('------------------------------------')


if __name__ == '__main__':
    # python3 split_dataset.py -p 0.75 --in_ann ~/Downloads/VOCdevkit/VOC2012/Annotations/ --in_img ~/Downloads/VOCdevkit/VOC2012/JPEGImages/ --output ~/Documents/DATA/VOC

    args = argparser.parse_args()

    paths = (args.in_img, args.in_ann, args.output)
    sample_from_dir(paths, args.p)