from keras.models import Model, load_model
from keras.layers import Reshape, Conv2D, Input, MaxPooling2D, BatchNormalization, Lambda
from keras.layers.advanced_activations import LeakyReLU

from keras.layers.merge import concatenate

import tensorflow as tf
import numpy as np
import pickle, argparse, json, os

from keras.utils.vis_utils import plot_model

from net.netparams import YoloParams
from net.netdecode import YoloOutProcess


class YoloArchitecture(object):

    def __init__(self):

        self.in_model_name = YoloParams.IN_MODEL
        self.plot_name = YoloParams.ARCH_FNAME

    def get_model(self, loss_func):

        yolo_model = self._load_yolo_model(loss_func)

        if YoloParams.YOLO_MODE == 'train':
            new_yolo_model = self._setup_transfer_learning(yolo_model)
            #new_name = self.tl_weights_name.split('.')[0] + '_rand.h5'
            #new_yolo_model.save_weights(new_name)

        elif YoloParams.YOLO_MODE in ['inference','validate','video']:
            new_yolo_model = yolo_model

        else:
            raise ValueError(
            'Please set \'--action\' to \'train\', \'validate\' or pass an image file/dir.')
            
        if self.plot_name:
            plot_model(new_yolo_model, to_file=self.plot_name, show_shapes=True)

        return new_yolo_model


    def _load_yolo_model(self, loss_func):
        # Error if not compiled with yolo_loss?
        if os.path.isfile(self.in_model_name):
            model = load_model(self.in_model_name,
                custom_objects={'yolo_loss': loss_func})
            return model
        else:
            raise ValueError('Need to load full model in order to do '
                'transfer learning. Run script again with desired TL '
                'config and weight file to generate model.')
            
        
    def weights_to_model(self, in_path, out_path):
        yolo_model = self._yolo_v2_architecture()

        try:
            yolo_model.load_weights(in_path)
        
        except IOError as e:
            print('File for pre-trained weights not found.')

        yolo_model.save(out_path)
        return yolo_model



    def _yolo_v2_architecture(self):
        # Parse from cfg!
        self.layer_counter = 0

        def space_to_depth_x2(x):
   
            import tensorflow as tf
            return tf.space_to_depth(x, block_size=2)

        
        def conv2D_bn_leaky(inp, filters, kernel_size=(3,3), strides=(1,1), maxpool=False):
            self.layer_counter += 1
            x = Conv2D(filters, kernel_size=kernel_size, strides=strides,
             padding='same', use_bias=False)(inp)

            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)
            if maxpool:
                return MaxPooling2D(pool_size=(2, 2))(x)
            return x

        input_image = Input(shape=(YoloParams.INPUT_SIZE, YoloParams.INPUT_SIZE, 3), name='input')

        # Layer 1
        x = conv2D_bn_leaky(input_image, 32, (3,3), (1,1), maxpool=True)

        # Layer 2
        x = conv2D_bn_leaky(x, 64, maxpool=True)

        # Layer 3
        x = conv2D_bn_leaky(x, 128)

        # Layer 4
        x = conv2D_bn_leaky(x, 64, kernel_size=(1,1))

        # Layer 5
        x = conv2D_bn_leaky(x, 128, maxpool=True)

        # Layer 6
        x = conv2D_bn_leaky(x, 256)

        # Layer 7
        x = conv2D_bn_leaky(x, 128, kernel_size=(1,1))

        # Layer 8
        x = conv2D_bn_leaky(x, 256, maxpool=True)

        # Layer 9
        x = conv2D_bn_leaky(x, 512)

        # Layer 10
        x = conv2D_bn_leaky(x, 256, kernel_size=(1,1))

        # Layer 11
        x = conv2D_bn_leaky(x, 512)

        # Layer 12
        x = conv2D_bn_leaky(x, 256, kernel_size=(1,1))

        # Layer 13
        x = conv2D_bn_leaky(x, 512)

        skip_connection = x
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 14
        x = conv2D_bn_leaky(x, 1024)

        # Layer 15
        x = conv2D_bn_leaky(x, 512, kernel_size=(1,1))
        # Layer 16
        x = conv2D_bn_leaky(x, 1024)

        # Layer 17
        x = conv2D_bn_leaky(x, 512, kernel_size=(1,1))
        # Layer 18
        x = conv2D_bn_leaky(x, 1024)

        # Layer 19
        x = conv2D_bn_leaky(x, 1024)

        # Layer 20
        x = conv2D_bn_leaky(x, 1024)

        # Layer 21
        skip_connection = conv2D_bn_leaky(skip_connection, 64, kernel_size=(1,1))
        skip_connection = Lambda(space_to_depth_x2)(skip_connection)
        x = concatenate([skip_connection, x])

        # Layer 22
        x = conv2D_bn_leaky(x, 1024)

        # Final Conv2D
        x = Conv2D(YoloParams.NUM_BOUNDING_BOXES * (4 + 1 + YoloParams.NUM_CLASSES), (1,1), 
            strides=(1,1), padding='same')(x)     


        output = Reshape((YoloParams.GRID_SIZE, YoloParams.GRID_SIZE, 
            YoloParams.NUM_BOUNDING_BOXES, 4 + 1 + YoloParams.NUM_CLASSES))(x)

        yolo_model = Model(input_image, output)

        return yolo_model



    def _setup_transfer_learning(self, yolo_model):

        new_yolo_model = self._yolo_v2_update(yolo_model)

        layer   = new_yolo_model.layers[-2] # the last convolutional layer
        weights = layer.get_weights()

        S2 = YoloParams.GRID_SIZE*YoloParams.GRID_SIZE
        new_kernel = np.random.normal(size=weights[0].shape)/S2
        new_bias   = np.random.normal(size=weights[1].shape)/S2

        layer.set_weights([new_kernel, new_bias])

        return new_yolo_model



    def _yolo_v2_update(self, old_yolo_model):

        x = Conv2D(YoloParams.NUM_BOUNDING_BOXES * (4 + 1 + YoloParams.NUM_CLASSES), (1,1), 
            strides=(1,1), padding='same', name='conv_23')(old_yolo_model.layers[-3].output)
        
        output = Reshape((YoloParams.GRID_SIZE, YoloParams.GRID_SIZE, 
            YoloParams.NUM_BOUNDING_BOXES, 4 + 1 + YoloParams.NUM_CLASSES))(x)

        yolo_model = Model(old_yolo_model.input, output)

        return yolo_model


def generate_model():

    yolo_arch = YoloArchitecture()
    
    d = os.path.dirname(YoloParams.WEIGHT_FILE)

    out_fname = os.path.join(d, 'model.h5')

    print('------------------------------------')
    print('Reading weights from: %s'%YoloParams.WEIGHT_FILE)
    print('Loading into YOLO V2 architecture and storing...')
    print('\n\n')
    yolo_arch.weights_to_model(YoloParams.WEIGHT_FILE, out_fname)
    print('\tModel saved: %s'%out_fname)
    print('\n\n------------------------------------')
    print('Done.')




