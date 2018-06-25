
import pickle, argparse, json, cv2, os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from keras import backend as K
from keras.layers import Lambda
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop

from net.utils import parse_annotation, yolo_normalize, mkdir_p, \
setup_logging, TrainValTensorBoard, draw_boxes

from net.netparams import YoloParams
from net.netloss import YoloLoss
from net.neteval import YoloDataGenerator, YoloEvaluate
from net.netdecode import YoloOutProcess
from net.netarch import YoloArchitecture


class YoloInferenceModel(object):

    def __init__(self, model):
        self._yolo_out = YoloOutProcess()
        self._inf_model = self._extend_processing(model)

    def _extend_processing(self, model):
        output = Lambda(self._yolo_out, name='lambda_2')(model.output)
        return Model(model.input, output)


    def _prepro_single_image(self, image):
        image = cv2.resize(image, 
            (YoloParams.INPUT_SIZE, YoloParams.INPUT_SIZE))
        # yolo normalize
        image = yolo_normalize(image)
        image = image[:,:,::-1]
        # cv2 has the channel as bgr, revert to to rgb for Yolo Pass
        image = np.expand_dims(image, 0)

        return image

    def predict(self, image):

        image = self._prepro_single_image(image)
        output = self._inf_model.predict(image)[0]

        if output.size == 0:
            return [np.array([])]*4

        boxes = output[:,:4]
        scores = output[:,4]
        label_idxs = output[:,5].astype(int)

        labels = [YoloParams.CLASS_LABELS[l] for l in label_idxs]

        return boxes, scores, label_idxs, labels 



class YoloV2(object):


    def __init__(self):

        self.yolo_arch = YoloArchitecture()
        self.yolo_loss = YoloLoss()

        self.trained_model_name = YoloParams.OUT_MODEL_NAME
        self.debug_timings = True


    def run(self, **kwargs):

        self.model = self.yolo_arch.get_model(self.yolo_loss)
        
        if YoloParams.YOLO_MODE == 'train':
            self.training()

        else: 
            self.inf_model = YoloInferenceModel(self.model)

            if YoloParams.YOLO_MODE == 'inference':
                self.inference(YoloParams.PREDICT_IMAGE)

            elif YoloParams.YOLO_MODE == 'validate':
                self.validation()

            elif YoloParams.YOLO_MODE == 'video':
                self.video_inference(YoloParams.PREDICT_IMAGE)

        # Sometimes bug: https://github.com/tensorflow/tensorflow/issues/3388
        K.clear_session()


    def inference(self, path):

        flag = self.debug_timings

        if os.path.isdir(path):
            fnames = [os.path.join(path, f) for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))]

            out_fname_mod = '.png'
            out_path = os.path.join(path, 'out') 
            mkdir_p(out_path)

        else:   
            fnames = [path]
            out_fname_mod = '_pred.png'
            out_path = os.path.dirname(path)
            flag = False                  

        for f in tqdm(fnames, desc='Processing Batch'):

            image = cv2.imread(f)
            plt.figure(figsize=(10,10))

            boxes, scores, _, labels = self.inf_model.predict(image.copy())

            image = draw_boxes(image, (boxes, scores, labels))
            out_name =  os.path.join(out_path, os.path.basename(f).split('.')[0] + out_fname_mod)           
            cv2.imwrite(out_name, image)


    def video_inference(self, filename):

        cap = cv2.VideoCapture(filename)    
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        size = (video_width, video_height)
        fps = round(cap.get(cv2.CAP_PROP_FPS))

        #fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")

        writer = cv2.VideoWriter(filename.split('.')[0]+"_pred.mp4", fourcc, fps, size)
        
        for i in tqdm(range(video_len)):
        #while(cap.isOpened()):

            ret, frame = cap.read()
            #if ret==True:

            boxes, scores, _, labels = self.inf_model.predict(frame)

            frame_pred = draw_boxes(frame, (boxes, scores, labels))

            writer.write(frame_pred)

            #cv2.imshow('Output',frame_pred)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            #else:
            #    break

        cap.release()
        writer.release()
        cv2.destroyAllWindows()


    def video_inference2(self):
        # MAKEONE FOR WEBCAM, USE COMMENTED CODE FROM ABOVE
        video_reader = cv2.VideoCapture(video_inp)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                                       cv2.VideoWriter_fourcc(*'XVID'), 
                                       50.0, 
                                       (frame_w, frame_h))


        for i in tqdm(range(nb_frames)):
            ret, image = video_reader.read()

            boxes, scores, _, labels = self.inf_model.predict(image)

            image = draw_boxes(image, (boxes, scores, labels))
            
            video_writer.write(np.uint8(image))
            
        video_reader.release()
        video_writer.release()  



    def validation(self):

        valid_data, _ = parse_annotation(
            YoloParams.VALIDATION_ANN_PATH, YoloParams.VALIDATION_IMG_PATH)

        generator = YoloDataGenerator(valid_data, shuffle=True)

        yolo_eval = YoloEvaluate(generator=generator, model=self.inf_model)
        AP = yolo_eval()

        mAP_values = []
        for class_label, ap in AP.items():
            print("AP( %s ): %.3f"%(class_label, ap))
            mAP_values.append( ap )

        print('-------------------------------')
        print("mAP: %.3f"%(np.mean(mAP_values)))

        return AP
        


    def training(self):

        train_data, _ = parse_annotation(
            YoloParams.TRAIN_ANN_PATH, YoloParams.TRAIN_IMG_PATH)
        valid_data, _ = parse_annotation(
            YoloParams.VALIDATION_ANN_PATH, YoloParams.VALIDATION_IMG_PATH)

        train_gen = YoloDataGenerator(train_data, shuffle=True)
        valid_gen = YoloDataGenerator(valid_data, shuffle=True)


        early_stop = EarlyStopping(monitor='val_loss', 
                               min_delta=0.001, 
                               patience=3, 
                               mode='min', 
                               verbose=1)


        log_path = setup_logging()

        checkpoint_path = os.path.join(log_path, self.trained_model_name)
        checkpoint = ModelCheckpoint(
                                checkpoint_path, 
                                monitor='val_loss', 
                                verbose=1, 
                                save_best_only=True, 
                                mode='min', 
                                period=1)

        #tb_path = os.path.join(log_path, )
        tensorboard = TrainValTensorBoard(
                            log_dir=log_path,
                            histogram_freq=0,
                            write_graph=True,
                            write_images=False)

        optimizer = Adam(
                        lr=YoloParams.L_RATE, 
                        beta_1=0.9, 
                        beta_2=0.999, 
                        epsilon=1e-08, 
                        decay=0.0)

        self.model.compile(loss=self.yolo_loss, optimizer=optimizer) #, metrics=['accuracy'])

        self.model.fit_generator(
                        generator=train_gen,
                        steps_per_epoch=len(train_gen),
                        verbose=YoloParams.TRAIN_VERBOSE, 
                        validation_data=valid_gen,
                        validation_steps=len(valid_gen),
                        callbacks=[early_stop, checkpoint, tensorboard], 
                        epochs=YoloParams.NUM_EPOCHS,
                        max_queue_size=20)



if __name__ == '__main__':
    # Example: python3 yolov2.py data/birds.png -m models/coco/yolo_model_coco.h5 -c confs/config_coco.json -t 0.35

    var = YoloV2()
    var.run()






















