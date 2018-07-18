
from net.netarch import generate_model
from net.netparams import YoloParams
from yolov2 import YoloV2, YoloInferenceModel
import os

from kmeans_anchors import gen_anchors


# Add CPU option
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == '__main__':

    if YoloParams.WEIGHT_FILE:
        generate_model()
    elif YoloParams.GEN_ANCHORS_PATH:
        gen_anchors(YoloParams.GEN_ANCHORS_PATH)
    else:
        YoloV2().run()
        


