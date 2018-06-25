
from net.netarch import generate_model
from net.netparams import YoloParams
from yolov2 import YoloV2, YoloInferenceModel
import os


# Add CPU option
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == '__main__':

    if YoloParams.WEIGHT_FILE:
        generate_model()

    else:
        YoloV2().run()
        


