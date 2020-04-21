from base_camera import BaseCamera
from pipeline import stream_prediction
import cv2
import torch
import os
import configparser


config = configparser.ConfigParser()
config.read('config.ini')


PATH_TO_DETECTION_MODEL = config['path_to_models']['detection_model']
PATH_TO_CLASSIFICATION_MODEL = config['path_to_models']['recognition_model']

torch.backends.quantized.engine = 'qnnpack'
detection_model = torch.jit.load(os.path.join(PATH_TO_DETECTION_MODEL, 'model_quantized.pt'))
detection_model.to('cpu').eval()

classification_model = torch.jit.load(os.path.join(PATH_TO_CLASSIFICATION_MODEL, 'model_quantized.pt'))
classification_model.to('cpu').eval()


class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, image = camera.read()
            image = stream_prediction(image, detection_model, classification_model)
            yield image

