import os
import cv2
from base_camera import BaseCamera

import cv2
import torch
import numpy as np
import os
from torchvision.transforms import ToTensor
from utils.utils import ImageToTensor, xywh2xyxy, get_most_confident_bbox, transform_bbox_coords


PATH_TO_DETECTION_MODEL = 'log/detection/faced_model_lite'
PATH_TO_CLASSIFICATION_MODEL = 'log/classification/mini_xception'
EMOTIONS_LIST = ['Anger', 'Happy', 'Neutral', 'Surprise']
DETECTION_SIZE = (288, 288)
CLASSIFICATION_SIZE = (64, 64)
DETECTION_THRESHOLD = 0.3
GRID_SIZE = 9


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
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, image = camera.read()
            orig_shape = image.shape[:2]  # (H, W)

            with torch.no_grad():
                detection_image = cv2.resize(image, DETECTION_SIZE)
                detection_image = ImageToTensor()(detection_image)
                detection_image = detection_image.unsqueeze(0)
                output = detection_model(detection_image)  # Prediction
                x, y, z = get_most_confident_bbox(output, 2)
                pred_xywh = transform_bbox_coords(output, x, y, z, DETECTION_SIZE, GRID_SIZE)
                pred_xyxy = xywh2xyxy(pred_xywh)

                if output[0, z + 4, x, y].item() > DETECTION_THRESHOLD:  # prediction confidence threshold

                    bbox_l_y = int((pred_xyxy[1]) * (orig_shape[0] / DETECTION_SIZE[1]))  # Transform bbox coords
                    bbox_r_y = int((pred_xyxy[3]) * (
                                orig_shape[0] / DETECTION_SIZE[1]))  # correspondingly to DETECTION_SHAPE -> orig_shape

                    bbox_l_x = int((pred_xyxy[0]) * (orig_shape[1] / DETECTION_SIZE[0]))
                    bbox_r_x = int((pred_xyxy[2]) * (orig_shape[1] / DETECTION_SIZE[0]))

                    bbox_x_c = (bbox_l_x + bbox_r_x) // 2
                    bbox_h = bbox_r_y - bbox_l_y

                    bbox_l_x = bbox_x_c - bbox_h // 2  # Make bbox square with sides equal to bbox_h
                    bbox_r_x = bbox_x_c + bbox_h // 2

                    bbox_l_y = np.clip(bbox_l_y, 0, orig_shape[0])  # clip coordinates which limit image borders
                    bbox_r_y = np.clip(bbox_r_y, 0, orig_shape[0])
                    bbox_l_x = np.clip(bbox_l_x, 0, orig_shape[1])
                    bbox_r_x = np.clip(bbox_r_x, 0, orig_shape[1])

                    # Converting image to format and shape required by recognition model
                    cl_image = image[bbox_l_y:bbox_r_y, bbox_l_x:bbox_r_x, :]

                    cl_image = cv2.resize(cl_image, CLASSIFICATION_SIZE, CLASSIFICATION_SIZE)
                    cl_image = cv2.cvtColor(cl_image, cv2.COLOR_BGR2GRAY)
                    cl_image = ToTensor()(cl_image).unsqueeze(0)

                    # Paint bbox and emotion prediction
                    pred_emo = EMOTIONS_LIST[classification_model(cl_image).argmax(dim=1).item()]
                    image = cv2.rectangle(image, (bbox_l_x, bbox_l_y), (bbox_r_x, bbox_r_y), color=(0, 255, 0),
                                          thickness=2)
                    image = cv2.putText(image,
                                        pred_emo,
                                        (bbox_l_x, bbox_l_y + 10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        color=(0, 0, 255),
                                        fontScale=0.5,
                                        thickness=2)

                    yield cv2.imencode('.jpg', image)[1].tobytes()

                else:
                    yield cv2.imencode('.jpg', image)[1].tobytes()
