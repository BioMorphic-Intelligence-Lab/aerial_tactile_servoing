"""
TacTip driver for ROS2, heavily inspired by RealSensor implementation
"""
import os
import cv2

from .dependencies.image_transforms import process_image
from .dependencies.models import create_model
from .dependencies.utils import load_json_obj
from .dependencies.label_encoder import LabelEncoder
from .dependencies.label_encoder import BASE_MODEL_PATH
from .dependencies.labelled_model import LabelledModel

class TacTip:
    def __init__(self, source = 4):
        # set up the camera
        self.source = source
        self.cam = cv2. VideoCapture(self.source)
        for _ in range(10):
            self.cam.read()
        
        # initialize params
        self.model_label_params = {}
        self.model_image_params = {}
        self.model_params = {}
        self.sensor_params = {}
        self.setup_params()

        if self.model_label_params == {}:
            raise Exception("Model label params not found")
        if self.model_image_params == {}:
            raise Exception("Model image params not found")
        if self.model_params == {}:
            raise Exception("Model params not found")
        if self.sensor_params == {}:
            raise Exception("Sensor params not found")

        # create the label encoder/decoder
        label_encoder = LabelEncoder(self.model_label_params, device='cpu')
        
        # setup the model
        model = create_model(
            in_dim=self.model_image_params["image_processing"]["dims"],
            in_channels=1,
            out_dim=label_encoder.out_dim,
            model_params=self.model_params,
            saved_model_dir=BASE_MODEL_PATH,
            device='cpu'
        )
        model.eval()

        self.pose_model = LabelledModel(
            model,
            self.model_image_params['image_processing'],
            label_encoder,
            device='cpu'
        )

    def setup_params(self):
        self.model_label_params = load_json_obj(os.path.join(BASE_MODEL_PATH, 'model_label_params'))
        self.model_image_params = load_json_obj(os.path.join(BASE_MODEL_PATH, 'model_image_params'))
        self.model_params = load_json_obj(os.path.join(BASE_MODEL_PATH, 'model_params'))
        self.sensor_params = load_json_obj(os.path.join(BASE_MODEL_PATH, 'processed_image_params'))

    def read(self):
        _, img = self.cam.read()
        return img

    def process(self, raw_outfile=None, proc_outfile=None):
        img = self.read()
        if raw_outfile:
            cv2.imwrite(raw_outfile, img)
        img = process_image(img, **self.sensor_params)
        if proc_outfile:
            cv2.imwrite(proc_outfile, img)
        return img
    
    def predict(self, processed_img):
        return self.pose_model.predict(processed_img)

    def get_measurement(self):
        img = self.process()
        return self.predict(img)