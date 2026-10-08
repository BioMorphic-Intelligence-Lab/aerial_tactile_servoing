"""
TacTip driver for ROS2, heavily inspired by RealSensor implementation
"""
import os
import cv2
import time

from .dependencies.image_transforms import process_image
from .dependencies.models import create_model
from .dependencies.utils import load_json_obj
from .dependencies.label_encoder import LabelEncoder
from .dependencies.label_encoder import BASE_MODEL_PATH
from .dependencies.labelled_model import LabelledModel


def resolve_model_dir(model_dir):
    """None -> the default model directory; an absolute path -> itself; a bare name ->
    share/tactip_ros2_driver/models/<name>."""
    if not model_dir:
        return BASE_MODEL_PATH
    if os.path.isabs(model_dir):
        return model_dir
    return os.path.join(os.path.dirname(BASE_MODEL_PATH), 'models', model_dir)


class TacTip:
    def __init__(self, source = 4, model_dir = None):
        """model_dir: absolute path to a model directory, or a bare name resolved against
        share/tactip_ros2_driver/models/. None keeps the single-model path so existing
        single-arm missions are unaffected. B1 and B2 are physically different sensors with their
        own force limits and image processing, so each instance must load its own model."""
        self.model_path = resolve_model_dir(model_dir)

        # Params are loaded before opening the camera so the exposure below can come from them.
        self.model_label_params = {}
        self.model_image_params = {}
        self.model_params = {}
        self.sensor_params = {}
        self.setup_params()

        # set up the camera
        self.source = source
        #self.cam = cv2. VideoCapture(self.source)
        self.cam = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
        #for _ in range(10):
        #    self.cam.read()

        # Ask for the frame size the model was trained on. OpenCV otherwise opens the camera in its
        # DEFAULT mode, which is not necessarily its maximum -> and a smaller frame would silently
        # make the bbox crop a no-op and put the circle mask in the wrong place.
        bbox = self.sensor_params.get('bbox')
        self.expected_size = (int(bbox[2]), int(bbox[3])) if bbox else None
        if self.expected_size is not None:
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.expected_size[0]))
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.expected_size[1]))

        # Match the exposure used during data collection. The adaptive threshold makes the model
        # largely brightness-invariant, but a fixed exposure also fixes exposure *time*, which
        # avoids motion blur and frame-rate variation from auto-exposure hunting under vibration.
        if self.sensor_params.get('exposure') is not None:
            self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual (V4L2), 3 = auto
            self.cam.set(cv2.CAP_PROP_EXPOSURE, float(self.sensor_params['exposure']))

        # Give V4L2 time to initialize video stream buffers
        time.sleep(0.5)

        # Flush warm-up frames safely without blocking indefinitely
        for _ in range(5):
            if self.cam.grab():
                self.cam.retrieve()
            else:
                time.sleep(0.05)

        # Confirm the camera actually gave us that size. A mismatch corrupts every prediction
        # without raising anything anywhere, so fail at startup rather than in flight.
        if self.expected_size is not None:
            ok, probe = self.cam.read()
            if not ok or probe is None:
                raise Exception(f"Camera /dev/video{self.source} opened but returned no frame.")
            h, w = probe.shape[:2]
            if (w, h) != self.expected_size:
                raise Exception(
                    f"Camera /dev/video{self.source} delivers {w}x{h}, but this model was trained at "
                    f"{self.expected_size[0]}x{self.expected_size[1]}. The bbox crop and the circle "
                    "mask are in pixels, so the model would see a different image than it was "
                    "trained on. Fix the camera mode (v4l2-ctl) before flying.")
            print(f"[TacTip] /dev/video{self.source} confirmed at {w}x{h}, "
                  f"mask radius {self.sensor_params.get('circle_mask_radius')} px, "
                  f"thresh {self.sensor_params.get('thresh')}")

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
            saved_model_dir=self.model_path,
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
        self.model_label_params = load_json_obj(os.path.join(self.model_path, 'model_label_params'))
        self.model_image_params = load_json_obj(os.path.join(self.model_path, 'model_image_params'))
        self.model_params = load_json_obj(os.path.join(self.model_path, 'model_params'))
        self.sensor_params = load_json_obj(os.path.join(self.model_path, 'processed_image_params'))

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