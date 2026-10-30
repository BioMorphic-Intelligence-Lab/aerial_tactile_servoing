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


class TacTip:
    # Startup probe: how hard to try for a first good frame before declaring the camera dead.
    PROBE_MAX_ATTEMPTS = 15
    PROBE_RETRY_DELAY = 0.05  # seconds
    # Runtime read: a dropped frame is transient, so retry a few times within a single cycle.
    READ_MAX_ATTEMPTS = 3

    def __init__(self, source, model_name):
        """model_dir: absolute path to a model directory, or a bare name resolved against
        share/tactip_ros2_driver/models/. None keeps the single-model path so existing
        single-arm missions are unaffected. B1 and B2 are physically different sensors with their
        own force limits and image processing, so each instance must load its own model."""
        self.model_path = os.path.join(BASE_MODEL_PATH,model_name)

        # Params are loaded before opening the camera so the exposure below can come from them.
        self.model_label_params = {}
        self.model_image_params = {}
        self.model_params = {}
        self.sensor_params = {}
        self.setup_params()

        # set up the camera
        self.source = source
        self.cam = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
        # VideoCapture never raises on a failed open; it just returns a closed object. Check it here
        # so the error names the real cause (device could not be opened) instead of surfacing later
        # as a misleading "returned no frame" from the probe below.
        if not self.cam.isOpened():
            raise RuntimeError(
                f"Could not open camera /dev/video{self.source}. "
                f"Check it is connected and is a capture device (`v4l2-ctl --list-devices`), "
                f"that the index in `sources` is correct, and that no other process is using it "
                f"(`fuser /dev/video{self.source}`)."
            )
        for _ in range(10):
            self.cam.read()

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
            # A freshly opened USB camera (especially two sharing a bus) routinely drops the first
            # frames, so a single failed read is not evidence the device is dead. Retry the probe a
            # number of times and only fail if we never get a frame -- this keeps the fail-fast
            # guarantee for a genuinely absent/busy camera without aborting a healthy one on one
            # unlucky frame.
            ok, probe = False, None
            for _ in range(self.PROBE_MAX_ATTEMPTS):
                ok, probe = self.cam.read()
                if ok and probe is not None:
                    break
                time.sleep(self.PROBE_RETRY_DELAY)
            if not ok or probe is None:
                raise Exception(
                    f"Camera /dev/video{self.source} opened but returned no frame after "
                    f"{self.PROBE_MAX_ATTEMPTS} attempts (last ok: {ok})."
                )
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
        else:
            print(f"[TacTip] /dev/video{self.source} bbox {bbox}")
            print(self.sensor_params)

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
        # A dropped frame mid-flight is transient (USB jitter, vibration), not a reason to bring
        # the node down, so retry a few times within the cycle. Return None only if every attempt
        # failed; the caller decides what to do with a missing frame.
        for _ in range(self.READ_MAX_ATTEMPTS):
            ok, img = self.cam.read()
            if ok and img is not None:
                return img
        return None

    def process(self, raw_outfile=None, proc_outfile=None):
        img = self.read()
        if img is None:
            # Signal "no frame this cycle" instead of feeding None into process_image (which would
            # raise and crash the timer callback). The caller skips this cycle.
            return None
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