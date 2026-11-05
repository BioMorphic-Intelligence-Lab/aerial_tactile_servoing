"""
TacTip driver for ROS2, heavily inspired by RealSensor implementation
"""
import os
import cv2
import time
import threading
from dataclasses import dataclass

import torch

from .dependencies.image_transforms import process_image
from .dependencies.models import create_model
from .dependencies.utils import load_json_obj
from .dependencies.label_encoder import LabelEncoder
from .dependencies.label_encoder import BASE_MODEL_PATH
from .dependencies.labelled_model import LabelledModel
from .fast_ssim import structural_similarity as ssim


@dataclass(frozen=True)
class Frame:
    """One captured frame, already processed and scored by the capture thread."""
    image: object          # processed image (model input), from process_image()
    ssim: float            # SSIM against this sensor's contact reference
    seq: int               # increments per captured frame; gaps mean frames were overwritten
    stamp: object          # clock() at capture
    mono: float            # time.monotonic() at capture, for frame-age measurements
    t_process: float       # seconds spent in process_image
    t_ssim: float          # seconds spent in SSIM
    ref_refreshed: bool    # this frame became the new SSIM reference


class TacTip:
    # Startup probe: how hard to try for a first good frame before declaring the camera dead.
    PROBE_MAX_ATTEMPTS = 15
    PROBE_RETRY_DELAY = 0.05  # seconds
    # Synchronous read (startup only): a dropped frame is transient, so retry a few times.
    READ_MAX_ATTEMPTS = 3
    # Grabber thread: back-off after a failed read so a dead camera does not spin a core.
    GRAB_FAIL_BACKOFF = 0.005  # seconds

    def __init__(self, source, model_name, clock=time.time, ssim_dims=None):
        """model_dir: absolute path to a model directory, or a bare name resolved against
        share/tactip_ros2_driver/models/. None keeps the single-model path so existing
        single-arm missions are unaffected. B1 and B2 are physically different sensors with their
        own force limits and image processing, so each instance must load its own model.

        clock: callable returning the capture timestamp attached to each grabbed frame. The node
        passes its ROS clock's now() so message stamps respect use_sim_time.

        ssim_dims: (width, height) the processed image is downsampled to before SSIM, or None for
        full resolution. Full-frame SSIM is memory-bound (~15 full-size float32 temporaries) and
        cost ~16 ms per frame on board; 320x240 is an exact 2x reduction (resize ~0.01 ms) that
        cuts SSIM ~9x. Only the SSIM sees the downsampled image; the model input is unchanged.
        SSIM scores depend on resolution, so contact thresholds are tied to this setting."""
        self.model_path = os.path.join(BASE_MODEL_PATH,model_name)
        self._clock = clock

        # Latest-frame slot, filled by the capture thread (see start()). Guarded by self._cond.
        self._cond = threading.Condition()
        self._frame = None         # latest Frame
        self._seq = 0              # increments on every successfully captured frame
        self._mono = None          # time.monotonic() of the latest frame
        self.failed_reads = 0      # cumulative failed cam.read() calls in the capture thread
        self.process_errors = 0    # cumulative exceptions in process_image / SSIM
        self.last_error = None
        self._stop_event = threading.Event()
        self._thread = None

        # SSIM contact reference (processed image). Owned by the capture thread once running;
        # other threads only request a refresh.
        self._ssim_ref = None
        self._ref_refresh = threading.Event()
        self.ssim_dims = tuple(ssim_dims) if ssim_dims else None

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

    # ------------------------------------------------------------------------------------------
    # Non-blocking capture.
    #
    # cam.read() blocks until the camera delivers the next frame (~58 ms at the cameras' fixed
    # ~17 fps). Reading two cameras one after another from the processing loop meant any extra
    # work pushed the second read past a frame boundary, halving the rate. Instead, a capture
    # thread per camera reads continuously and publishes only the newest frame; consumers take it
    # without waiting.
    #
    # The capture thread also runs process_image and SSIM (~25 ms together on 640x480). Both
    # release the GIL, so the cameras' preprocessing runs in parallel and the single inference
    # worker only predicts and publishes. Done serially in the worker instead, two cameras' worth
    # of preprocessing saturated it below the camera rate. As long as read + preprocessing stays
    # under one frame period, the thread is back in read() before the next frame is due, so no
    # frame queues up in the driver.
    # ------------------------------------------------------------------------------------------

    def set_ssim_reference(self, processed_img):
        """Set the SSIM contact reference directly (before start())."""
        if self.running:
            raise RuntimeError("Use request_ssim_reference() while the capture thread is running.")
        self._ssim_ref = self._ssim_view(processed_img)

    def _ssim_view(self, processed_img):
        """The image SSIM is computed on: the processed image, downsampled to ssim_dims if set.
        Used for both the reference and live frames so their sizes always match."""
        img = processed_img.squeeze()
        if self.ssim_dims is None:
            return img
        return cv2.resize(img, self.ssim_dims, interpolation=cv2.INTER_AREA)

    def request_ssim_reference(self):
        """Ask the capture thread to use its next frame as the SSIM reference (thread-safe)."""
        self._ref_refresh.set()

    def start(self, cond=None):
        """Start the capture thread. Pass a shared threading.Condition to be woken whenever this
        (or any other sensor sharing it) has a new frame; it also guards the frame slot."""
        if self._thread is not None:
            return
        if cond is not None:
            self._cond = cond
        # Ask the driver to keep at most one queued frame. These cameras may ignore it, but the
        # thread draining continuously makes the driver queue irrelevant either way.
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Until the first frame arrives, frame_age() counts from now, so a camera that never
        # delivers is reported as stale after the timeout rather than immediately.
        if self._mono is None:
            self._mono = time.monotonic()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name=f'tactip_capture_video{self.source}', daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the capture thread and release the camera."""
        self._stop_event.set()
        if self._thread is not None:
            # read() returns within one frame period, so this join is short.
            self._thread.join(timeout=1.0)
            self._thread = None
        self.cam.release()

    @property
    def running(self):
        return self._thread is not None

    def _capture_loop(self):
        while not self._stop_event.is_set():
            ok, img = self.cam.read()
            if not ok or img is None:
                # Transient drop (USB jitter, vibration): keep going. The consumer notices a
                # dead camera through frame_age().
                self.failed_reads += 1
                time.sleep(self.GRAB_FAIL_BACKOFF)
                continue
            stamp = self._clock()
            mono = time.monotonic()
            try:
                t0 = time.perf_counter()
                processed = process_image(img, **self.sensor_params)
                t1 = time.perf_counter()
                view = self._ssim_view(processed)
                refreshed = self._ref_refresh.is_set() or self._ssim_ref is None
                if refreshed:
                    self._ref_refresh.clear()
                    self._ssim_ref = view
                score = ssim(self._ssim_ref, view)
                t2 = time.perf_counter()
            except Exception as e:
                # A bad frame must not kill the thread (the sensor would go silent for good).
                self.process_errors += 1
                self.last_error = repr(e)
                continue
            with self._cond:
                self._seq += 1
                self._frame = Frame(processed, score, self._seq, stamp, mono,
                                    t1 - t0, t2 - t1, refreshed)
                self._mono = mono
                self._cond.notify_all()

    @property
    def seq(self):
        """Sequence number of the newest frame (0 = none yet)."""
        return self._seq

    def latest(self):
        """Newest Frame, or None if none yet. Never blocks on the camera."""
        with self._cond:
            return self._frame

    def frame_age(self):
        """Seconds since the newest frame was captured (or since start() if none yet; inf
        before start())."""
        mono = self._mono
        return float('inf') if mono is None else time.monotonic() - mono

    def read(self):
        """Blocking read, for use before start() only (e.g. the startup SSIM reference).
        Retries a few dropped frames; returns None if every attempt failed."""
        if self.running:
            raise RuntimeError("TacTip.read() called while the grabber thread owns the camera; "
                               "use latest() instead.")
        for _ in range(self.READ_MAX_ATTEMPTS):
            ok, img = self.cam.read()
            if ok and img is not None:
                return img
        return None

    def process(self, img=None, raw_outfile=None, proc_outfile=None):
        """Apply the sensor's image processing to a raw frame. Without img, does a blocking
        read() first (startup only). Returns None if no frame could be read."""
        if img is None:
            img = self.read()
            if img is None:
                return None
        if raw_outfile:
            cv2.imwrite(raw_outfile, img)
        img = process_image(img, **self.sensor_params)
        if proc_outfile:
            cv2.imwrite(proc_outfile, img)
        return img

    def predict(self, processed_img):
        # inference_mode skips autograd bookkeeping; model is already in eval().
        with torch.inference_mode():
            return self.pose_model.predict(processed_img)

    def get_measurement(self):
        img = self.process()
        return None if img is None else self.predict(img)