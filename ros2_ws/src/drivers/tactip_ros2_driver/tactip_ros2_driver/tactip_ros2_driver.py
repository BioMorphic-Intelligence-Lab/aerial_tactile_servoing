import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
import threading
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import TwistStamped, TransformStamped, WrenchStamped
from std_msgs.msg import Float64, Int8
from tf2_ros import TransformBroadcaster

from std_srvs.srv import SetBool

from .tactip import TacTip

# ---------------------------------------------------------------------------------------------
# Pipeline: event-driven, one inference per camera frame.
#
#   capture thread per camera (TacTip.start)   cam.read() -> process_image -> SSIM
#                                              -> newest-frame slot -> notify
#   inference worker (this node)               wake on any new frame -> predict (if contact)
#                                              -> publish, for each sensor with a new frame
#   main thread                                rclpy.spin (service, parameters)
#
# The feedback rate is bound by the cameras (fixed ~17 fps, no usable V4L2 controls), not by the
# model. Blocking reads in a timer used to cap it at ~7 Hz: a frame-period overrun made each read
# wait for the next frame. Capture threads decouple capture (and per-camera preprocessing) from
# inference, and the worker runs exactly once per new frame, so no frame is predicted twice and
# none waits on a poll timer. If the worker falls behind, older frames are overwritten (latest
# wins, no queue).
#
# Inference stays serial in a single process on purpose: two PyTorch processes oversubscribe the
# onboard 8 cores and predict goes 15 -> 64 ms. torch_threads bounds intra-op threads for the same
# reason, leaving cores for the grabbers, SSIM and rclpy.
# ---------------------------------------------------------------------------------------------


class _TimingStats:
    """Per-sensor accumulators for the timing log, reset every log period."""

    def __init__(self):
        self.reset(seq=0, failed_reads=0)

    def reset(self, seq, failed_reads):
        self.seq0 = seq
        self.failed_reads0 = failed_reads
        self.processed = 0
        self.predicted = 0
        self.cap_skipped = 0
        self.t_proc = 0.0
        self.t_ssim = 0.0
        self.t_pred = 0.0
        self.t_prep = 0.0    # inside predict: image prep + tensor conversion
        self.t_fwd = 0.0     # inside predict: model forward pass
        self.t_dec = 0.0     # inside predict: label decoding
        self.t_pub = 0.0
        self.age_sum = 0.0
        self.age_max = 0.0


class TactipDriver(Node):
    # No new frame from a sensor for this long -> warn (camera stalled/unplugged).
    STALE_TIMEOUT = 0.5  # seconds
    STALE_WARN_INTERVAL = 1.0  # seconds, per sensor
    TIMING_LOG_PERIOD = 5.0  # seconds
    # Fake-data publish rate when 'frequency' is 0 (roughly the real cameras' rate).
    FAKE_DEFAULT_RATE = 17.0  # Hz

    def __init__(self):
        super().__init__('tactip_driver')

        # Parameters
        # Sensor-specific
        self.declare_parameter('sources', [0])
        self.declare_parameter('dimensions', [8])
        self.declare_parameter('ssim_contact_thresholds', [0.9])
        self.declare_parameter('model_names', ['model_name'])

        # Node-level
        # Max publish rate per sensor [Hz]. 0 = publish every camera frame.
        self.declare_parameter('frequency', 0.)
        self.declare_parameter('fake_data', False)
        self.declare_parameter('verbose', True)
        # Periodic timing log (rates, per-stage times, frame age) every TIMING_LOG_PERIOD s
        self.declare_parameter('test_model_time', False)
        self.declare_parameter('torch_threads', 4)  # 0 = leave PyTorch's default (all cores)
        self.declare_parameter('zero_when_no_contact', True)

        # Grabbing the parameters
        self.fake_data = self.get_parameter('fake_data').get_parameter_value().bool_value
        self.ssim_thresholds = self.get_parameter('ssim_contact_thresholds').get_parameter_value().double_array_value
        self.model_names = self.get_parameter('model_names').get_parameter_value().string_array_value
        self.dimensions = self.get_parameter('dimensions').get_parameter_value().integer_array_value
        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value
        self.verbose = self.get_parameter('verbose').get_parameter_value().bool_value
        self.timing_log = self.get_parameter('test_model_time').get_parameter_value().bool_value
        self.zero_when_no_contact = self.get_parameter('zero_when_no_contact').get_parameter_value().bool_value
        torch_threads = self.get_parameter('torch_threads').get_parameter_value().integer_value
        sources = list(self.get_parameter('sources').get_parameter_value().integer_array_value)

        if torch_threads > 0:
            torch.set_num_threads(torch_threads)

        # Initializing lists
        self.sensors = []
        self.sensor_frames = []
        self.contact_frames = []
        self.pose_publishers_ = []
        self.wrench_publishers_ = []
        self.ssim_publishers_ = []
        self.contact_publishers_ = []

        # Other node data
        self.fake_channels = None
        self._worker = None
        self._stop = threading.Event()

        # Node feedback
        self.get_logger().info(f"Tactip driver initialized with dimensions: {list(self.dimensions)} and ssim thresholds: {self.ssim_thresholds}")

        # Initialize the sensor(s)
        if not self.fake_data:
            # Initialize TacTip sensors
            for i, s in enumerate(sources):
                try:
                    self.sensors.append(TacTip(s, self.model_names[i], clock=self.get_clock().now))
                except Exception as e:
                    # A missing/busy camera is not recoverable mid-mission, so fail fast at startup
                    # with one clear, node-tagged line naming the sensor instead of a raw traceback.
                    self.get_logger().fatal(
                        f"Failed to initialize TacTip sensor {i} "
                        f"(model '{self.model_names[i]}', /dev/video{s}): {e}"
                    )
                    raise SystemExit(1)
                self.get_logger().info(f"Start sensor {i}, reading from /dev/video{s}" )
                self.get_logger().info(f"With model {self.model_names[i]}")
                self.get_logger().info(f"Sensor processing params: {self.sensors[-1].sensor_params}")

                # Reference SSIM image, read synchronously before the capture thread starts. The camera is
                # warm here (the probe just succeeded), but a dropped frame would make process()
                # return None; fail fast with a clear message rather than a cryptic NoneType error,
                # since every later frame needs this reference.
                ref_image = self.sensors[-1].process()
                if ref_image is None:
                    self.get_logger().fatal(
                        f"Failed to capture reference SSIM image for TacTip sensor {i} "
                        f"(model '{self.model_names[i]}', /dev/video{s}): no frame returned."
                    )
                    raise SystemExit(1)
                self.sensors[-1].set_ssim_reference(ref_image)
            # Create publishers
            self.create_publishers_(sources)
        else:
            self.fake_channels = len(sources)
            self.get_logger().info(f"Using fake data mode on {self.fake_channels} channels. No camera input.")
            self.create_publishers_(range(self.fake_channels))

        # Broadcaster TF
        self.broadcaster_tf = TransformBroadcaster(self)

        # Start service to refresh SSIM of all images
        self.srv_get_ssim_ref = self.create_service(
            SetBool,                    # service type
            'set_ssim_ref',             # service name
            self.get_ssim_ref_callback  # callback
        )

        if self.fake_data:
            rate = self.frequency if self.frequency > 0 else self.FAKE_DEFAULT_RATE
            self.timer = self.create_timer(1.0 / rate, self.fake_timer_callback)
            return

        # Start capture + inference
        n = len(self.sensors)
        self.min_period = 1.0 / self.frequency if self.frequency > 0 else 0.0
        self._last_seq = [0] * n
        self._next_due = [0.0] * n
        self._last_stale_warn = [0.0] * n
        self._process_errors_seen = [0] * n
        self._timing = [_TimingStats() for _ in range(n)]
        self._worker_busy = 0.0
        self._timing_t0 = time.monotonic()

        self._cond = threading.Condition()
        for sensor in self.sensors:
            sensor.start(self._cond)
        self._worker = threading.Thread(target=self._inference_loop, name='tactip_inference', daemon=True)
        self._worker.start()
        rate_txt = f"capped at {self.frequency:.1f} Hz" if self.min_period > 0 else "every camera frame"
        self.get_logger().info(f"Capture started on {n} sensor(s); publishing {rate_txt}.")

    # -----------------------------------------------------------------------------------------
    # Inference worker
    # -----------------------------------------------------------------------------------------

    def _has_new_frame(self):
        return any(s.seq > self._last_seq[i] for i, s in enumerate(self.sensors))

    def _inference_loop(self):
        while not self._stop.is_set():
            with self._cond:
                self._cond.wait_for(lambda: self._stop.is_set() or self._has_new_frame(),
                                    timeout=self.STALE_TIMEOUT)
                if self._stop.is_set():
                    break
                batch = [(i, s.latest()) for i, s in enumerate(self.sensors)
                         if s.seq > self._last_seq[i]]

            busy_start = time.perf_counter()
            # Oldest frame first so one sensor cannot starve the other.
            batch.sort(key=lambda b: b[1].mono)
            for i, frame in batch:
                self._last_seq[i] = frame.seq
                if self.min_period > 0:
                    if frame.mono < self._next_due[i]:
                        self._timing[i].cap_skipped += 1
                        continue
                    self._next_due[i] = max(self._next_due[i] + self.min_period, frame.mono)
                try:
                    self._handle_frame(i, frame)
                except Exception as e:
                    # One bad frame must not kill the worker (the node would keep running but go
                    # silent). Log and carry on with the next frame.
                    self.get_logger().error(f"Tactip {i}: processing failed: {e!r}",
                                            throttle_duration_sec=1.0)
            self._worker_busy += time.perf_counter() - busy_start

            self._check_stale()
            if self.timing_log and time.monotonic() - self._timing_t0 >= self.TIMING_LOG_PERIOD:
                self._log_timing()

    def _handle_frame(self, i, frame):
        """Predict (if in contact) and publish one frame. Processing and SSIM were already done
        by the sensor's capture thread."""
        stats = self._timing[i]
        if frame.ref_refreshed:
            self.get_logger().info(f"Tactip {i}: SSIM reference updated.")

        # Deduce contact; run the model when in contact (or always, if not zeroing)
        t2 = time.perf_counter()
        if frame.ssim < self.ssim_thresholds[i] or not self.zero_when_no_contact:
            pose, wrench = self.predict_pose_wrench_(i, frame.image)
            stats.predicted += 1
            prep, fwd, dec = self.sensors[i].pose_model.last_timing
            stats.t_prep += prep
            stats.t_fwd += fwd
            stats.t_dec += dec
        else:
            pose, wrench = [0.0] * 6, [0.0] * 6
        t3 = time.perf_counter()

        self.publish_data_(index=i, stamp=frame.stamp.to_msg(), ssim=frame.ssim, pose=pose, wrench=wrench)
        t4 = time.perf_counter()

        age = time.monotonic() - frame.mono
        stats.processed += 1
        stats.t_proc += frame.t_process
        stats.t_ssim += frame.t_ssim
        stats.t_pred += t3 - t2
        stats.t_pub += t4 - t3
        stats.age_sum += age
        stats.age_max = max(stats.age_max, age)

    def _check_stale(self):
        now = time.monotonic()
        for i, sensor in enumerate(self.sensors):
            if sensor.process_errors != self._process_errors_seen[i]:
                self._process_errors_seen[i] = sensor.process_errors
                self.get_logger().error(
                    f"Tactip {i}: image processing failed ({sensor.process_errors} total), "
                    f"last error: {sensor.last_error}", throttle_duration_sec=1.0)
            age = sensor.frame_age()
            if age > self.STALE_TIMEOUT and now - self._last_stale_warn[i] > self.STALE_WARN_INTERVAL:
                self._last_stale_warn[i] = now
                self.get_logger().warn(
                    f"Tactip {i} (/dev/video{sensor.source}): no new frame for {age:.2f} s "
                    f"({sensor.failed_reads} failed reads so far).")

    def _log_timing(self):
        now = time.monotonic()
        dt = now - self._timing_t0
        for i, sensor in enumerate(self.sensors):
            st = self._timing[i]
            n = max(st.processed, 1)
            ms = lambda t: 1000.0 * t / n
            n_pred = max(st.predicted, 1)
            ms_pred = lambda t: 1000.0 * t / n_pred  # per frame the model actually ran on
            self.get_logger().info(
                f"[timing] tactip_{i}: grab {(sensor.seq - st.seq0) / dt:.1f} Hz | "
                f"out {st.processed / dt:.1f} Hz (predict {st.predicted / dt:.1f} Hz, "
                f"cap-skipped {st.cap_skipped}) | "
                f"capture thread: proc {ms(st.t_proc):.1f} ssim {ms(st.t_ssim):.1f} ms | "
                f"worker: predict {ms_pred(st.t_pred):.1f} (prep {ms_pred(st.t_prep):.1f} "
                f"fwd {ms_pred(st.t_fwd):.1f} decode {ms_pred(st.t_dec):.1f}) "
                f"pub {ms(st.t_pub):.1f} ms | "
                f"frame age mean {ms(st.age_sum):.1f} max {1000.0 * st.age_max:.1f} ms | "
                f"read fails {sensor.failed_reads - st.failed_reads0}")
            st.reset(seq=sensor.seq, failed_reads=sensor.failed_reads)
        self.get_logger().info(f"[timing] worker busy {100.0 * self._worker_busy / dt:.0f}%")
        self._worker_busy = 0.0
        self._timing_t0 = now

    # -----------------------------------------------------------------------------------------
    # Data conversion and publishing
    # -----------------------------------------------------------------------------------------

    def predict_pose_wrench_(self, index: int, sensor_image):
        data = self.sensors[index].predict(sensor_image)

        # Convention: Tactip outputs own pose in the contact frame (P_CS)
        data[2] = -data[2]  # Invert Z to comply with convention -
        data[3] = data[3]  # Invert Rx to comply with convention
        data[4] = -data[4]  # Invert Ry to comply with convention -
        data[6] = -data[6]  # Invert shear x to comply with convention
        data[7] = data[7]  # Invert shear y to comply with convention -

        if self.dimensions[index]>6:
            wrench = [data[12], data[13], data[14], 0.0, 0.0, 0.0] # fx, fy, fz
        else:
            wrench = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        pose = [data[6], data[7], data[2], data[3], data[4], 0.0]
        return pose, wrench

    def fake_timer_callback(self):
        stamp = self.get_clock().now().to_msg()
        for i in range(self.fake_channels):
            self.publish_data_(
                index=i,
                stamp=stamp,
                ssim=0.3,
                pose=[0.0, 0.0, -3.0, 0.0, 0.0, 0.0],
                wrench=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def get_ssim_ref_callback(self, request, response):
        # Each sensor's capture thread adopts its next frame as the reference (within one frame
        # period). Returns immediately rather than waiting for that frame.
        self.get_logger().info("Received request for SSIM reference.")
        if request.data:
            for sensor in self.sensors:
                sensor.request_ssim_reference()
        response.success = True
        return response

    def evaluate_P_SC(self, x_CS, y_CS, z_CS, alpha, beta):
        alpha = np.deg2rad(alpha)
        beta = np.deg2rad(beta)
        P_SC = np.zeros((4,4))
        P_SC[0,0] = np.cos(beta)
        P_SC[0,1] = 0
        P_SC[0,2] = -np.sin(beta)
        P_SC[0,3] = -x_CS*np.cos(beta) + z_CS*np.sin(beta)
        P_SC[1,0] = np.sin(alpha)*np.sin(beta)
        P_SC[1,1] = np.cos(alpha)
        P_SC[1,2] = np.sin(alpha)*np.cos(beta)
        P_SC[1,3] = -x_CS*np.sin(alpha)*np.sin(beta) - y_CS*np.cos(alpha) - z_CS*np.sin(alpha)*np.cos(beta)
        P_SC[2,0] = np.sin(beta)*np.cos(alpha)
        P_SC[2,1] = -np.sin(alpha)
        P_SC[2,2] = np.cos(alpha)*np.cos(beta)
        P_SC[2,3] = -x_CS*np.sin(beta)*np.cos(alpha) + y_CS*np.sin(alpha) - z_CS*np.cos(alpha)*np.cos(beta)
        P_SC[3,0] = 0
        P_SC[3,1] = 0
        P_SC[3,2] = 0
        P_SC[3,3] = 1

        return P_SC

    def evaluate_P_CS(self, x_CS, y_CS, z_CS, alpha, beta):
        P_CS = np.zeros((4,4))
        P_CS[0,0] = np.cos(beta)
        P_CS[0,1] = np.sin(alpha)*np.sin(beta)
        P_CS[0,2] = np.sin(beta)*np.cos(alpha)
        P_CS[0,3] = x_CS
        P_CS[1,0] = 0
        P_CS[1,1] = np.cos(alpha)
        P_CS[1,2] = -np.sin(alpha)
        P_CS[1,3] = y_CS
        P_CS[2,0] = -np.sin(beta)
        P_CS[2,1] = np.sin(alpha)*np.cos(beta)
        P_CS[2,2] = np.cos(alpha)*np.cos(beta)
        P_CS[2,3] = z_CS
        P_CS[3,0] = 0
        P_CS[3,1] = 0
        P_CS[3,2] = 0
        P_CS[3,3] = 1

    def publish_data_(self, index: int, stamp, ssim: float, pose: list, wrench: list):
        """Publish one sensor's measurement. stamp is the frame's capture time, so every message
        of this measurement carries the same stamp and consumers can see its true age."""
        if len(pose)>6:
            self.get_logger().warn(f"Pose too large, ignoring extra data")

        # Output the data if verbose set to True
        if self.verbose:
            self.print_data_(index, pose, wrench)

        msg_ssim = Float64()
        msg_ssim.data = float(ssim)
        self.ssim_publishers_[index].publish(msg_ssim)

        msg_contact = Int8()
        msg_contact.data = int(ssim < self.ssim_thresholds[index])
        self.contact_publishers_[index].publish(msg_contact)

        msg = TwistStamped()
        msg.header.stamp = stamp
        msg.twist.linear.x = pose[0]
        msg.twist.linear.y = pose[1]
        msg.twist.linear.z = pose[2]
        msg.twist.angular.x = pose[3]
        msg.twist.angular.y = pose[4]
        msg.twist.angular.z = pose[5]
        self.pose_publishers_[index].publish(msg)

        # Invert the transform in quaternion form
        q = R.from_euler('xyz', [np.deg2rad(pose[3]), np.deg2rad(pose[4]), np.deg2rad(pose[5])]).as_quat()
        q_inv = np.array([-q[0], -q[1], -q[2], q[3]])
        R_inv = R.from_quat(q_inv).as_matrix()
        translation_inv = - (R_inv @ np.array([pose[0], pose[1], pose[2]]))

        # Broadcast the TF
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.sensor_frames[index]
        t.child_frame_id = self.contact_frames[index]
        t.transform.translation.x = float(translation_inv[0])/1000.
        t.transform.translation.y = float(translation_inv[1])/1000.
        t.transform.translation.z = float(translation_inv[2])/1000.
        t.transform.rotation.x = float(q_inv[0])
        t.transform.rotation.y = float(q_inv[1])
        t.transform.rotation.z = float(q_inv[2])
        t.transform.rotation.w = float(q_inv[3])
        self.broadcaster_tf.sendTransform(t)

        if self.wrench_publishers_ and wrench:
            wrench_msg = WrenchStamped()
            wrench_msg.header.stamp = stamp
            wrench_msg.header.frame_id = self.contact_frames[index]
            wrench_msg.wrench.force.x = wrench[0]
            wrench_msg.wrench.force.y = wrench[1]
            wrench_msg.wrench.force.z = wrench[2]
            wrench_msg.wrench.torque.x = wrench[3]
            wrench_msg.wrench.torque.y = wrench[4]
            wrench_msg.wrench.torque.z = wrench[5]
            self.wrench_publishers_[index].publish(wrench_msg)

    def print_data_(self, index: int, pose: list, wrench: list):
            if self.dimensions[index]<=6:
                self.get_logger().info(f"Outputs for Tactip {index}: \n Pose x {pose[0]:.2f}, y {pose[1]:.2f}, z {pose[2]:.2f}, Rx {pose[3]:.2f}, Ry {pose[4]:.2f}, Rz {pose[5]:.2f}")
            elif self.dimensions[index]>6:
                self.get_logger().info(f"Outputs for Tactip {index}: \n Pose x {pose[0]:.2f}, y {pose[1]:.2f}, z {pose[2]:.2f}, Rx {pose[3]:.2f}, Ry {pose[4]:.2f}, Rz {pose[5]:.2f}\n Wrench Fx {wrench[0]:.2f}, Fy {wrench[1]:.2f}, Fz {wrench[2]:.2f}, Tx {wrench[3]:.2f}, Ty {wrench[4]:.2f}, Tz {wrench[5]:.2f}")

    def create_publishers_(self, indices: list):
        for i, s in enumerate(indices):
            self.pose_publishers_.append(self.create_publisher(TwistStamped, '/tactip_'+str(i)+'/pose', 10))
            self.ssim_publishers_.append(self.create_publisher(Float64, '/tactip_'+str(i)+'/ssim', 10))
            self.contact_publishers_.append(self.create_publisher(Int8, '/tactip_'+str(i)+'/contact', 10))
            if self.dimensions[i] > 6:
                self.wrench_publishers_.append(self.create_publisher(WrenchStamped, '/tactip_'+str(i)+'/wrench', 10))
            self.sensor_frames.append('present_sensor_'+str(i))
            self.contact_frames.append('present_contact_'+str(i))
        return

    def destroy_node(self):
        # Stop the worker first (it uses the sensors), then the capture threads / cameras.
        self._stop.set()
        if self._worker is not None:
            with self._cond:
                self._cond.notify_all()
            self._worker.join(timeout=2.0)
            self._worker = None
        for sensor in self.sensors:
            sensor.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TactipDriver()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
