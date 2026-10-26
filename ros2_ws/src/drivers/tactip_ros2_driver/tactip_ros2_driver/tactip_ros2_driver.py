import rclpy
from rclpy.node import Node
import time
import os

import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import TwistStamped, TransformStamped, WrenchStamped
from std_msgs.msg import Float64, Int8
from tf2_ros import TransformBroadcaster

from std_srvs.srv import SetBool

from .tactip import TacTip

class TactipDriver(Node):
    def __init__(self):
        super().__init__('tactip_driver')

        # Parameters
        # Sensor-specific
        self.declare_parameter('sources', [0])
        self.declare_parameter('dimensions', [8])
        self.declare_parameter('ssim_contact_thresholds', [0.9])
        self.declare_parameter('model_names', ['model_name'])

        # Node-level
        self.declare_parameter('frequency', 10.)
        self.declare_parameter('fake_data', False)
        self.declare_parameter('verbose', True)
        self.declare_parameter('test_model_time', False)
        self.declare_parameter('zero_when_no_contact', True)

        # Grabbing the parameters
        self.fake_data = self.get_parameter('fake_data').get_parameter_value().bool_value
        self.ssim_thresholds = self.get_parameter('ssim_contact_thresholds').get_parameter_value().double_array_value
        self.model_names = self.get_parameter('model_names').get_parameter_value().string_array_value
        self.dimensions = self.get_parameter('dimensions').get_parameter_value().integer_array_value

        # Initializing lists
        self.sensors = []
        self.sensor_frames = []
        self.contact_frames = []
        self.pose_publishers_ = []
        self.wrench_publishers_ = []
        self.ssim_publishers_ = []
        self.contact_publishers_ = []
        self.ref_ssim_images_ = []

        # Other node data
        self.fake_channels = None

        # Node feedback
        self.get_logger().info(f"Tactip driver initialized with dimensions: {list(self.dimensions)} and ssim thresholds: {self.ssim_thresholds}")

        # Initialize the sensor(s)
        if not self.fake_data:
            # Initialize TacTip sensors
            for i, s in enumerate(self.get_parameter('sources').get_parameter_value().integer_array_value):
                self.sensors.append(TacTip(s,self.model_names[i]))
                self.get_logger().info(f"Start sensor {i}, reading from /dev/video{s}" )
                self.get_logger().info(f"Sensor processing params: {self.sensors[-1].sensor_params}")

                # Reference SSIM image
                self.ref_ssim_images_.append(self.sensors[-1].process().squeeze())
            # Create publishers
            self.create_publishers_(list(self.get_parameter('sources').get_parameter_value().integer_array_value))
        else:
            self.fake_channels = len(self.get_parameter('sources').get_parameter_value().integer_array_value)
            self.get_logger().info(f"Using fake data mode on {self.fake_channels} channels. No camera input.")
            self.create_publishers_(range(self.fake_channels))             

        # Broadcaster TF
        self.broadcaster_tf = TransformBroadcaster(self)

        # Run testing model evaluation time functionality if enabled in Launch
        if self.get_parameter('test_model_time').get_parameter_value().bool_value:
            self.get_logger().info("Testing model execution time. It will run 1000 predictions through the model and print the average time taken.")
            self.test_model_execution_time()

        # Start service to refresh SSIM of all images
        self.srv_get_ssim_ref = self.create_service(
            SetBool,                    # service type
            'set_ssim_ref',             # service name
            self.get_ssim_ref_callback  # callback
        )
 
        # Set up timer
        self.cycle_counter = 0
        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value
        self.period = 1.0/self.frequency # seconds
        self.timer = self.create_timer(self.period, self.timer_callback)

    def timer_callback(self):
        # Fake data hijack
        if self.fake_data:
            for i in range(self.fake_channels):
                self.publish_fake_data(index=i)
            return

        # Evaluate for each sensor
        for i, sensor in enumerate(self.sensors):
            sensor_image = sensor.process()
            ssim_score = ssim(self.ref_ssim_images_[i], sensor_image.squeeze())

            # Deduce contact, if contact, publish real data
            if ssim_score < self.ssim_thresholds[i]: # contact true
                self.publish_real_data_(index=i, sensor_image=sensor_image, ssim=ssim_score)
            else:
                if self.get_parameter('zero_when_no_contact').get_parameter_value().bool_value:
                    self.publish_zero_data_(index=i, ssim=ssim_score)
                else:
                    self.publish_real_data_(index=i, sensor_image=sensor_image, ssim=ssim_score)

        self.cycle_counter +=1

    def publish_real_data_(self, index: int, sensor_image, ssim: float):
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

        self.publish_data_(index=index,
                           ssim=ssim,
                           pose=[data[6], data[7], data[2], data[3], data[4], 0.0],
                           wrench=wrench)

    def publish_zero_data_(self, index, ssim):
        self.publish_data_(index=index,
                            ssim=ssim,
                            pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            wrench=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def publish_fake_data(self, index):
        self.publish_data_(
            index=index,
            ssim=0.3,
            pose=[0.0, 0.0, -3.0, 0.0, 0.0, 0.0],
            wrench=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_model_execution_time(self, iterations = 1000):
        start_time = time.time()
        capture_time = np.zeros((len(self.sensors), iterations))
        predict_time = np.zeros((len(self.sensors), iterations))
        for i in range(iterations):
            for s, sensor in enumerate(self.sensors):
                capture_start_time = time.time()
                processed_image = sensor.process()
                capture_end_time = time.time()
                _ = sensor.predict(processed_image)
                predict_end_time = time.time()
                capture_time[s, i] = capture_end_time - capture_start_time
                predict_time[s, i] = predict_end_time - capture_end_time
        end_time = time.time()
        self.get_logger().info(f"Total time taken for {iterations} iterations: {end_time - start_time} [seconds]")
        self.get_logger().info(f"Average time taken: {(end_time - start_time)/iterations:.3f} [seconds/it]")
        self.get_logger().info(f"Iterations per second: {iterations/(end_time - start_time):.2f} [Hz]")
        for s, sensor in enumerate(self.sensors):
            self.get_logger().info(f"[{s}] Average capture time: {sum(capture_time[s,:])/iterations*1000:.4f} [ms/it]")
            self.get_logger().info(f"[{s}] Average predict time: {sum(predict_time[s,:])/iterations*1000:.4f} [ms/it]")

    def get_ssim_ref_callback(self, request, response):
        # This is a dummy service callback. You can implement functionality to update the reference image or return the current reference image's SSIM score.
        self.get_logger().info("Received request for SSIM reference.")
        if request.data:
            for i, sensor in enumerate(self.sensors):
                self.ref_ssim_images_[i] = sensor.process().squeeze()
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

    def publish_data_(self, index: int, ssim:float=None, pose: list=None, wrench: list=None):
        if len(pose)>6:
            self.get_logger().warn(f"Pose too large, ignoring extra data")

        # Output the data if verbose set to True
        if self.get_parameter('verbose').get_parameter_value().bool_value:
            self.print_data_(index, pose, wrench)
        
        msg_ssim = Float64()
        msg_ssim.data = ssim
        self.ssim_publishers_[index].publish(msg_ssim)

        msg_contact = Int8()
        msg_contact.data = int(ssim < self.ssim_thresholds[index])
        self.contact_publishers_[index].publish(msg_contact)

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
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
        t.header.stamp = self.get_clock().now().to_msg()
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
            wrench_msg.header.stamp = self.get_clock().now().to_msg()
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

def main(args=None):
    rclpy.init(args=args)
    node = TactipDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()