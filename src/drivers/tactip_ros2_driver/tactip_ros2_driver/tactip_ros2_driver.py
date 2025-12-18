import rclpy
from rclpy.node import Node
from math import pi
import time
import os
import math
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import TwistStamped, TransformStamped
from std_msgs.msg import Float64, Int8
from tf2_ros import TransformBroadcaster

from .tactip import TacTip

from .dependencies.label_encoder import BASE_MODEL_PATH

class TactipDriver(Node):
    def __init__(self):
        super().__init__('tactip_driver')

        # Parameters
        self.declare_parameter('source', 0)
        self.declare_parameter('frequency', 10.)
        self.declare_parameter('dimension', 5)
        self.declare_parameter('verbose', True)
        self.declare_parameter('test_model_time', False)
        self.declare_parameter('save_debug_image', False)
        self.declare_parameter('save_interval', 1.)
        self.declare_parameter('ssim_contact_threshold', 0.7)
        self.declare_parameter('save_directory', 'Please set a save_directory in the launch file')
        self.declare_parameter('zero_when_no_contact', False)
        self.declare_parameter('fake_data', False)
        self.fake_data = self.get_parameter('fake_data').get_parameter_value().bool_value
        self.image_save_interval = self.get_parameter('save_interval').get_parameter_value().double_value
        self.ssim_threshold = self.get_parameter('ssim_contact_threshold').get_parameter_value().double_value
        self.dimension = self.get_parameter('dimension').get_parameter_value().integer_value

        # Node feedback
        self.get_logger().info("Tactip driver initialized with dimension: "+str(self.dimension))
        self.get_logger().info(BASE_MODEL_PATH)
        if not self.fake_data:
            # Initialize TacTip sensor
            self.sensor = TacTip(self.get_parameter('source').get_parameter_value().integer_value)
            self.get_logger().info(f"Reading from /dev/video{self.get_parameter('source').get_parameter_value().integer_value}" )
            self.get_logger().info(f"Sensor processing params: {self.sensor.sensor_params}")
            # Reference image
            self.ref_image_ssim = self.sensor.process().squeeze()
        else:
            self.get_logger().info("Using fake data mode. No camera input.")

        # Set up period image saving if enabled in launch
        if self.get_parameter('save_debug_image').get_parameter_value().bool_value:
            self.image_outfile_path = self.get_parameter('save_directory').get_parameter_value().string_value
            self.get_logger().info(f'Saving debug images in {self.image_outfile_path}')
            if not os.path.exists(self.image_outfile_path):
                os.makedirs(self.image_outfile_path)
            self.img_counter = 0
            if len(os.listdir(self.image_outfile_path)) != 0: # If directory is not empty, exit to avoid overwriting data
                self.get_logger().error("Directory not empty. Please delete or move data before proceeding.")
                self.get_logger().error("Node will now exit. Please kill stack with  Ctrl+C.")
                return
            else:
                with open(os.path.join(self.get_parameter('save_directory').get_parameter_value().string_value, 'image_list.csv'), 'w') as f:
                    f.writelines([f"Image number, shear_x, shear_y, pose_z, pose_Rx, pose_Ry, shear_Rz \n"]) # Create empty csv file to log images

        # publishers
        self.publisher_pose_ = self.create_publisher(TwistStamped, '/tactip/pose', 10)
        self.publisher_ssim_ = self.create_publisher(Float64, '/tactip/ssim', 10)
        self.publisher_contact_ = self.create_publisher(Int8, '/tactip/contact', 10)

        # Broadcaster TF
        self.broadcaster_tf = TransformBroadcaster(self)

        # Run testing model evaluation time functionality if enabled in Launch
        if self.get_parameter('test_model_time').get_parameter_value().bool_value:
            self.get_logger().info("Testing model execution time. It will run 1000 predictions through the model and print the average time taken.")
            self.test_model_execution_time()

        # Set up timer
        self.cycle_counter = 0
        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value
        self.period = 1.0/self.frequency # seconds
        self.timer = self.create_timer(self.period, self.timer_callback)

    def timer_callback(self):
        # Fake data hijack
        if self.fake_data:
            self.publish_fake_data()
            return

        # read the data
        # Save an image every second
        if self.get_parameter('save_debug_image').get_parameter_value().bool_value and self.cycle_counter%int(self.frequency*self.image_save_interval) == 0:
            raw_outfile = os.path.join(self.image_outfile_path,'raw_image'+str(self.img_counter)+'.png')
            proc_outfile = os.path.join(self.image_outfile_path,'sensor_image'+str(self.img_counter)+'.png')
            sensor_image = self.sensor.process(raw_outfile=raw_outfile, proc_outfile=proc_outfile)
            self.img_counter += 1
            self.cycle_counter = 0
        
        # Just read data without saving
        else:
            sensor_image = self.sensor.process()
        
        # Get SSIM
        ssim_score = ssim(self.ref_image_ssim, sensor_image.squeeze())
        msg = Float64()
        msg.data = ssim_score
        self.publisher_ssim_.publish(msg)

        # Deduce contact, if contact, publish real data
        if ssim_score < self.ssim_threshold:
            contact = True
            self.publish_real_data(sensor_image)
        else:
            contact = False
            if self.get_parameter('zero_when_no_contact').get_parameter_value().bool_value:
                self.publish_zero_data()
            else:
                self.publish_real_data(sensor_image)
        msg = Int8()
        msg.data = contact
        self.publisher_contact_.publish(msg)

        self.cycle_counter +=1

    def publish_real_data(self, sensor_image):
        data = self.sensor.predict(sensor_image)

        if self.get_parameter('save_debug_image').get_parameter_value().bool_value and self.cycle_counter%int(self.frequency*self.image_save_interval) == 0:
            with open(os.path.join(self.get_parameter('save_directory').get_parameter_value().string_value, 'image_list.csv'), 'a') as f:
                f.writelines([f"{self.img_counter}, {data[6]:.2f}, {data[7]:.2f}, {data[2]:.2f}, {data[3]:.2f}, {data[4]:.2f}, {data[11]:.2f} \n"]) # Log image data


        data[2] = -data[2]  # Invert Z to comply with convention
        data[3] = -data[3]  # Invert Rx to comply with
        data[4] = data[4]  # 
        data[6] = data[6]  # Invert shear x
        data[7] = -data[7]  #

        # Rotation from output frame (P_CS - sensor frame in contact frame) to desired frame (P_SC - contact frame in sensor frame)
        if self.dimension == 3:
            P_SC = self.evaluate_P_SC(0., 0., data[2], data[3], data[4]) # Input pose elements as rad and meter, set sign to comply with convention
        elif self.dimension == 5:
            P_SC = self.evaluate_P_SC(data[6], data[7], data[2], data[3], data[4]) # Input pose elements as rad and meter, set sign to comply with convention
        
        rot_pred_pos = P_SC[0:3,3]
        rot_pred_ang = np.rad2deg(R.from_matrix(P_SC[0:3, 0:3]).as_euler('xyz'))
        rot_pred_pose = np.concatenate([rot_pred_pos, rot_pred_ang])

        if self.get_parameter('verbose').get_parameter_value().bool_value and self.dimension == 3:
            self.get_logger().info(f"[P_SC] Z (mm): {rot_pred_pose[2]:.2f} \t Rx (deg): {rot_pred_pose[3]:.2f} \t Ry (deg): {rot_pred_pose[4]:.2f}")
        
        elif self.get_parameter('verbose').get_parameter_value().bool_value and self.dimension == 5:
            self.get_logger().info(f"[P_SC] X (mm): {rot_pred_pose[0]:.2f} \t Y (mm): {rot_pred_pose[1]:.2f} \t Z (mm): {rot_pred_pose[2]:.2f} \t Rx (deg): {rot_pred_pose[3]:.2f} \t Ry (deg): {rot_pred_pose[4]:.2f}")

        # publish the data
        # The model outputs are in mm and deg, so convert to SI
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = rot_pred_pose[0]
        msg.twist.linear.y = rot_pred_pose[1]
        msg.twist.linear.z = rot_pred_pose[2]
        msg.twist.angular.x = rot_pred_pose[3] # deg2rad
        msg.twist.angular.y = rot_pred_pose[4] # deg2rad
        msg.twist.angular.z = rot_pred_pose[5] # deg2rad
        self.publisher_pose_.publish(msg)
        #self.get_logger().info(f"Published data: {msg}")

        # Invert the transform
        q = R.from_euler('xyz', [np.deg2rad(data[3]), np.deg2rad(data[4]), np.deg2rad(data[5])]).as_quat()
        q_inv = np.array([-q[0], -q[1], -q[2], q[3]])
        R_inv = R.from_quat(q_inv).as_matrix()
        translation_inv = - (R_inv @ np.array([data[6],data[7],data[2]]))

        # Broadcast the TF
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "present_sensor_frame"
        t.child_frame_id = "present_contact_frame"
        t.transform.translation.x = float(translation_inv[0])
        t.transform.translation.y = float(translation_inv[1])
        t.transform.translation.z = float(translation_inv[2])
        t.transform.rotation.x = float(q_inv[0])
        t.transform.rotation.y = float(q_inv[1])
        t.transform.rotation.z = float(q_inv[2])
        t.transform.rotation.w = float(q_inv[3])
        self.broadcaster_tf.sendTransform(t)

    def publish_zero_data(self):  
        if self.get_parameter('verbose').get_parameter_value().bool_value and self.dimension == 3:
            self.get_logger().info(f"[P_SC] Z (mm): 0.00 \t Rx (deg): 0.00 \t Ry (deg): 0.00 (no contact)")
        elif self.get_parameter('verbose').get_parameter_value().bool_value and self.dimension == 5:
            self.get_logger().info(f"[P_SC] X (mm): 0.00 \t Y (mm): 0.00 \t Z (mm): 0.00 \t Rx (deg): 0.00 \t Ry (deg): 0.00 (no contact)")

        if self.get_parameter('save_debug_image').get_parameter_value().bool_value and self.cycle_counter%int(self.frequency*self.image_save_interval) == 0:
            with open(os.path.join(self.get_parameter('save_directory').get_parameter_value().string_value, 'image_list.csv'), 'a') as f:
                f.writelines([f"{self.img_counter}, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 \n"]) # Log image data

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
        self.publisher_pose_.publish(msg)

        # Broadcast the TF
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "present_sensor_frame"
        t.child_frame_id = "present_contact_frame"
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.broadcaster_tf.sendTransform(t)

    def test_model_execution_time(self, iterations = 1000):
        start_time = time.time()
        capture_time = []
        predict_time = []
        for i in range(iterations):
            capture_start_time = time.time()
            processed_image = self.sensor.process()
            capture_end_time = time.time()
            _ = self.sensor.predict(processed_image)
            predict_end_time = time.time()
            capture_time.append(capture_end_time - capture_start_time)
            predict_time.append(predict_end_time - capture_end_time)
        end_time = time.time()
        self.get_logger().info(f"Time taken for {iterations} iterations: {end_time - start_time} [seconds]")
        self.get_logger().info(f"Average time taken: {(end_time - start_time)/iterations} [seconds/it]")
        self.get_logger().info(f"Iterations per second: {iterations/(end_time - start_time)} [it/s]")
        self.get_logger().info(f"Average capture time: {sum(capture_time)/iterations} [seconds/it]")
        self.get_logger().info(f"Average predict time: {sum(predict_time)/iterations} [seconds/it]")
    
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
    
    def publish_fake_data(self):
        # Generate fake data for testing without camera
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        t = time.time()
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.03*math.sin(0.5*t)
        msg.twist.angular.x = np.deg2rad(15.*math.sin(0.4*t))
        msg.twist.angular.y = np.deg2rad(15.*math.sin(0.6*t))
        msg.twist.angular.z = np.deg2rad(15.*math.sin(0.8*t))
        self.publisher_pose_.publish(msg)

        ssim_score = 0.3
        msg_ssim = Float64()
        msg_ssim.data = ssim_score
        self.publisher_ssim_.publish(msg_ssim)

        contact = ssim_score < self.ssim_threshold
        msg_contact = Int8()
        msg_contact.data = contact
        self.publisher_contact_.publish(msg_contact)

def main(args=None):
    rclpy.init(args=args)
    node = TactipDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()