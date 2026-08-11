#include "mocap_forwarder.hpp"

#include <Eigen/Dense>

MocapForwarder::MocapForwarder():rclcpp::Node("mocap_forwarder")
{
    /* Declare all the parameters */
    this->declare_parameter("sub_topic", "/odom");

    this->_sub_topic = this->get_parameter("sub_topic").as_string();

    /* Init Subscriber*/
    this->_vehicle_odom_sub = this->create_subscription<nav_msgs::msg::Odometry>(
                this->_sub_topic, rclcpp::SensorDataQoS(), std::bind(
                &MocapForwarder::_handle_visual_odometry, this, std::placeholders::_1));
    
    rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
    auto qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, 5), qos_profile);
                
    this->_fmu_odom_sub = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
                "/fmu/out/vehicle_odometry", qos, std::bind(
                &MocapForwarder::_handle_vehicle_odometry, this, std::placeholders::_1));
    
    /* Init Publisher */
    this->_vehicle_odom_pub = this->create_publisher<px4_msgs::msg::VehicleOdometry>(
        "/fmu/in/vehicle_visual_odometry", 10);

    /* Tactile servoing publishers */
    this->_vehicle_body_angles_pub = this->create_publisher<geometry_msgs::msg::Vector3Stamped>(
        "/state/body_angles", 10);
    this->_vehicle_body_angular_rates_pub = this->create_publisher<geometry_msgs::msg::Vector3Stamped>(
        "/state/body_velocity", 10);
}

void MocapForwarder::_handle_visual_odometry(
    const nav_msgs::msg::Odometry::SharedPtr msg)
{
    px4_msgs::msg::VehicleOdometry odom{};
    odom.timestamp = this->now().nanoseconds() / 1000.0;
    odom.timestamp_sample = odom.timestamp;
    odom.pose_frame = px4_msgs::msg::VehicleOdometry::POSE_FRAME_NED;

    odom.position = {(float)msg->pose.pose.position.y,
                     (float)msg->pose.pose.position.x,
                    -(float)msg->pose.pose.position.z};

    Eigen::Quaterniond q = Eigen::Quaterniond(0.7071068, 0.0, 0.0, 0.7071068)
                         * Eigen::Quaterniond(msg->pose.pose.orientation.w,
                                              msg->pose.pose.orientation.y,
                                              msg->pose.pose.orientation.x,
                                              -msg->pose.pose.orientation.z);
    odom.q = {(float)q.w(), (float)q.x(), (float)q.y(), (float)q.z()};

    this->_vehicle_odom_pub->publish(odom);
}

void MocapForwarder::_handle_vehicle_odometry(
    const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
{
    geometry_msgs::msg::Vector3Stamped angle_msg{};
    angle_msg.header.stamp.nanosec = this->now().nanoseconds();
    angle_msg.header.stamp.sec = this->now().seconds();
    angle_msg.vector.z = std::atan2(2*(msg->q[0]*msg->q[3] + msg->q[1]*msg->q[2]),
        (1-2*(msg->q[2]*msg->q[2] + msg->q[3]*msg->q[3])));
    angle_msg.vector.y = std::atan2(std::sqrt(1+2*(msg->q[0]*msg->q[2] - msg->q[1]*msg->q[3])),
        std::sqrt(1-2*(msg->q[0]*msg->q[2] - msg->q[1]*msg->q[3])));
    angle_msg.vector.x = std::atan2(2*(msg->q[0]*msg->q[1] + msg->q[2]*msg->q[3]),
        1-2*(msg->q[1]*msg->q[1] + msg->q[2]*msg->q[2]));

    this->_vehicle_body_angles_pub->publish(angle_msg);
    
    geometry_msgs::msg::Vector3Stamped rate_msg{};
    rate_msg.header.stamp.nanosec = this->now().nanoseconds();
    rate_msg.header.stamp.sec = this->now().seconds(); // Check if this is forwarded correctly because angle_msg out put is 0.0
    angle_msg.vector.x = msg->angular_velocity[0];
    angle_msg.vector.y = msg->angular_velocity[1];
    angle_msg.vector.z = msg->angular_velocity[2];

    this->_vehicle_body_angular_rates_pub->publish(rate_msg);
}


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
    
  rclcpp::spin(std::make_shared<MocapForwarder>());
  rclcpp::shutdown();
  return 0;
}
