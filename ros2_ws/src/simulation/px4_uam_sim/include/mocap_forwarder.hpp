#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "px4_msgs/msg/vehicle_odometry.hpp"
#include "px4_msgs/msg/timesync_status.hpp"

using namespace std::chrono_literals;

class MocapForwarder : public rclcpp::Node
{
public:
    MocapForwarder();

private:
    /**
     * @brief Publish a trajectory setpoint
     */
    void _handle_visual_odometry(const nav_msgs::msg::Odometry::SharedPtr msg);

    /**
     * @brief publish vehicle angles and angular rates from flight controller
     */
    void _handle_vehicle_odometry(const px4_msgs::msg::VehicleOdometry::SharedPtr msg);

    std::string _sub_topic;

private: 

    /* Callback Functions */
    void _timesync_callback(const px4_msgs::msg::TimesyncStatus::SharedPtr msg);    uint64_t 
    
    get_timestamp();

    /* Publishers and Subscribers */
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr _vehicle_odom_sub;
    rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr _fmu_odom_sub;
    
    rclcpp::Publisher<px4_msgs::msg::VehicleOdometry>::SharedPtr _vehicle_odom_pub;
    rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr _vehicle_body_angular_rates_pub;
    rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr _vehicle_body_angles_pub;
};
