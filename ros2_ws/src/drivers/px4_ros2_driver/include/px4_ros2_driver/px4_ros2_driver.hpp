#include "rclcpp/rclcpp.hpp"

#include "px4_msgs/msg/offboard_control_mode.hpp"
#include "px4_msgs/msg/trajectory_setpoint.hpp"
#include "px4_msgs/msg/vehicle_attitude_setpoint.hpp"
#include "px4_msgs/msg/vehicle_rates_setpoint.hpp"
#include "px4_msgs/msg/vehicle_status.hpp"
#include "px4_msgs/msg/vehicle_command.hpp"

#include "geometry_msgs/msg/vector3_stamped.hpp"

using namespace px4_msgs::msg;

enum OffboardMode {
  POSITION,
  VELOCITY,
  ACCELERATION,
  ATTITUDE,
  BODY_RATE,
  THRUST_AND_TORQUE,
  DIRECT_ACTUATOR
};

class PX4ROS2Driver : public rclcpp::Node
{
  public:
    PX4ROS2Driver();
    ~PX4ROS2Driver();

  private:
    // Callbacks
    void timer_callback();
    void vehicleStatusCallback(const VehicleStatus::SharedPtr msg);
    void vehicleBodyRatesReferenceCallback(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg);

    void land();
  
    // Publish methods
    void publishOffboardControlMode();
    void publishSetpoint(_Float32 setpoint0, _Float32 setpoint1, _Float32 setpoint2, _Float32 setpoint3);
    void publishTrajectorySetpoint(_Float32 x, _Float32 y, _Float32 z, _Float32 yaw);
    void publishVehicleAttitudeSetpoint();
    void publishVehicleRatesSetpoint(_Float32 roll_rate, _Float32 pitch_rate, _Float32 yaw_rate, _Float32 thrust);
    void publishVehicleThrustSetpoint();
    void publishActuatorMotorsSetpoint();
    void publishVehicleCommand(
      uint16_t command, 
      float param1 = 0.0, 
      float param2 = 0.0, 
      float param3 = 0.0, 
      float param4 = 0.0, 
      float param5 = 0.0, 
      float param6 = 0.0, 
      float param7 = 0.0);

    rclcpp::TimerBase::SharedPtr timer_;

    // Private methods
    void setDriverMode();

    // Publishers
    rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
    rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
    rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
    rclcpp::Publisher<VehicleAttitudeSetpoint>::SharedPtr vehicle_attitude_setpoint_publisher_;
    rclcpp::Publisher<VehicleRatesSetpoint>::SharedPtr vehicle_rates_setpoint_publisher_;

    // Subscriptions
    rclcpp::Subscription<VehicleStatus>::SharedPtr vehicle_status_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr vehicle_body_rates_reference_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr vehicle_velocity_reference_subscription_;

    OffboardMode offboard_mode_;

    // Node data
    VehicleStatus vehicle_status_;
    float latest_rate_reference_[3];
    float latest_vel_reference_[3];
  };