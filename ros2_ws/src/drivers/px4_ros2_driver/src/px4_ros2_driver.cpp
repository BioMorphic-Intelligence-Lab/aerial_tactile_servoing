#include "px4_ros2_driver/px4_ros2_driver.hpp"


/* This example creates a subclass of Node and uses std::bind() to register a
* member function as a callback from the timer. */

// Initializer
PX4ROS2Driver::PX4ROS2Driver() :
    Node("px4_ros2_driver"),
    offboard_control_mode_publisher_(this->create_publisher<OffboardControlMode>("/fmu/in/offboard_control_mode", 10)),
    trajectory_setpoint_publisher_(this->create_publisher<TrajectorySetpoint>("/fmu/in/trajectory_setpoint", 10)),
    vehicle_command_publisher_(this->create_publisher<VehicleCommand>("/fmu/in/vehicle_command", 10)),
    vehicle_rates_setpoint_publisher_(this->create_publisher<VehicleRatesSetpoint>("/fmu/in/vehicle_rates_setpoint", 10)),
    vehicle_status_subscription_(nullptr),
    vehicle_body_rates_reference_subscription_(nullptr),
    latest_rate_reference_{0.0, 0.0, 0.0},
    latest_vel_reference_{0.0, 0.0, 0.0},
    offboard_mode_(OffboardMode::POSITION)
    {
        // Set QOS settings
        rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
        auto qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, 5), qos_profile);

        // Subscriptions
        
        vehicle_status_subscription_ = this->create_subscription<VehicleStatus>("fmu/out/vehicle_status", qos, std::bind(&PX4ROS2Driver::vehicleStatusCallback, this, std::placeholders::_1));
        vehicle_body_rates_reference_subscription_ = this->create_subscription<geometry_msgs::msg::Vector3Stamped>("/references/body_velocities", qos, std::bind(&PX4ROS2Driver::vehicleBodyRatesReferenceCallback, this, std::placeholders::_1));
        vehicle_velocity_reference_subscription_ = this->create_subscription<geometry_msgs::msg::Vector3Stamped>("/references/velocity", qos, std::bind(&PX4ROS2Driver::vehicleVelocityReferenceCallback, this, std::placeholders::_1));

        // Mode parameter
        this->declare_parameter("offboard_mode", "position");
        setDriverMode();

        // Create timer
        this->declare_parameter("frequency", 10);
        float frequency = this->get_parameter("frequency").as_double();
        timer_ = this->create_wall_timer(std::chrono::duration<float>(1.0f / frequency), std::bind(&PX4ROS2Driver::timer_callback, this));

        RCLCPP_INFO(this->get_logger(), "PX4 ROS2 Driver has been initialized");
      }

void PX4ROS2Driver::timer_callback()
{
    publishOffboardControlMode();
    publishSetpoint(latest_rate_reference_[2], latest_rate_reference_[1], latest_rate_reference_[0], );
}

void PX4ROS2Driver::vehicleStatusCallback(const VehicleStatus::SharedPtr msg)
{
  
}

void PX4ROS2Driver::vehicleBodyRatesReferenceCallback(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
{
    latest_rate_reference_[0] = msg->vector.z; // Yaw rate
    latest_rate_reference_[1] = msg->vector.y; // Pitch rate
    latest_rate_reference_[2] = msg->vector.x; // Roll rate
}

void PX4ROS2Driver::vehicleVelocityReferenceCallback(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
{
    latest_rate_reference_[0] = msg->vector.x; // Yaw rate
    latest_rate_reference_[1] = msg->vector.y; // Pitch rate
    latest_rate_reference_[2] = msg->vector.z; // Roll rate
}

void PX4ROS2Driver::setDriverMode()
{
    std::string mode = this->get_parameter("offboard_mode").as_string();
    if (mode=="position")
    {
        offboard_mode_ = OffboardMode::POSITION;
    }
    else if (mode=="velocity")
    {
        offboard_mode_ = OffboardMode::VELOCITY;
    }
    else if (mode=="acceleration")
    {
        offboard_mode_ = OffboardMode::ACCELERATION;
    }
    else if (mode=="attitude")
    {
        offboard_mode_ = OffboardMode::ATTITUDE;
    }
    else if (mode=="body_rate")
    {
        offboard_mode_ = OffboardMode::BODY_RATE;
    }
    else if (mode=="thrust_and_torque")
    {
        offboard_mode_ = OffboardMode::THRUST_AND_TORQUE;
    }
    else if (mode=="direct_actuator")
    {
        offboard_mode_ = OffboardMode::DIRECT_ACTUATOR;
    }
    else
    {
        RCLCPP_ERROR(this->get_logger(), "Invalid offboard mode: %s", mode.c_str());
        RCLCPP_ERROR(this->get_logger(), "Crashing node due to invalid mode. Please set a valid mode.");
        exit(1); // Crash the node if the mode is valid
    }
    
}

void PX4ROS2Driver::land()
{
  publishVehicleCommand(VehicleCommand::VEHICLE_CMD_NAV_LAND, 0, 0);
}

void PX4ROS2Driver::publishOffboardControlMode()
{
    OffboardControlMode msg{};
    msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
    msg.position = false;
    msg.velocity = false;
    msg.acceleration = false;
    msg.attitude = false;
    msg.body_rate = false;
    msg.thrust_and_torque = false;
    msg.direct_actuator = false;

    switch (offboard_mode_)
    {
    case OffboardMode::POSITION:
        msg.position = true;
        break;
    case OffboardMode::VELOCITY:
        msg.velocity = true;
        break;
    case OffboardMode::ACCELERATION:
        msg.acceleration = true;
        break;
    case OffboardMode::ATTITUDE:
        msg.attitude = true;
        break;
    case OffboardMode::BODY_RATE:
        msg.body_rate = true;
        break;
    case OffboardMode::THRUST_AND_TORQUE:
        msg.thrust_and_torque = true;
        break;
    case OffboardMode::DIRECT_ACTUATOR:
        msg.direct_actuator = true;
        break;
    }

    offboard_control_mode_publisher_->publish(msg);
}

void PX4ROS2Driver::publishSetpoint(float setpoint0, float setpoint1, float setpoint2, float setpoint3)
{
    switch (offboard_mode_)
    {
      case OffboardMode::POSITION || OffboardMode::VELOCITY || OffboardMode::ACCELERATION:
        publishTrajectorySetpoint(setpoint0, setpoint1, setpoint2, setpoint3);
        break;
      case OffboardMode::ATTITUDE:
        publishVehicleAttitudeSetpoint();
        break;
      case OffboardMode::BODY_RATE:
        publishVehicleRatesSetpoint(setpoint0, setpoint1, setpoint2, setpoint3);
        break;
      case OffboardMode::THRUST_AND_TORQUE:
        publishVehicleThrustSetpoint();
        break;
      case OffboardMode::DIRECT_ACTUATOR:
        publishActuatorMotorsSetpoint();
        break;
    }
}

void PX4ROS2Driver::publishTrajectorySetpoint(float x, float y, float z, float yaw)
{
    TrajectorySetpoint msg{};
    switch (offboard_mode_)
    {
        case OffboardMode::POSITION:
            msg.position[0] = x;
            msg.position[1] = y;
            msg.position[2] = z;
            msg.yaw = yaw;
            break;
        case OffboardMode::VELOCITY:
            msg.velocity[0] = x;
            msg.velocity[1] = y;
            msg.velocity[2] = z;
            msg.yawspeed = yaw;
            break;
        case OffboardMode::ACCELERATION:
            msg.acceleration[0] = x;
            msg.acceleration[1] = y;
            msg.acceleration[2] = z;
            msg.yawspeed = yaw;
            break;
    }
    msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
    trajectory_setpoint_publisher_->publish(msg);
}

void PX4ROS2Driver::publishVehicleAttitudeSetpoint()
{
  std::runtime_error("Not implemented yet");
}

void PX4ROS2Driver::publishVehicleRatesSetpoint(float roll_rate, float pitch_rate, float yaw_rate, float thrust)
{
    VehicleRatesSetpoint msg{};
    msg.roll = roll_rate;
    msg.pitch = pitch_rate;
    msg.yaw = yaw_rate;
    msg.thrust_body[0] = 0.0;
    msg.thrust_body[1] = 0.0;
    msg.thrust_body[2] = thrust;
    msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
    vehicle_rates_setpoint_publisher_->publish(msg);
}

void PX4ROS2Driver::publishVehicleThrustSetpoint()
{
  std::runtime_error("Not implemented yet");
}

void PX4ROS2Driver::publishActuatorMotorsSetpoint()
{
  std::runtime_error("Not implemented yet");
}

void PX4ROS2Driver::publishVehicleCommand(
    uint16_t command, 
    float param1 = 0.0, 
    float param2 = 0.0, 
    float param3 = 0.0, 
    float param4 = 0.0, 
    float param5 = 0.0, 
    float param6 = 0.0, 
    float param7 = 0.0)
{
	VehicleCommand msg{};
	msg.param1 = param1;
	msg.param2 = param2;
    msg.param3 = param3;
    msg.param4 = param4;
    msg.param5 = param5;
    msg.param6 = param6;
    msg.param7 = param7;
	msg.command = command;
	msg.target_system = 1;
	msg.target_component = 1;
	msg.source_system = 1;
	msg.source_component = 1;
	msg.from_external = true;
	msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
	vehicle_command_publisher_->publish(msg);
}


int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PX4ROS2Driver>());
    rclcpp::shutdown();
    return 0;
}