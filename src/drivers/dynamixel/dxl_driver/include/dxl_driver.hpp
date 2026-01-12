#ifndef DXL_DRIVER_NODE_HPP_
#define DXL_DRIVER_NODE_HPP_

#include "rclcpp/rclcpp.hpp"
#include "dynamixel_sdk/dynamixel_sdk.h"

#include <sensor_msgs/msg/joint_state.hpp>

#include <dxl_driver/srv/set_torque_enable.hpp>
#include <dxl_driver/srv/set_home_position.hpp>
#include <dxl_driver/srv/set_max_velocity.hpp>

#define TICKS_PER_ROTATION 4096                 // rad = output/TICKS_PER_ROTATION*2pi
#define MA_PER_TICK 2.69                        // mA = output * MA_PER_TICK
#define RAD_PER_SECOND_PER_TICK 0.023968667     // vel_rad/s = output * RAD_PER_SECOND_PER_TICK


#define DEVICE_NAME "/dev/ttyUSB0"
#define BAUDRATE 115200
#define PROTOCOL_VERSION 2.0

namespace DXLREGISTER
{
    uint8_t const MODEL_NUMBER      = 0x00; // Size: 2 bytes
    uint8_t const ID                = 0x02; // Size: 4 bytes
    uint8_t const BAUD_RATE         = 0x08; // Size; 1 byte
    uint8_t const RETURN_DELAY      = 0x09; // Size: 1 byte
    uint8_t const DRIVE_MODE        = 0x0A; // Size: 1 byte
    uint8_t const OPERATING_MODE    = 0x0B; // Size; 1 byte
    uint8_t const SHADOW_ID         = 0x0C; // Size: 1 byte
    uint8_t const HOMING_OFFSET     = 0x14; // Size: 4 bytes
    uint8_t const MOVING_THRESHOLD  = 0x18; // Size: 4 bytes
    uint8_t const CURRENT_LIMIT     = 0x26; // Size: 2 bytes
    uint8_t const VELOCITY_LIMIT    = 0x2C; // Size: 4 bytes
    uint8_t const MAX_POSITION      = 0x30; // Size: 4 bytes
    uint8_t const MIN_POSITION      = 0x34; // Size: 4 bytes
    uint8_t const TORQUE_ENABLE     = 0x40; // Size: 1 byte
    uint8_t const LED               = 0x41; // Size: 1 byte
    uint8_t const HARDWARE_ERROR    = 0x46; // Size: 1 byte
    uint8_t const GOAL_PWM          = 0x64; // Size: 2 bytes
    uint8_t const GOAL_CURRENT      = 0x66; // Size: 2 bytes
    uint8_t const GOAL_VELOCITY     = 0x68; // Size: 4 bytes
    uint8_t const PROFILE_ACCEL     = 0x6C; // Size: 4 bytes
    uint8_t const PROFILE_VELOCITY  = 0x70; // Size: 4 bytes
    uint8_t const GOAL_POSITION     = 0x74; // Size: 4 bytes
    uint8_t const PRESENT_PWM       = 0x7C; // Size: 2 bytes, read-only
    uint8_t const PRESENT_CURRENT   = 0x7E; // Size: 2 bytes, read-only
    uint8_t const PRESENT_VELOCITY  = 0x80; // Size: 4 bytes, read-only
    uint8_t const PRESENT_POSITION  = 0x84; // Size: 4 bytes, read-only
};

enum DXLMode{
    CURRENT             = 0, // Current control mode
    VELOCITY            = 1, // Velocity control mode
    POSITION            = 3, // Joint control mode, position within limits
    EXTENDED_POSITION   = 4, // Multi-turn position control mode
    POSITION_CURRENT    = 5, // Position/current control mode
    PWM                 = 16 // PWM control mode
};

enum UNIT
{
    COUNTS = 0,
    RAD = 1,
    DEG = 2
};

struct ServoData
{
    uint8_t id;
    double gear_ratio;
    double max_velocity; // In rad/s
    int direction;
    DXLMode operating_mode;
    double min_angle; // In rad
    double max_angle; // In rad
    double goal_position; // In rad
    double goal_velocity; // In rad/s
    double present_position;
    double present_velocity;
    double present_current;
    double start_offset; // in rad
    uint32_t present_pwm;
    int16_t home_position; // In ticks at the servo horn
};


class DXLDriver : public rclcpp::Node
{
public:

    DXLDriver(dynamixel::GroupSyncRead *positionReader, dynamixel::GroupSyncRead *velocityReader,
                dynamixel::GroupSyncRead *currentReader, dynamixel::GroupSyncRead *PWMReader, 
                dynamixel::GroupSyncWrite *positionWriter, dynamixel::GroupSyncWrite *velocityWriter);
    virtual ~DXLDriver();

    void srv_set_torque_enable_callback(const std::shared_ptr<dxl_driver::srv::SetTorqueEnable::Request> request,
                                        std::shared_ptr<dxl_driver::srv::SetTorqueEnable::Response> response);
    void srv_set_home_position_callback(const std::shared_ptr<dxl_driver::srv::SetHomePosition::Request> request,
                                        std::shared_ptr<dxl_driver::srv::SetHomePosition::Response> response);
    void srv_set_max_velocity_callback(const std::shared_ptr<dxl_driver::srv::SetMaxVelocity::Request> request,
                                        std::shared_ptr<dxl_driver::srv::SetMaxVelocity::Response> response);

private:
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_servo_reference;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_servo_state;
    rclcpp::Service<dxl_driver::srv::SetTorqueEnable>::SharedPtr set_torque_enable_srv_;
    rclcpp::Service<dxl_driver::srv::SetHomePosition>::SharedPtr set_home_position_srv_;
    rclcpp::Service<dxl_driver::srv::SetMaxVelocity>::SharedPtr set_max_velocity_srv_;

    dynamixel::GroupSyncRead *gsrPosition;
    dynamixel::GroupSyncRead *gsrVelocity;
    dynamixel::GroupSyncRead *gsrCurrent;
    dynamixel::GroupSyncRead *gsrPWM;

    dynamixel::GroupSyncWrite *gswPosition;
    dynamixel::GroupSyncWrite *gswVelocity;

    /// @brief Callback for the timer, execute interface loop
    void loop();
    void write_goal_positions();
    void write_goal_velocities();
    int read_all_servo_data();
    void servo_reference_callback(const sensor_msgs::msg::JointState::SharedPtr msg);
    void publish_all_servo_data();
    //void setup_port();
    void enable_all_torque();
    double pos_int2rad(uint8_t id, int32_t position_ticks); // return the rad position with gear ratio and direction
    int32_t pos_rad2int(uint8_t id, double position_rads);
    double vel_int2rad(uint8_t id, int32_t velocity_ticks);
    int32_t vel_rad2int(uint8_t id, double velocity_rads); 
    double cur_int2amp(uint16_t current_ticks);

    int read_present_positions();
    int read_present_velocities();
    int read_present_currents();
    int read_present_pwms();

    void configure_servos();

    bool write_home_position_at_current_position();
    bool write_torque_enable(int8_t torque_enable);
    bool write_profile_velocity(const uint8_t id);
    bool write_velocity_limit(const uint8_t id);

    void check_parameter_sizes(size_t num_servos) const;

    // Data
    rclcpp::TimerBase::SharedPtr timer_;

    std::vector<uint8_t> ids_;
    std::unordered_map<int, size_t> id2index_; 
    int num_servos_;
    std::vector<ServoData> servodata_;


};

#endif  // DXL_DRIVER_NODE_HPP_
