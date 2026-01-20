#include <math.h>
#include <cstdint>

#include "dxl_driver.hpp"

dynamixel::PortHandler * portHandler;
dynamixel::PacketHandler * packetHandler;

// dynamixel::GroupSyncRead *gsrPosition;
// dynamixel::GroupSyncRead *gsrVelocity;
// dynamixel::GroupSyncRead *gsrCurrent;
// dynamixel::GroupSyncRead *gsrPWM;

uint8_t dxl_error = 0;
uint32_t goal_position = 0;
int dxl_comm_result = COMM_TX_FAIL;

DXLDriver::DXLDriver(dynamixel::GroupSyncRead *positionReader, dynamixel::GroupSyncRead *velocityReader,
    dynamixel::GroupSyncRead *currentReader, dynamixel::GroupSyncRead *PWMReader, 
    dynamixel::GroupSyncWrite *positionWriter, dynamixel::GroupSyncWrite *velocityWriter)
: Node("dxl_driver")
{
    gsrPosition = positionReader;
    gsrVelocity = velocityReader;
    gsrCurrent = currentReader;
    gsrPWM = PWMReader;

    gswPosition = positionWriter;
    gswVelocity = velocityWriter;

    // Declare all parameters
    this->declare_parameter("driver.frequency", 1.);
    this->declare_parameter("servos.ids", std::vector<int>{1});
    this->declare_parameter("servos.operating_modes", std::vector<int>{4});
    this->declare_parameter("servos.homing_modes", std::vector<int>{0});
    this->declare_parameter("servos.directions", std::vector<int>{1});
    this->declare_parameter("servos.max_speeds", std::vector<double>{0.1});
    this->declare_parameter("servos.gear_ratios", std::vector<double>{1.0});
    this->declare_parameter("servos.min_angles", std::vector<double>(0.0));
    this->declare_parameter("servos.max_angles", std::vector<double>(0.0));
    this->declare_parameter("servos.start_offsets", std::vector<double>(0.0));

    // Generate uint8_t vector of ids
    std::vector<int64_t> int_ids = this->get_parameter("servos.ids").as_integer_array();
    ids_.resize(int_ids.size());
    std::transform(int_ids.begin(), int_ids.end(), ids_.begin(),
                    [](int val) { return static_cast<uint8_t>(val); });
    num_servos_ = static_cast<int>(int_ids.size());
    check_parameter_sizes(num_servos_); // Check if param sizes match

    // Assign the servo settings
    std::vector<int64_t> operating_modes = this->get_parameter("servos.operating_modes").as_integer_array();
    std::vector<int64_t> directions = this->get_parameter("servos.directions").as_integer_array();
    std::vector<double> gear_ratios = this->get_parameter("servos.gear_ratios").as_double_array();
    std::vector<int64_t> homing_modes = this->get_parameter("servos.homing_modes").as_integer_array();
    std::vector<double> max_speeds = this->get_parameter("servos.max_speeds").as_double_array();
    std::vector<double> min_angles = this->get_parameter("servos.min_angles").as_double_array();
    std::vector<double> max_angles = this->get_parameter("servos.max_angles").as_double_array();
    std::vector<double> start_offsets = this->get_parameter("servos.start_offsets").as_double_array();

    for (int i = 0; i < num_servos_; i++)
    {
        id2index_[ids_[i]] = i;

        ServoData current_servo_data;
        current_servo_data.operating_mode = static_cast<DXLMode>(operating_modes[i]);
        current_servo_data.direction = static_cast<int>(directions[i]);
        current_servo_data.gear_ratio = gear_ratios[i];
        current_servo_data.min_angle = min_angles[i];
        current_servo_data.max_angle = max_angles[i];
        current_servo_data.goal_position = 0.0;
        current_servo_data.goal_velocity = 0.0;
        current_servo_data.max_velocity = max_speeds[i];
        current_servo_data.start_offset = start_offsets[i];
        servodata_.push_back(current_servo_data);

        int dxl_addparam_result = false;
        dxl_addparam_result = gsrPosition->addParam(ids_[i]);
        if (dxl_addparam_result != true) {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Failed to addparam to groupSyncRead Position for Dynamixel ID %d", servodata_[i].id);
        }
        dxl_addparam_result = gsrVelocity->addParam(ids_[i]);
        if (dxl_addparam_result != true) {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Failed to addparam to groupSyncRead Velocity for Dynamixel ID %d", servodata_[i].id);
        }
        dxl_addparam_result = gsrCurrent->addParam(ids_[i]);
        if (dxl_addparam_result != true) {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Failed to addparam to groupSyncRead Current for Dynamixel ID %d", servodata_[i].id);
        }
        dxl_addparam_result = gsrPWM->addParam(ids_[i]);
        if (dxl_addparam_result != true) {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Failed to addparam to groupSyncRead PWMS for Dynamixel ID %d", servodata_[i].id);
        }
    }
    RCLCPP_INFO(this->get_logger(), "Succeeded assigning servodata");

    // ROS2 servo interfaces
    sub_servo_reference = this->create_subscription<sensor_msgs::msg::JointState>(
      "/servo/in/state", 10, 
      std::bind(&DXLDriver::servo_reference_callback, this, std::placeholders::_1));
    pub_servo_state = this->create_publisher<sensor_msgs::msg::JointState>("/servo/out/state", 10);

    set_torque_enable_srv_ = this->create_service<dxl_driver::srv::SetTorqueEnable>(
        "/set_torque_enable", std::bind(&DXLDriver::srv_set_torque_enable_callback,
                this, std::placeholders::_1, std::placeholders::_2));
    set_home_position_srv_ = this->create_service<dxl_driver::srv::SetHomePosition>(
        "/set_home_position", std::bind(&DXLDriver::srv_set_home_position_callback,
                this, std::placeholders::_1, std::placeholders::_2));
    set_max_velocity_srv_ = this->create_service<dxl_driver::srv::SetMaxVelocity>(
        "/set_max_velocity", std::bind(&DXLDriver::srv_set_max_velocity_callback,
                this, std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "Succeeded creating ROS2 interfaces");

    // Set max (profile) velocities and operating mode
    configure_servos();
    RCLCPP_INFO(this->get_logger(), "Succeeded configuring servos");

    // Read current positions and set as reference
    int result = read_all_servo_data();
    if (result != 0)
    {
        RCLCPP_ERROR(this->get_logger(), "Failed initial data read, quitting driver for safety");
        exit(-13);
    }
    for (int i = 0; i < num_servos_; i++)
    {
        if ((!(servodata_[i].min_angle == 0.0 && servodata_[i].max_angle == 0.0)) && 
            (servodata_[i].present_position < servodata_[i].min_angle || servodata_[i].present_position > servodata_[i].max_angle))
        {
            RCLCPP_ERROR(this->get_logger(), "[ID: %d] Present position %.3f out of position limits (MIN: %.3f, MAX: %.3f). Exiting.", 
                servodata_[i].id,
                servodata_[i].present_position,
                servodata_[i].min_angle,
                servodata_[i].max_angle);
            exit(-13);
        }
        servodata_[i].goal_position = servodata_[i].present_position;
        servodata_[i].goal_velocity = servodata_[i].present_velocity;
        RCLCPP_INFO(this->get_logger(), "[ID: %d] Present position %.3f, goal position %.3f",
            servodata_[i].id,
            servodata_[i].present_position, 
            servodata_[i].goal_position);
    }

    // Set up the servos and engage torque
    enable_all_torque(); 

    // Timer for publishing servo state
    double node_frequency_ = this->get_parameter("driver.frequency").as_double();
    timer_ = this->create_wall_timer(std::chrono::milliseconds(int(1000./node_frequency_)), std::bind(&DXLDriver::loop, this));
}

DXLDriver::~DXLDriver()
{
    // Disable Torque of DYNAMIXEL
    packetHandler->write1ByteTxRx(
        portHandler,
        BROADCAST_ID,
        DXLREGISTER::TORQUE_ENABLE,
        0,
        &dxl_error
    );
}

void DXLDriver::loop()
{
    // WARNING: current implementation causes setting of the first servo to be applied for all servos
    if (servodata_[0].operating_mode == DXLMode::EXTENDED_POSITION)
    {
        write_goal_positions(); // set servo references on servo
    }
    else if (servodata_[0].operating_mode == DXLMode::VELOCITY)
    {
        write_goal_velocities();
    }
    read_all_servo_data(); // get data from the servos
    publish_all_servo_data(); // publish data to ROS topic
}

void DXLDriver::write_goal_positions()
{
    for(int i = 0; i<num_servos_; i++)
    {
        int dxl_addparam_result = false;
        uint8_t param_goal_position[4];
        double goal_position;
        if (!(servodata_[i].min_angle == 0.0 && servodata_[i].max_angle == 0.0))
        {
            goal_position = std::clamp(servodata_[i].goal_position, servodata_[i].min_angle, servodata_[i].max_angle);
        }
        else
        {
            goal_position = servodata_[i].goal_position;
        }
        int32_t goal_pos_ticks = pos_rad2int(servodata_[i].id, goal_position);
        // RCLCPP_INFO(this->get_logger(), "goal position %f", goal_position);
        // RCLCPP_INFO(this->get_logger(), "goal position ticks %i", goal_pos_ticks);

        param_goal_position[0] = DXL_LOBYTE(DXL_LOWORD(goal_pos_ticks));
        param_goal_position[1] = DXL_HIBYTE(DXL_LOWORD(goal_pos_ticks));
        param_goal_position[2] = DXL_LOBYTE(DXL_HIWORD(goal_pos_ticks));
        param_goal_position[3] = DXL_HIBYTE(DXL_HIWORD(goal_pos_ticks));
        dxl_addparam_result = gswPosition->addParam(ids_[i], param_goal_position);
        if (dxl_addparam_result != true) 
        {
            RCLCPP_ERROR(this->get_logger(), "[ID: %d] Failed to addparam to groupSyncWrite with error code %d", 
                servodata_[i].id, 
                dxl_addparam_result
                );
        }
    }

    dxl_comm_result = gswPosition->txPacket();
    if (dxl_comm_result != COMM_SUCCESS) {
        RCLCPP_INFO(this->get_logger(), "%s", packetHandler->getTxRxResult(dxl_comm_result));
    } else if (dxl_error != 0) {
        RCLCPP_INFO(this->get_logger(), "%s", packetHandler->getRxPacketError(dxl_error));
    }
    gswPosition->clearParam(); 
}

void DXLDriver::write_goal_velocities()
{
    for(int i = 0; i<num_servos_; i++)
    {
        int dxl_addparam_result = false;
        uint8_t param_goal_velocity[4];
        double goal_velocity;
        double max_velocity;
        if (!(servodata_[i].min_angle == 0.0 && servodata_[i].max_angle == 0.0))
        {
            // Calculate max velocity based on distance to angle limit
            max_velocity = std::min(std::min(5.*servodata_[i].max_velocity*std::max(servodata_[i].present_position - servodata_[i].min_angle, 0.0), 
                5.*servodata_[i].max_velocity*std::max(servodata_[i].max_angle-servodata_[i].present_position, 0.0)),
                servodata_[i].max_velocity);
            goal_velocity = servodata_[i].goal_velocity;
        }
        else
        {
            max_velocity = servodata_[i].max_velocity;
            goal_velocity = servodata_[i].goal_velocity;
        }
        int32_t goal_vel_ticks = vel_rad2int(servodata_[i].id, std::clamp(goal_velocity, -max_velocity, max_velocity)); // Deal with negative velocities
        // RCLCPP_INFO(this->get_logger(), "[ID: %d] goal vel %f", servodata_[i].id, goal_velocity);
        // RCLCPP_INFO(this->get_logger(), "[ID: %d] goal vel ticks %i", servodata_[i].id, goal_vel_ticks);

        param_goal_velocity[0] = DXL_LOBYTE(DXL_LOWORD(goal_vel_ticks));
        param_goal_velocity[1] = DXL_HIBYTE(DXL_LOWORD(goal_vel_ticks));
        param_goal_velocity[2] = DXL_LOBYTE(DXL_HIWORD(goal_vel_ticks));
        param_goal_velocity[3] = DXL_HIBYTE(DXL_HIWORD(goal_vel_ticks));
        dxl_addparam_result = gswVelocity->addParam(ids_[i], param_goal_velocity);
        if (dxl_addparam_result != true) 
        {
            RCLCPP_ERROR(this->get_logger(), "[ID: %d] Failed to addparam to groupSyncWrite with error code %d", 
                servodata_[i].id, 
                dxl_addparam_result
                );
        }
    }
    dxl_comm_result = gswVelocity->txPacket();
    if (dxl_comm_result != COMM_SUCCESS) {
        RCLCPP_INFO(this->get_logger(), "%s", packetHandler->getTxRxResult(dxl_comm_result));
    } else if (dxl_error != 0) {
        RCLCPP_INFO(this->get_logger(), "%s", packetHandler->getRxPacketError(dxl_error));
    }
    gswVelocity->clearParam(); 
}

int DXLDriver::read_all_servo_data()
{
    int res = 0;
    res += read_present_positions();
    res += read_present_velocities();
    res += read_present_currents();
    res += read_present_pwms();
    return res;
}

void DXLDriver::publish_all_servo_data()
{
    std::vector<double> present_positions(num_servos_);
    std::vector<double> present_velocities(num_servos_);
    std::vector<double> present_currents(num_servos_);
    std::vector<std::string> names(num_servos_);
    for (int i=0; i<num_servos_; i++)
    {
        present_positions[i] = servodata_[i].present_position;
        present_velocities[i] = servodata_[i].present_velocity;
        present_currents[i] = servodata_[i].present_current;
        names[i] = "q" + std::to_string(i);
    }

    auto servo_state_msg = sensor_msgs::msg::JointState();
    servo_state_msg.header.stamp = this->get_clock()->now();
    servo_state_msg.name = names;
    servo_state_msg.position = present_positions; // In rad at output shaft
    servo_state_msg.velocity = present_velocities; // In rad/s at output shaft
    servo_state_msg.effort = present_currents; // In mA
    this->pub_servo_state->publish(servo_state_msg);
}

int DXLDriver::read_present_positions()
{
    dxl_comm_result = gsrPosition->txRxPacket();
    if (dxl_comm_result == COMM_SUCCESS)
    {
        for (int i=0; i<num_servos_; i++)
        {
            int32_t present_position_ticks = gsrPosition->getData(ids_[i], DXLREGISTER::PRESENT_POSITION, 4);
            servodata_[i].present_position = pos_int2rad(servodata_[i].id, present_position_ticks);
            // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
            //     "[ID: %i] Present position ticks %d, present position rads %.3f", 
            //     servodata_[i].id, 
            //     present_position_ticks, 
            //     servodata_[i].present_position);
        }
    }
    else
    {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Failed to get present positions with GroupSyncRead. Error code %d", dxl_comm_result);
    }
    return dxl_comm_result;
}

int DXLDriver::read_present_velocities()
{
    dxl_comm_result = gsrVelocity->txRxPacket();
    if (dxl_comm_result == COMM_SUCCESS)
    {
        for (int i=0; i<num_servos_; i++)
        {
            servodata_[i].present_velocity = vel_int2rad(servodata_[i].id, gsrVelocity->getData(ids_[i], DXLREGISTER::PRESENT_VELOCITY, 4));
        }
    }
    else
    {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Failed to get present velocities with GroupSyncRead. Error code %d", dxl_comm_result);
    }
    return dxl_comm_result;
}

int DXLDriver::read_present_currents()
{
    dxl_comm_result = gsrCurrent->txRxPacket();
    if (dxl_comm_result == COMM_SUCCESS)
    {
        for (int i=0; i<num_servos_; i++)
        {
            servodata_[i].present_current = cur_int2amp(gsrCurrent->getData(ids_[i], DXLREGISTER::PRESENT_CURRENT, 4));
        }
    }
    else
    {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Failed to get present currents with GroupSyncRead. Error code %d", dxl_comm_result);
    }
    return dxl_comm_result;
}

int DXLDriver::read_present_pwms()
{
    dxl_comm_result = gsrPWM->txRxPacket();
    if (dxl_comm_result == COMM_SUCCESS)
    {
        for (int i=0; i<num_servos_; i++)
        {
            servodata_[i].present_pwm = gsrPWM->getData(ids_[i], DXLREGISTER::PRESENT_PWM, 4);
        }
    }
    else
    {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Failed to get present pwms with GroupSyncRead. Error code %d", dxl_comm_result);
    }
    return dxl_comm_result;
}

void DXLDriver::configure_servos()
{
    for (int i = 0; i < num_servos_; i++)
    {
        // Write operating mode 4 to use velocity-based profiles
        // Profile velocity resets to 0 when writing mode.
        dxl_comm_result = packetHandler->write1ByteTxRx(
            portHandler,
            servodata_[i].id,
            DXLREGISTER::OPERATING_MODE,
            servodata_[i].operating_mode,
            &dxl_error
        );
        if (dxl_comm_result == COMM_SUCCESS)
        {
            if (servodata_[i].operating_mode == DXLMode::EXTENDED_POSITION)
            {
                RCLCPP_INFO(this->get_logger(), "[ID: %i] Set extended position operating mode", 
                    static_cast<int>(servodata_[i].id));
            }
            else if (servodata_[i].operating_mode == DXLMode::VELOCITY)
            {
                RCLCPP_INFO(this->get_logger(), "[ID: %i] Set velocity operating mode", 
                    static_cast<int>(servodata_[i].id));
            }
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "[ID: %i] Failed to set operating mode, result %i, error %i", servodata_[i].id, dxl_comm_result, dxl_error);
        }
        
        // Write maximum profile velocity
        if (!write_profile_velocity(servodata_[i].id))
        {
            RCLCPP_ERROR(this->get_logger(), "[ID: %i] Failed to set max profile velocity", servodata_[i].id);
        }
        if (!write_velocity_limit(servodata_[i].id))
        {
            RCLCPP_ERROR(this->get_logger(), "[ID: %i] Failed to set velocity limit", servodata_[i].id);
        }
    }

}

// Writes max velocity from the servodata_
bool DXLDriver::write_profile_velocity(const uint8_t id)
{
    int i = id2index_[id];
    uint32_t profile_velocity = static_cast<uint32_t>(std::abs(vel_rad2int(servodata_[i].id, servodata_[i].max_velocity)));
    dxl_comm_result = packetHandler->write4ByteTxRx(
        portHandler,
        servodata_[i].id,
        DXLREGISTER::PROFILE_VELOCITY,
        profile_velocity,
        &dxl_error
    );
    if (dxl_comm_result == COMM_SUCCESS)
    {
        RCLCPP_INFO(this->get_logger(), "[ID: %i] Set max velocity %f rad/s | %d ticks", 
            static_cast<int>(servodata_[i].id), 
            servodata_[i].max_velocity, 
            profile_velocity);
    }
    else
    {
        RCLCPP_ERROR(this->get_logger(), "[ID: %i] Failed to set max profile velocity, result %i, error %i", servodata_[i].id, dxl_comm_result, dxl_error);
        return false;
    }

    // Read back the maximum profile velocity
    uint32_t backread_velocity;
    dxl_comm_result = packetHandler->read4ByteTxRx(
        portHandler,
        servodata_[i].id,
        DXLREGISTER::PROFILE_VELOCITY,
        &backread_velocity,
        &dxl_error
    );
    if (dxl_comm_result == COMM_SUCCESS)
    {
        //RCLCPP_INFO(this->get_logger(), "[ID: %i] Backread max velocity %d ticks", servodata_[i].id, backread_velocity);
        if (backread_velocity == profile_velocity)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    return false;
}

bool DXLDriver::write_velocity_limit(const uint8_t id)
{
    int i = id2index_[id];
    uint32_t velocity_limit = static_cast<uint32_t>(std::abs(vel_rad2int(servodata_[i].id, servodata_[i].max_velocity))+1);
    dxl_comm_result = packetHandler->write4ByteTxRx(
        portHandler,
        servodata_[i].id,
        DXLREGISTER::VELOCITY_LIMIT,
        velocity_limit,
        &dxl_error
    );
    if (dxl_comm_result == COMM_SUCCESS)
    {
        RCLCPP_INFO(this->get_logger(), "[ID: %i] Set velocity limit %f rad/s | %d ticks", 
            static_cast<int>(servodata_[i].id), 
            servodata_[i].max_velocity, 
            velocity_limit);
        return true;
    }
    else
    {
        RCLCPP_ERROR(this->get_logger(), "[ID: %i] Failed to set velocity limit, result %i, error %i", servodata_[i].id, dxl_comm_result, dxl_error);
        return false;
    }

    // Read back the maximum profile velocity
    uint32_t backread_velocity;
    dxl_comm_result = packetHandler->read4ByteTxRx(
        portHandler,
        servodata_[i].id,
        DXLREGISTER::VELOCITY_LIMIT,
        &backread_velocity,
        &dxl_error
    );
    if (dxl_comm_result == COMM_SUCCESS)
    {
        //RCLCPP_INFO(this->get_logger(), "[ID: %i] Backread max velocity %d ticks", servodata_[i].id, backread_velocity);
        if (backread_velocity+1 == velocity_limit)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    return false;
}

bool DXLDriver::write_home_position_at_current_position()
{
    RCLCPP_INFO(this->get_logger(), "[SERVICE] Writing home position at the current position");
    // Update servo current positions
    int32_t homing_offset;
    bool success = true;
    read_present_positions();

    for (int i = 0; i < num_servos_; i++)
    {
        // Read current homing offset
        dxl_comm_result = packetHandler->read4ByteTxRx(
            portHandler,
            servodata_[i].id,
            DXLREGISTER::HOMING_OFFSET,
            reinterpret_cast<uint32_t *>(&homing_offset),
            &dxl_error
        );
        if (dxl_comm_result!=COMM_SUCCESS)
        {
            RCLCPP_ERROR(this->get_logger(), "[ID: %i] Read error, comm result %i, error %i", servodata_[i].id, dxl_comm_result, dxl_error);
            return false; // Exit if cannot read the current homing offset
        }

        // Update homing offset with current position
        int32_t new_homing_offset = -(pos_rad2int(servodata_[i].id, servodata_[i].present_position) - homing_offset - pos_rad2int(servodata_[i].id, servodata_[i].start_offset));
        RCLCPP_INFO(this->get_logger(), "[ID: %i] Home position was %d, new home position %d", servodata_[i].id, homing_offset, new_homing_offset);
        dxl_comm_result = packetHandler->write4ByteTxRx(
            portHandler,
            servodata_[i].id,
            DXLREGISTER::HOMING_OFFSET,
            static_cast<uint32_t>(new_homing_offset),
            &dxl_error
        );
        if (dxl_comm_result == COMM_SUCCESS)
        {
            servodata_[i].present_position = servodata_[i].start_offset;
            servodata_[i].goal_position = servodata_[i].start_offset;
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "[ID: %i] Failed to set homing offset, result %i, error %i", servodata_[i].id, dxl_comm_result, dxl_error);
            success = false;
        }
    }
    return success;
}

bool DXLDriver::write_torque_enable(int8_t torque_enable)
{

    // TODO: Set current position as goal position?
    bool success = true;
    for (int i = 0; i < num_servos_; i++)
    {
        // Enable Torque of DYNAMIXEL
        dxl_comm_result = packetHandler->write1ByteTxRx(
            portHandler,
            servodata_[i].id,
            DXLREGISTER::TORQUE_ENABLE,
            torque_enable,
            &dxl_error
        );
        if (dxl_comm_result == COMM_SUCCESS)
        {
            RCLCPP_INFO(this->get_logger(), "[ID: %i] Set torque enable %i", servodata_[i].id, torque_enable);
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "[ID: %i] Failed to set torque enable, result %i, error %i", servodata_[i].id, dxl_comm_result, dxl_error);
            success = false;
        }
    }
    return success;
}

double DXLDriver::pos_int2rad(uint8_t id, int32_t position_ticks)
{
    int i = id2index_[id];
    double position_rads = static_cast<double>(servodata_[i].direction) * static_cast<double>(position_ticks) / servodata_[i].gear_ratio / 4096.f * 2.f * M_PI;
    return position_rads;
}

int32_t DXLDriver::pos_rad2int(uint8_t id, double position_rads)
{
    int i = id2index_[id];
    int32_t position_ticks = static_cast<int32_t>(servodata_[i].direction) * static_cast<int32_t>(position_rads * servodata_[i].gear_ratio * 4096 / (2.f*M_PI));
    return position_ticks;
}

double DXLDriver::vel_int2rad(uint8_t id, int32_t velocity_ticks)
{
    int i = id2index_[id];
    return static_cast<double>(servodata_[i].direction * velocity_ticks * RAD_PER_SECOND_PER_TICK / servodata_[id2index_[id]].gear_ratio);
}

int32_t DXLDriver::vel_rad2int(uint8_t id, double velocity_rads)
{
    int i = id2index_[id];
    return static_cast<int32_t>(servodata_[i].direction * velocity_rads / RAD_PER_SECOND_PER_TICK * servodata_[id2index_[id]].gear_ratio);
}

double DXLDriver::cur_int2amp(uint16_t current_ticks)
{
    return static_cast<double>(current_ticks*MA_PER_TICK);
}

void DXLDriver::servo_reference_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    for (int i = 0; i < num_servos_; i++)
    {
        if (servodata_[i].operating_mode == DXLMode::EXTENDED_POSITION)
        {
            servodata_[i].goal_position = msg->position[i];
        }
        else if (servodata_[i].operating_mode == DXLMode::VELOCITY)
        {
            servodata_[i].goal_velocity = msg->velocity[i];
        }
    }
}

void DXLDriver::srv_set_torque_enable_callback(
    const std::shared_ptr<dxl_driver::srv::SetTorqueEnable::Request> request,
    std::shared_ptr<dxl_driver::srv::SetTorqueEnable::Response> response)
{
    RCLCPP_INFO(this->get_logger(), "[SERVICE] Setting torque enable to %d", request->set_torque_enable);
    bool success;
    success = write_torque_enable(request->set_torque_enable);
    response->success = success;
}

void DXLDriver::srv_set_home_position_callback(
    const std::shared_ptr<dxl_driver::srv::SetHomePosition::Request> request,
    std::shared_ptr<dxl_driver::srv::SetHomePosition::Response> response)
{
    if (request->set_home_position)
    {
        bool success;
        success = write_home_position_at_current_position();
        response->success = success;
    }
}

void DXLDriver::srv_set_max_velocity_callback(
    const std::shared_ptr<dxl_driver::srv::SetMaxVelocity::Request> request,
    std::shared_ptr<dxl_driver::srv::SetMaxVelocity::Response> response)
{
    RCLCPP_INFO(this->get_logger(), "[SERVICE] Setting max velocity to %f", request->max_velocity);
    bool success;
    response->success = true;
    if (request->max_velocity != 0)
    {
        for (int i = 0; i < num_servos_; i++)
        {
            servodata_[i].max_velocity = request->max_velocity;
            success = write_profile_velocity(servodata_[i].id) && write_velocity_limit(servodata_[i].id);
            if (success == false){
                response->success = false;
            }
        }
    }
    else{
        response->success = false;
    }
}

void DXLDriver::enable_all_torque()
{
    // Enable Torque of DYNAMIXEL
    dxl_comm_result = packetHandler->write1ByteTxRx(
        portHandler,
        BROADCAST_ID,
        DXLREGISTER::TORQUE_ENABLE,
        1,
        &dxl_error
    );

    if (dxl_comm_result != COMM_SUCCESS) {
        RCLCPP_ERROR(rclcpp::get_logger("dxl_driver"), "[ID: %d] Failed to enable torque.", BROADCAST_ID);
    } else {
        RCLCPP_INFO(rclcpp::get_logger("dxl_driver"), "[ID: %d] Succeeded to enable torque.", BROADCAST_ID);
    }
}

void DXLDriver::check_parameter_sizes(size_t num_servos) const
{
    if (this->get_parameter("servos.operating_modes").as_integer_array().size() != num_servos)
    {
        RCLCPP_ERROR(this->get_logger(), "servos.operating_modes not the same size as number of servos!");
        exit(-1);
    }
    if (this->get_parameter("servos.homing_modes").as_integer_array().size() != num_servos)
    {
        RCLCPP_ERROR(this->get_logger(), "servos.homing_modes not the same size as number of servos!");
        exit(-1);
    }
    if (this->get_parameter("servos.directions").as_integer_array().size() != num_servos)
    {
        RCLCPP_ERROR(this->get_logger(), "servos.directions not the same size as number of servos!");
        exit(-1);
    }
    if (this->get_parameter("servos.max_speeds").as_double_array().size() != num_servos)
    {
        RCLCPP_ERROR(this->get_logger(), "servos.max_speeds not the same size as number of servos!");
        exit(-1);
    }
    if (this->get_parameter("servos.gear_ratios").as_double_array().size() != num_servos)
    {
        RCLCPP_ERROR(this->get_logger(), "servos.gear_ratios not the same size as number of servos!");
        exit(-1);
    }
    if (this->get_parameter("servos.min_angles").as_double_array().size() != num_servos)
    {
        RCLCPP_ERROR(this->get_logger(), "servos.min_angles not the same size as number of servos!");
        exit(-1);
    }
    if (this->get_parameter("servos.max_angles").as_double_array().size() != num_servos)
    {
        RCLCPP_ERROR(this->get_logger(), "servos.max_angles not the same size as number of servos!");
        exit(-1);
    }
    if (this->get_parameter("servos.start_offsets").as_double_array().size() != num_servos)
    {
        RCLCPP_ERROR(this->get_logger(), "servos.start_offsets not the same size as number of servos!");
        exit(-1);
    }
}

void setup_port()
{
    dxl_comm_result = portHandler->openPort();
    if (dxl_comm_result == false) {
        RCLCPP_ERROR(rclcpp::get_logger("dxl_driver"), "Failed to open the port!");
        exit(-12);
    } else {
        RCLCPP_INFO(rclcpp::get_logger("dxl_driver"), "Succeeded to open the port.");
    }

    // Set the baudrate of the serial port (use DYNAMIXEL Baudrate)
    dxl_comm_result = portHandler->setBaudRate(BAUDRATE);
    if (dxl_comm_result == false) {
        RCLCPP_ERROR(rclcpp::get_logger("dxl_driver"), "Failed to set the baudrate!");
    } else {
        RCLCPP_INFO(rclcpp::get_logger("dxl_driver"), "Succeeded to set the baudrate.");
    } 
}

int main(int argc, char * argv[])
{
    portHandler = dynamixel::PortHandler::getPortHandler(DEVICE_NAME);
    packetHandler = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

    dynamixel::GroupSyncRead gsrPosition(portHandler, packetHandler, DXLREGISTER::PRESENT_POSITION, 4);
    dynamixel::GroupSyncRead gsrVelocity(portHandler, packetHandler, DXLREGISTER::PRESENT_VELOCITY, 4);
    dynamixel::GroupSyncRead gsrCurrent(portHandler, packetHandler, DXLREGISTER::PRESENT_CURRENT, 2);
    dynamixel::GroupSyncRead gsrPWM(portHandler, packetHandler, DXLREGISTER::PRESENT_PWM, 2);
    
    dynamixel::GroupSyncWrite gswPosition(portHandler, packetHandler, DXLREGISTER::GOAL_POSITION, 4);
    dynamixel::GroupSyncWrite gswVelocity(portHandler, packetHandler, DXLREGISTER::GOAL_VELOCITY, 4);

    setup_port();

    rclcpp::init(argc, argv);

    auto dxl_driver = std::make_shared<DXLDriver>(
        &gsrPosition,
        &gsrVelocity,
        &gsrCurrent,
        &gsrPWM,
        &gswPosition,
        &gswVelocity);
    rclcpp::spin(dxl_driver);
    rclcpp::shutdown();

    return 0;
}
