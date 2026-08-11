#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_control_mode.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <stdint.h>
#include <Eigen/Dense>
#include <cmath>

#include <chrono>
#include <iostream>

using namespace std::chrono;
using namespace std::chrono_literals;
using namespace px4_msgs::msg;

struct PIDState{
        double prev_t;
        double error;
        double error_int;
        double error_dot;
        double Kp;
        double Ki;
        double Kd;

        PIDState() : prev_t(-1), error(0.0), error_int(0.0), error_dot(0.0), Kp(0.2), Kd(0.0), Ki(0.1) { }
};

struct UAMState{
	double prev_t;
	Eigen::Vector3d pos;
	Eigen::Vector3d vel_NED_est;
	Eigen::Vector3d rpy;
	Eigen::Vector3d pos_NED;
	Eigen::Vector3d rpy_NED;
	double q_al;
	double q_ar;
	double q_ll;
	double q_lr;	

	UAMState() : prev_t(-1), pos(Eigen::Vector3d(0.0,0.0,0.0)), rpy(Eigen::Vector3d(0.0,0.0,0.0)), pos_NED(Eigen::Vector3d(0.0,0.0,0.0)), rpy_NED(Eigen::Vector3d(0.0,0.0,0.0)), q_al(0.0), q_ar(0.0), q_ll(0.0), q_lr(0.0), vel_NED_est(Eigen::Vector3d(0.0,0.0,0.0)) { }
};

struct SetpointState{
	double vx;
	double vy;
	double vz;
	double yaw_rate;
	double x;
	double y;
	double z;
	double yaw;
	int uav_mode; //0:position, 1:velocity
	int limbs_mode; //0:position, 1:velocity
	double q_al;
	double q_ar;
	double q_ll;
	double q_lr;
	double q_al_dot;
	double q_ar_dot;
	double q_ll_dot;
	double q_lr_dot;

	SetpointState() : vx(0), vy(0), vz(0), yaw_rate(0), x(0), y(0), z(0), yaw(0), uav_mode(0), limbs_mode(0), q_al(0.0), q_ar(0.0), q_ll(0.0), q_lr(0.0), q_al_dot(0.0), q_ar_dot(0.0), q_ll_dot(0.0), q_lr_dot(0.0)  { }
};

class UAMOffboardCommander : public rclcpp::Node
{
public:
	UAMOffboardCommander() : Node("offboard_commander")
	{
          HAS_VICON = false;
          
	  offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>("/fmu/in/offboard_control_mode", 10);
	  trajectory_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>("/fmu/in/trajectory_setpoint", 10);
	  vehicle_command_publisher_ = this->create_publisher<VehicleCommand>("/fmu/in/vehicle_command", 10);
	  
	  vicon_sub = this->create_subscription<nav_msgs::msg::Odometry>("/odom", 10, std::bind(&UAMOffboardCommander::vicon_callback, this, std::placeholders::_1));
          vel_cmd_sub = this->create_subscription<geometry_msgs::msg::TwistStamped>("/uam/cmd_vel", 10, std::bind(&UAMOffboardCommander::vel_cmd_callback, this, std::placeholders::_1));
          pose_cmd_sub = this->create_subscription<geometry_msgs::msg::PoseStamped>("/uam/cmd_pose", 10, std::bind(&UAMOffboardCommander::pose_cmd_callback, this, std::placeholders::_1));          

	  offboard_setpoint_counter_ = 0;
	  start_time = 0.0;
	  timer_ = this->create_wall_timer(100ms, std::bind(&UAMOffboardCommander::timer_callback, this));
	}

	void arm();
	void disarm();
	void timer_callback();
	void vicon_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
	void vel_cmd_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg);
	void pose_cmd_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);	

private:
	rclcpp::TimerBase::SharedPtr timer_;

	rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
	rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
	rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
	
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr vicon_sub;
        rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr vel_cmd_sub;        
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_cmd_sub;

        std::atomic<uint64_t> timestamp_;   //!< common synced timestamped
	uint64_t offboard_setpoint_counter_;   //!< counter for the number of setpoints sent
	double start_time;
        UAMState UAM_STATE;
        SetpointState UAM_CMD;    
        PIDState x_pid;
        PIDState y_pid;
        PIDState z_pid;
        bool HAS_VICON;
        
        void publish_offboard_control_mode();
	void publish_trajectory_setpoint();
	void publish_vehicle_command(uint16_t command, float param1 = 0.0, float param2 = 0.0);

};
//==================================================//
void UAMOffboardCommander::timer_callback()
{
  if(!(this->HAS_VICON)){
    return;
  }
  
  
  if (this->offboard_setpoint_counter_ == 10) {
    // Change to Offboard mode after 10 setpoints
    this->publish_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1, 6);
    // Arm the vehicle
    this->arm();
    rclcpp::Time t = this->get_clock()->now();
    this->start_time = t.seconds();
  }

  // offboard_control_mode needs to be paired with trajectory_setpoint
  this->publish_offboard_control_mode();
  this->publish_trajectory_setpoint();

  // stop the counter after reaching 11
  if(this->offboard_setpoint_counter_ < 11) {
    this->offboard_setpoint_counter_++;
  }
    
}

void UAMOffboardCommander::vicon_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  (UAM_STATE).pos = Eigen::Vector3d(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
  tf2::Quaternion q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
  tf2::Matrix3x3 R(q);
  double roll, pitch, yaw;
  R.getRPY(roll,pitch,yaw);
  (UAM_STATE).rpy = Eigen::Vector3d(roll, pitch, yaw);
  Eigen::Vector3d prev_pos_NED = UAM_STATE.pos_NED;
  (UAM_STATE).pos_NED = Eigen::Vector3d(msg->pose.pose.position.y, msg->pose.pose.position.x, -msg->pose.pose.position.z);
  Eigen::Quaterniond q_NED_eigen = Eigen::Quaterniond(0.7071068, 0.0, 0.0, 0.7071068)
                         * Eigen::Quaterniond(msg->pose.pose.orientation.w,
                                              msg->pose.pose.orientation.y,
                                              msg->pose.pose.orientation.x,
                                              -msg->pose.pose.orientation.z);
  tf2::Quaternion q_NED_tf2(q_NED_eigen.x(), q_NED_eigen.y(), q_NED_eigen.z(), q_NED_eigen.w());
  tf2::Matrix3x3 R_NED(q_NED_tf2);
  double roll_NED, pitch_NED, yaw_NED;
  R_NED.getRPY(roll_NED, pitch_NED, yaw_NED);
  (UAM_STATE).rpy_NED = Eigen::Vector3d(roll_NED, pitch_NED, yaw_NED);
  
  double dt;
  if(!(this->HAS_VICON)){
    this->HAS_VICON = true;
    (UAM_STATE).prev_t = this->get_clock()->now().seconds();      
    (UAM_CMD).uav_mode = 0;
    (UAM_CMD).x = (UAM_STATE).pos_NED(0);
    (UAM_CMD).y = (UAM_STATE).pos_NED(1);
    (UAM_CMD).z = (UAM_STATE).pos_NED(2) - 1.0;
    (UAM_CMD).yaw = (UAM_STATE).rpy_NED(2);    
  }else{//estimate velocities
    double dt = this->get_clock()->now().seconds() - UAM_STATE.prev_t;
    (UAM_STATE).prev_t = this->get_clock()->now().seconds(); 
    Eigen::Vector3d vel_est = (1/dt)*(UAM_STATE.pos_NED - prev_pos_NED);
    UAM_STATE.vel_NED_est = 0.3*vel_est + 0.7*UAM_STATE.vel_NED_est;
  }
}

void UAMOffboardCommander::vel_cmd_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg)
{
  if(!(this->HAS_VICON)){
    return;
  }
  UAM_CMD.uav_mode = 1;
  UAM_CMD.vx = msg->twist.linear.y;
  UAM_CMD.vy = msg->twist.linear.x;
  UAM_CMD.vz = -msg->twist.linear.z;
  UAM_CMD.yaw_rate = -msg->twist.angular.z;  
}

void UAMOffboardCommander::pose_cmd_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  if(!(this->HAS_VICON)){
    return;
  }
  UAM_CMD.uav_mode = 0;
  //
  x_pid.error =0.0;
  x_pid.error_dot = 0.0;
  x_pid.error_int = 0.0;
  y_pid.error =0.0;
  y_pid.error_dot = 0.0;
  y_pid.error_int = 0.0;
  z_pid.error =0.0;
  z_pid.error_dot = 0.0;
  z_pid.error_int = 0.0;  
  //
  
  Eigen::Vector3d cmd_pos_NED = Eigen::Vector3d(msg->pose.position.y, msg->pose.position.x, -msg->pose.position.z);
  Eigen::Quaterniond q_NED_eigen = Eigen::Quaterniond(0.7071068, 0.0, 0.0, 0.7071068)
                         * Eigen::Quaterniond(msg->pose.orientation.w,
                                              msg->pose.orientation.y,
                                              msg->pose.orientation.x,
                                              -msg->pose.orientation.z);
  tf2::Quaternion q_NED_tf2(q_NED_eigen.x(), q_NED_eigen.y(), q_NED_eigen.z(), q_NED_eigen.w());
  tf2::Matrix3x3 R_NED(q_NED_tf2);
  double roll_NED, pitch_NED, yaw_NED;
  R_NED.getRPY(roll_NED, pitch_NED, yaw_NED);
  UAM_CMD.yaw = yaw_NED;
  UAM_CMD.x = cmd_pos_NED(0);
  UAM_CMD.y = cmd_pos_NED(1);
  UAM_CMD.z = cmd_pos_NED(2);
}

/**
 * @brief Send a command to Arm the vehicle
 */
void UAMOffboardCommander::arm()
{
	publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0);

	RCLCPP_INFO(this->get_logger(), "Arm command send");
}

/**
 * @brief Send a command to Disarm the vehicle
 */
void UAMOffboardCommander::disarm()
{
	publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0);

	RCLCPP_INFO(this->get_logger(), "Disarm command send");
}

/**
 * @brief Publish the offboard control mode.
 *        For this example, only position and altitude controls are active.
 */
void UAMOffboardCommander::publish_offboard_control_mode()
{
	OffboardControlMode msg{};
//	if(UAM_CMD.uav_mode==0){
          msg.position = true;
          msg.velocity = false;
//        }else{
//          msg.position = false;
//	  msg.velocity = true;        
//        }
	msg.acceleration = false;
	msg.attitude = false;
	msg.body_rate = false;
	msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
	offboard_control_mode_publisher_->publish(msg);
}

/**
 * @brief Publish a trajectory setpoint
 *        For this example, it sends a trajectory setpoint to make the
 *        vehicle hover at 5 meters with a yaw angle of 180 degrees.
 */
void UAMOffboardCommander::publish_trajectory_setpoint()
{
        double dt;
        double vx_extra = 0.0;
        double vy_extra = 0.0;
        double vz_extra = 0.0;
        if(x_pid.prev_t<0){
          x_pid.prev_t = this->get_clock()->now().seconds();
          y_pid.prev_t = x_pid.prev_t;
          z_pid.prev_t = x_pid.prev_t;
          dt = 0.01;
        }else{
          dt = this->get_clock()->now().seconds() - x_pid.prev_t;
          x_pid.prev_t = this->get_clock()->now().seconds();
          y_pid.prev_t = x_pid.prev_t;
          z_pid.prev_t = x_pid.prev_t;
        }
        if(UAM_CMD.uav_mode==1){//velocity pids
          double x_error = UAM_STATE.vel_NED_est(0) - (this->UAM_CMD).vx;
          double x_error_dot = (1/dt)*(x_error - x_pid.error);
          x_pid.error_dot = 0.3*x_error_dot + 0.7*x_pid.error_dot;
          x_pid.error_int += 0.5*dt*(x_error + x_pid.error);
          x_pid.error = x_error;
          vx_extra = x_pid.Kp * x_pid.error + x_pid.Ki * x_pid.error_int + x_pid.Kd * x_pid.error_dot;
          double y_error = UAM_STATE.vel_NED_est(1) - (this->UAM_CMD).vy;
          double y_error_dot = (1/dt)*(y_error - y_pid.error);
          y_pid.error_dot = 0.3*y_error_dot + 0.7*y_pid.error_dot;
          y_pid.error_int += 0.5*dt*(y_error + y_pid.error);
          y_pid.error = y_error;
          vy_extra = y_pid.Kp * y_pid.error + y_pid.Ki * y_pid.error_int + y_pid.Kd * y_pid.error_dot;
          double z_error = UAM_STATE.vel_NED_est(2) - (this->UAM_CMD).vz;
          double z_error_dot = (1/dt)*(z_error - z_pid.error);
          z_pid.error_dot = 0.3*z_error_dot + 0.7*z_pid.error_dot;
          z_pid.error_int += 0.5*dt*(z_error + z_pid.error);
          z_pid.error = z_error;
          vz_extra = z_pid.Kp * z_pid.error + z_pid.Ki * z_pid.error_int + z_pid.Kd * z_pid.error_dot;          
        }


        TrajectorySetpoint msg{};
	rclcpp::Time t = this->get_clock()->now();
	double seconds = t.seconds();
	double time_since_start = seconds - start_time;
        if(UAM_CMD.uav_mode==0){
          msg.position = {(this->UAM_CMD).x, (this->UAM_CMD).y, (this->UAM_CMD).z};
    	  msg.yaw = (this->UAM_CMD).yaw; // [-PI:PI]
        }else{
          msg.position = {nan(""), nan(""), nan("")};
          msg.velocity = {(this->UAM_CMD).vx - vx_extra, (this->UAM_CMD).vy - vy_extra, (this->UAM_CMD).vz - vz_extra};
          msg.yaw = nan("");
          msg.yawspeed = (this->UAM_CMD).yaw_rate;
        }
        //std::cout << UAM_STATE.vel_NED_est.transpose() << std::endl;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
	trajectory_setpoint_publisher_->publish(msg);
}

/**
 * @brief Publish vehicle commands
 * @param command   Command code (matches VehicleCommand and MAVLink MAV_CMD codes)
 * @param param1    Command parameter 1
 * @param param2    Command parameter 2
 */
void UAMOffboardCommander::publish_vehicle_command(uint16_t command, float param1, float param2)
{
	VehicleCommand msg{};
	msg.param1 = param1;
	msg.param2 = param2;
	msg.command = command;
	msg.target_system = 1;
	msg.target_component = 1;
 	msg.source_system = 1;
	msg.source_component = 1;
	msg.from_external = true;
	msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
	vehicle_command_publisher_->publish(msg);
}

int main(int argc, char *argv[])
{
	std::cout << "Starting offboard commander node..." << std::endl;
	setvbuf(stdout, NULL, _IONBF, BUFSIZ);
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<UAMOffboardCommander>());

	rclcpp::shutdown();
	return 0;
}





