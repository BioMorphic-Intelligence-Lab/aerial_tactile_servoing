# System architecture

> **Status: scaffold.** Headings and known facts are in place; sections marked TODO need to be
> filled in.

This page explains *how* tactile error becomes drone and arm motion — the detail behind the
data-flow diagram in the [Overview](01-overview.md).

## Coordinate frames

The system reasons in several frames: the PX4 local NED/ENU frame, the drone body frame, the arm
joint frames, the TacTip sensor frame, and the **contact frame** (Z perpendicular to the
interaction surface). The planner defines references in the contact frame.

<!-- TODO: define each frame precisely, with a figure. Include the verified TacTip frame
     directions (see commit "Verified tactip frame directions") and the ats_drone.urdf /
     martijn.urdf kinematic parameters. -->
<!-- TODO: document link lengths. Note that L_1/L_2/L_3 differ between velocity_based_ats.py
     (0.12025 / 0.336 / 0.327) and manipulator_controller.py (0.110 / 0.330 / 0.273) — explain
     which is authoritative for the current hardware. -->

## The manipulator

Three revolute joints (`q1`, `q2`, `q3`) driven by Dynamixel servos. Forward kinematics place
the TacTip in the body frame.

<!-- TODO: joint diagram, DH-style parameters, joint limits (from dxl_ros2_vbats.yaml:
     directions, gear_ratios, min/max angles, start_offsets). -->

## Inverse kinematics and the servoing law

The velocity-based controller (`velocity_based_ats`) computes a body twist and joint velocities
from the tactile pose error using a **weighted pseudo-inverse of the Jacobian**, with a
secondary null-space objective that pulls the arm toward a nominal configuration.

Known ingredients (from the node parameters):
- PID gains on **depth**, **shear**, and **angular** error.
- Separate IK weight vectors for **flight** and **contact**, with a timed fade between them once
  contact is established (`fade_time_ik_weights`).
- A nominal joint configuration for the null-space objective.
- Anti-windup clipping and a low-pass filter (`alpha`) on the error.

<!-- TODO: write out the control law: error definition, Jacobian, weighted pseudo-inverse,
     null-space projection, and how body twist vs. joint velocity authority is split. -->
<!-- TODO: explain the flight→contact IK weight fade and why q1/q3 weights are ramped
     (see commits "Added fading of the IK weights", "Ramp q1 and q3 ik weights after contact"). -->
<!-- TODO: contrast velocity_based_ats vs pose_based_ats — when each is used. -->

## Contact detection

Contact is decided in the TacTip driver by comparing the live processed image to a stored
reference image using **structural similarity (SSIM)**; below a threshold, contact is asserted.
The reference is (re)set via a `SetBool` service (`set_ssim_ref`) that the mission director calls
before approach.

<!-- TODO: detail the SSIM pipeline, threshold tuning (ssim_contact_threshold), and the
     zero_when_no_contact behavior. Cross-link to 07-tactile-sensing.md. -->

## Setpoint relaying and authority

The controller publishes to `/controller/out/*`; the mission director relays these to
`/fmu/in/trajectory_setpoint` and `/servo/in/state` only in the `tactile_servoing` state. See
[Running a mission](06-running-a-mission.md) for the full state machine.

<!-- TODO: document the offboard control mode publishing cadence and PX4's offboard rate
     requirement, and how px4_ros2_driver guarantees it. -->

## Wrench observation

`wrench_observer` estimates the external contact wrench from IMU (`/fmu/out/sensor_combined`),
motor commands (`/fmu/out/actuator_motors`), and joint states.

<!-- TODO: document the observer model and outputs (/contact/out/*). State whether it is used in
     the control loop or for logging/analysis only. -->
