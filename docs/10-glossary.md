# Glossary

Project-specific terms. General ROS 2 / PX4 vocabulary is assumed.

| Term | Meaning |
|------|---------|
| **ATS** | Aerial Tactile Servoing — this project; closing a control loop on tactile feedback while airborne. |
| **UAM** | Unmanned Aerial Manipulator — the drone-plus-arm platform. |
| **TacTip** | The soft optical tactile sensor at the arm tip; a camera images a deforming marker dome. |
| **Contact frame** | Reference frame with Z perpendicular to the interaction surface; the planner defines references here. |
| **Contact pose** | The TacTip's estimate of the interaction: depth into the surface, shear (x/y), and local surface orientation. |
| **Depth** | How far the sensor is pressed into the surface (mm); the primary regulated quantity. |
| **Shear** | Lateral deformation of the sensor tangent to the surface. |
| **SSIM** | Structural similarity between the live sensor image and a stored reference image; used for contact detection. |
| **Mission director (MD)** | The finite-state-machine node that orchestrates a flight and relays setpoints. |
| **FSM state** | A phase of the mission (`takeoff`, `approach`, `tactile_servoing`, …); published numerically on `/md/state`. |
| **Servoing law / IK controller** | The weighted-pseudo-inverse inverse-kinematics controller that turns tactile error into drone + joint motion. |
| **IK weights (flight / contact)** | Per-DOF weights in the pseudo-inverse; faded from flight to contact values once contact is established. |
| **Offboard** | PX4 mode in which the companion computer commands the vehicle; required during a mission. |
| **q1 / q2 / q3** | The three arm joints (Dynamixel servos). |
| **Mission preset** | A named reference pattern the planner emits (`blockref_x`, `slide_x`, …). |
| **vbats** | Velocity-Based Aerial Tactile Servoing — the current primary controller/mission family. Outputs drone and servo velocity commands tracking a contact pose. |

<!-- TODO: add terms as new subsystems are documented. -->
