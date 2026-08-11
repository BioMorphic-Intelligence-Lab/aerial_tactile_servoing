# ATS Documentation

Documentation for the Aerial Tactile Servoing project, ordered as a reading path. If you are
new, read them in order — 01 and 02 give you the mental model and a working simulation before
anything else.

1. [Overview](01-overview.md) — what the system does and how the nodes fit together
2. [Getting started](02-getting-started.md) — install, build, run the simulation
3. [System architecture](03-system-architecture.md) — the control loop in detail
4. [Hardware setup](04-hardware-setup.md) — the physical platform
5. [Simulation](05-simulation.md) — the Gazebo model and sim-only nodes
6. [Running a mission](06-running-a-mission.md) — launch files and the flight state machine
7. [Tactile sensing](07-tactile-sensing.md) — the TacTip pipeline
8. [Data & logging](08-data-and-logging.md) — rosbags and data management
9. [Troubleshooting](09-troubleshooting.md)
10. [Glossary](10-glossary.md)

Package-level reference documentation lives next to the code, in each package's `README.md`
under `ros2_ws/src`.
