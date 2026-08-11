Package for UAM simulations with PX4 and Gazebo-ROS2
==============

You need to download and build PX4-SITL and a Micro-ROS Agent.

Go to: https://docs.px4.io/main/en/ros2/user_guide.html

There, follow the **Install PX4** and **Setup Micro XRCE-DDS Agent & Client** instructions. 

Also, download **QGroundControl** from: https://docs.qgroundcontrol.com/master/en/qgc-user-guide/getting_started/download_and_install.html

Some indicative dependencies include: ros-(distro)-gz-sim-vendor ros-(distro)-ros-gz-bridge ros-(distro)-effort-controllers ros-(distro)-mavros-msgs ros-(distro)-libmavconn

Next, clone this package to your ROS2 workspace, build with **colcon build**

The following are also necessary to run once:

      cp -r <ros_ws>/src/px4_uam_sim/urdf/meshes <ros_ws>/install/px4_uam_sim/share/px4_uam_sim
      echo 'export GZ_SIM_RESOURCE_PATH=$HOME/<ros2_ws>/src/px4_uam_sim/worlds:${GZ_SIM_RESOURCE_PATH}' >> ~/.bashrc


The provided script can be used to simplify simulation execution, but folder directories need to be customized before running.

The X500 model is from the [PX4-gazebo-models](https://github.com/PX4/PX4-gazebo-models) repository.
