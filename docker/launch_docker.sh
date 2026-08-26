# docker run -- run the docker container
# -rm -- remove the container after it is done running
# -it -- interactive terminal
# --user -- run the container as the ros2 user
# --name -- name the container
# --network -- connect the container to the host network (when set as host)
# -- ipc=host -- connect the container to the host IPC namespace
# -v -- mount a volume
# --device -- pass a device to the container, put the device path here (tactips, serial for FCU, serial for servos). Check out articulated robotics video 'devices in docker' for more info
# --privileged -- give the container full access to the host system (needed for GPIO)
# --mount -- bind a volume to the container, this is handy for active development as you can develop on the host and it gets automtically updated in the container

docker run -it --rm --name ats-container --network=host --ipc=host --device=/dev/ttyUSB0 --device=/dev/video0 --privileged --mount type=bind,src=/home/orangepi/aerial_tactile_servoing,dst=/aerial_tactile_servoing mbrummelhuis/ats-no-build