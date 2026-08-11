#!/bin/bash

session="px4-ts"
tmux new-session -d -s $session

window=0
tmux rename-window -t $session:$window -n 'simulation'
tmux split-window -h
tmux select-pane -t 0
tmux split-window -v

tmux select-pane -t 2
tmux split-window -v
#tmux select-layout even-horizontal
#tmux select-layout even-vertical


tmux select-pane -t 0
tmux send-keys -t $session:$window '~/QGroundControl.AppImage'

#window=1
#tmux new-window -t $session:$window -n 'gz_ros2'
tmux select-pane -t 1
tmux send-keys -t $session:$window 'source ~/aerial_tactile_servoing/install/setup.bash && ros2 launch ats_bringup gz_sim_one_arm.launch.py'

#window=2
#tmux new-window -t $session:$window -n 'px4_sitl'
tmux select-pane -t 2
tmux send-keys -t $session:$window 'source ~/aerial_tactile_servoing/install/setup.bash && cd ~/PX4-Autopilot && PX4_SYS_AUTOSTART=4001 PX4_SIMULATOR=GZ PX4_GZ_MODEL_NAME=my_custom_model ./build/px4_sitl_default/bin/px4'

tmux attach-session -t $session
