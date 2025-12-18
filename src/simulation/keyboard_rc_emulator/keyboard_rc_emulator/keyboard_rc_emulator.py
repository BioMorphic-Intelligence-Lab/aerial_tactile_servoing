#!/usr/bin/env python3
"""
ROS 2 node that listens to keyboard input and publishes PX4 RCChannels messages
on /fmu/in/rc_channels, emulating an RC transmitter for simulation.

Uses pynput to get proper key press / release events (hold-to-deflect).

Dependencies:
  pip install pynput
"""
import time

import rclpy
from rclpy.node import Node

from px4_msgs.msg import RcChannels

from pynput import keyboard


class KeyboardRC(Node):
    def __init__(self):
        super().__init__('keyboard_rc')

        self.publisher_ = self.create_publisher(
            RcChannels,
            '/fmu/in/rc_channels',
            10
        )

        # Stick positions (normalized [-1, 1])
        self.roll = 0.0
        self.pitch = 0.0
        self.throttle = -1.0
        self.yaw = 0.0

        # Stick rates while key is held (units per second)
        self.roll_rate = 0.0
        self.pitch_rate = 0.0
        self.yaw_rate = 0.0
        self.throttle_rate = 0.0

        self.max_rate = 1.5        # full-scale per second
        self.throttle_rate_max = 1.0

        self.last_time = time.time()

        self.timer = self.create_timer(0.02, self.update_and_publish)  # 50 Hz

        self.get_logger().info(
            'Keyboard RC started (pynput backend).\n'
            'Hold keys for continuous stick deflection:\n'
            '  w/s : pitch forward/back\n'
            '  a/d : roll left/right\n'
            '  q/e : yaw left/right\n'
            '  r/f : throttle up/down\n'
            '  space : reset sticks\n'
            '  esc : quit'
        )

        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()

    # ----------------------------
    # Keyboard callbacks
    # ----------------------------

    def on_press(self, key):
        if key == keyboard.Key.space:
            self.reset_sticks()
            return

        try:
            k = key.char
        except AttributeError:
            return

        if k == 'w':
            self.pitch_rate = +self.max_rate
        elif k == 's':
            self.pitch_rate = -self.max_rate
        elif k == 'a':
            self.roll_rate = -self.max_rate
        elif k == 'd':
            self.roll_rate = +self.max_rate
        elif k == 'q':
            self.yaw_rate = -self.max_rate
        elif k == 'e':
            self.yaw_rate = +self.max_rate
        elif k == 'r':
            self.throttle_rate = +self.throttle_rate_max
        elif k == 'f':
            self.throttle_rate = -self.throttle_rate_max

    def on_release(self, key):
        if key == keyboard.Key.esc:
            self.get_logger().info('ESC pressed, shutting down.')
            rclpy.shutdown()
            return False

        try:
            k = key.char
        except AttributeError:
            return

        if k in ['w', 's']:
            self.pitch_rate = 0.0
        elif k in ['a', 'd']:
            self.roll_rate = 0.0
        elif k in ['q', 'e']:
            self.yaw_rate = 0.0
        elif k in ['r', 'f']:
            self.throttle_rate = 0.0

    # ----------------------------
    # RC logic
    # ----------------------------

    def reset_sticks(self):
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.throttle = -1.0

    def normalize_to_pwm(self, value):
        value = max(-1.0, min(1.0, value))
        return int(1500 + value * 500)

    def update_and_publish(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        # Integrate stick positions
        self.roll += self.roll_rate * dt
        self.pitch += self.pitch_rate * dt
        self.yaw += self.yaw_rate * dt
        self.throttle += self.throttle_rate * dt

        # Clamp
        self.roll = max(-1.0, min(1.0, self.roll))
        self.pitch = max(-1.0, min(1.0, self.pitch))
        self.yaw = max(-1.0, min(1.0, self.yaw))
        self.throttle = max(-1.0, min(1.0, self.throttle))

        msg = RcChannels()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.timestamp_last_valid = msg.timestamp
        msg.channel_count = 16

        channels = [0.0] * 16
        channels[0] = self.roll
        channels[1] = self.pitch
        channels[2] = self.throttle
        channels[3] = self.yaw

        msg.channels = channels
        msg.rssi = 100

        self.publisher_.publish(msg)



def main(args=None):
    rclpy.init(args=args)
    node = KeyboardRC()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
