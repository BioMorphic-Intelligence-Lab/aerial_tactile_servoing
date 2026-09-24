from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'tactip_ros2_driver'

# Default model, used when a node does not set the 'model_dir' parameter (single-arm missions).
model_name = 'simple_cnn_C2_2026'

#  multiple driver instances (e.g. the dual-arm TacTips, which are physically different sensors
# with their own force limits and image processing) can each load their own model via 'model_dir'.
model_dirs = sorted(d for d in glob('resource/models/*') if os.path.isdir(d))

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/model', glob('resource/models/' + model_name + '/*.json')
                                           + glob('resource/models/' + model_name + '/*.pth')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ] + [
        ('share/' + package_name + '/models/' + os.path.basename(d),
         glob(d + '/*.json') + glob(d + '/*.pth'))
        for d in model_dirs
    ],
    install_requires=[
        'setuptools', 
        'scikit-image', 
        'scipy',
        'numpy',
        'opencv-python',
        'rclpy',
        'torch'
        ],
    zip_safe=True,
    maintainer='Martijn Brummelhuis',
    maintainer_email='mbrummelhuis@gmail.com',
    description='ROS2 interface package for TacTip optical tactile sensor',
    license='GPL-3.0-only',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'tactip_ros2_driver = tactip_ros2_driver.tactip_ros2_driver:main',
        ],
    },
)
