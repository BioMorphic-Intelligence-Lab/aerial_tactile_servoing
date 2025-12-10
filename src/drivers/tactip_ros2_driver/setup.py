from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'tactip_ros2_driver'

model_name = 'simple_cnn_hypopt_5D'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/model/', glob('resource/models/'+model_name+'/*')),
        (os.path.join('share', package_name), glob('launch/*.launch.py')), # Add the launch files
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
