from setuptools import find_packages, setup

package_name = 'mission_director'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='orangepi',
    maintainer_email='mbrummelhuis@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'vbats=mission_director.missions.vbats:main',  # consolidated POC (TactileMissionDirector)
            'door=mission_director.missions.door:main',
            'hover=mission_director.missions.hover:main',
            'pinch_grasp=mission_director.missions.pinch_grasp:main',
            'no_mocap_waypoints=mission_director.testing.no_mocap_waypoints:main'
        ],
    },
)
