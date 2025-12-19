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
            'uam_md_test = mission_director.uam_md_test:main',
            'uam_control_test = mission_director.uam_control_test:main',
            'ats_mission = mission_director.ats_mission:main',
            'ymca_md=mission_director.ymca_mission_director:main',
            'vbats_mission=mission_director.vbats_mission:main',
        ],
    },
)
