from setuptools import find_packages, setup

package_name = 'vehicle_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your@email.com',
    description='Vehicle control algorithms',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pid_controller = vehicle_control.pid_controller:main',
            'teleop_keyboard = vehicle_control.teleop_keyboard:main',
            'teleop_ackermann = vehicle_control.teleop_keyboard_ackermann:main',
        ],
    },
)
