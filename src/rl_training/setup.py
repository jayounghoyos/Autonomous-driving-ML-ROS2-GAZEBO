from setuptools import find_packages, setup

package_name = 'rl_training'

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
    maintainer='jayounghoyos',
    maintainer_email='jayoungh@eafit.edu.co',
    description='Deep Reinforcement Learning for Autonomous Navigation with Obstacle Avoidance',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'train_ppo = rl_training.train_ppo:main',
            'rl_agent = rl_training.rl_agent:main'
        ],
    },
)
