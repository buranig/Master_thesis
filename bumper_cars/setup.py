from setuptools import find_packages, setup

package_name = 'bumper_cars'

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
    maintainer='giacomo',
    maintainer_email='buranig@student.ethz.ch',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "controller = bumper_cars.controller:main",
            "sensor = bumper_cars.sensor:main",
            "model = bumper_cars.car_model:main",
            "plotter = bumper_cars.plotter:main",
            "converter = bumper_cars.converter:main",
            "broadcaster = bumper_cars.broadcaster:main",
        ],
    },
)
