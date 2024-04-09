import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'dwa_dev'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.json')),
    ],
    install_requires=['setuptools', 'lar_msgs', 'lar_utils'],
    zip_safe=True,
    maintainer='giacomo',
    maintainer_email='buranig@stuent.ethz.ch',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dwa = dwa_dev.DWA:main',
            'view_trajs = dwa_dev.view_trajs:main'
        ],
    },
)
