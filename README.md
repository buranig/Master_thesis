
# ROS2 Collision Avoidance for Bumper Cars

In this branch, the code from the ```master``` branch has been adapted as a ROS2 pacakge to perform collision-less multi-car control.

# Dependencies

```
pip3 install shapely cvxopt
```

## Installation instructions

In order to use this package on one of the lab's computers, the contents from ```la_racing``` and ```racing-project-ros2``` are necessary. Current maintainters of those repositories are Roland Schwan and Johannes Waibel, respectively. Follow the instructions they provide to properly install their repositories.

To install this repository along the other ROS2 packages, one must first checkout branch ```24spring-bumperless``` from both projects:

```
cd ~/ros2_ws/src/la_racing && \
git checkout 24spring-bumperless &&\
cd ../racing-project-ros2 &&\
git checkout 24spring-bumperless &&\
git clone -b ros2-humble git@github.com:buranig/Master_thesis.git collision-avoidance-ros2
```

This should not only copy the contents of this repository inside ```racing-project-ros2``` but also chechout the correct branch in each of the repositories.

Lastly, build all packages in the workspace:

```
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### Test the installation

Terminal 1:
```
ros2 launch lar_utils visualize.launch.py carNumber:=1
```

Terminal 2:
```
ros2 launch lar_utils simulate.launch.py carNumber:=1
```

Terminal 3:
```
ros2 launch bumper_cars collision-avoidance.launch.py carNumber:=1
```

Terminal 4 (_optional_):
```
ros2 run dwa_dev view_trajs
```

