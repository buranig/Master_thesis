
# ROS2 Collision Avoidance for Bumper Cars

In this branch, the code from the ```master``` branch has been adapted as a ROS2 pacakge to perform collision-less multi-car control.

# Dependencies

```
pip3 install shapely piqp
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
ros2 launch lar_utils visualize.launch.py carNumber:=2
```

Terminal 2:
```
ros2 launch lar_utils simulate.launch.py carNumber:=2
```

Terminal 3:
```
ros2 launch bumper_cars collision-avoidance.launch.py alg:=c3bf carNumber:=2
```

Terminal 4 (_optional_):
```
ros2 run dwa_dev view_trajs
```


### Run the installation on upstairs lab

Modify ```mocap.yaml``` located in:
```
~/ros2_ws/src/la_racing/src/ros2/lar_utils/config/mocap/
```
To ensure that all the required cars are being published by the Motive software.

Additionally, check for each car (IPs are like http://192.168.0.157, just changing the last number) that they are listening
to the right IP address, and that ROS_DOMAIN_ID is set to the right value.

Terminal 1:
```
ros2 launch lar_utils mocap.launch.py
```
Once it stabilizes, run the second terminal's command

Terminal 2:
```
ros2 launch lar_utils ekf.launch.py carNumber:=2
```
where you can change the number of cars to listen to.

Terminal 3:
```
ros2 launch lar_utils visualize.launch.py source:=real carNumber:=2
```
where you can verify that the car(s) and track are found correctly by the mocap setup.

Terminal 4:
```
ros2 launch lar_utils micro_ros.launch.py
```

Terminal 5:
```
ros2 launch bumper_cars collision-avoidance.launch.py source:=real gen_traj:=False alg:=c3bf carNumber:=2 static_throttle:=0.2
```

#### Description of the parameters

* **source**: [```sim```/```real```] Describes where the information comes from
* **gen_traj**: [```True```/```False```] For DWA, describes whether trajectories must be recomputed before running the algorithm.
* **carNumber**: [```(0-9)+```] Describes the amount of cars to be accounted for. Both for CA and to move them.
* **alg**: [```dwa```, ```cbf```, ```c3bf```] What algorithm to use for collision avoidance.
* **static_throttle**: [```0.0``` - ```1.0```] How fast we want cars that are not controlled via joystick to move.