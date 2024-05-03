
# ROS2 Collision Avoidance for Bumper Cars

In this branch, the code from the ```master``` branch has been adapted as a ROS2 pacakge to perform collision-less multi-car control.

To add it as a ROS2 package, simpy head to the ```your-workspace/src``` folder and git clone it there:

```
git clone -b ros2-humble git@github.com:buranig/Master_thesis.git collision-avoidance-ros2
```

Then, compile your current workspace with the newly added packages:
```
colcon build --symlink-install
```

## Command reference

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

