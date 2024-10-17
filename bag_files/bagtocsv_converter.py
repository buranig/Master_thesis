from pathlib import Path
import rosbag

bag_path = Path('/home/la019/ros2_ws/src/racing-project-ros2/collision_avoidance_ros2/bag_files/subset/subset_0.db3')
bag = rosbag.Bag(bag_path)
topics = bag.get_type_and_topic_info()[1].keys()
types = []
for val in bag.get_type_and_topic_info()[1].values():
    types.append(val[0])