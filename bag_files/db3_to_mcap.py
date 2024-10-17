#
#
# Python dependencies:
# pip3 install pandas mcap-ros2-support
#
#
import sys
from pathlib import Path

def main():

    import os
    current_dir = os.getcwd()

    mcap_in = sys.argv[1]
    # mcap_in = Path('/home/la019/ros2_ws/src/racing-project-ros2/collision_avoidance_ros2/bag_files/subset/subset_0.db3')

    if mcap_in.endswith(".db3"):
        # Got a single file
        if os.path.isfile(mcap_in):
            # Absolute filepath given
            print('db3 file given: ' + mcap_in)
            mcaplist = [mcap_in]
        elif os.path.isfile(current_dir + "/" + mcap_in):
            # Relative filepath given
            mcap_in = current_dir + "/" + mcap_in
            mcaplist = [mcap_in]
            print('db3 file given: ' + mcap_in)
    else:
        # Got a directory
        if os.path.isdir(mcap_in):  
            # Absolute directory path given, keep mcap_in
            mcap_dir = mcap_in
            print('db3 directory given: ' + mcap_dir)
            mcaplist = os.listdir(mcap_dir)

        elif os.path.isdir(current_dir + "/" + mcap_in):
            # Relative directory path given
            mcap_dir = current_dir + "/" + mcap_in
            print('db3 directory given: ' + mcap_dir)
            mcaplist = os.listdir(mcap_dir)

    for mcap_file in mcaplist:
        if mcap_file.endswith(".db3"):

            if 'mcap_dir' in locals():
                mcap_file = mcap_dir + "/" + mcap_file

            print('Parsing file: ' + mcap_file)
            mcap_filename = os.path.splitext(mcap_file)[0]

            cmd_str = 'mcap convert ' + mcap_file + ' ' + mcap_filename + '.mcap'
            # print('Command string: ' + cmd_str)
            os.system(cmd_str)


if __name__ == '__main__':
    main()
