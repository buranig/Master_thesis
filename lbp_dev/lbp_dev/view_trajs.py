import json
import matplotlib.pyplot as plt

from ament_index_python.packages import get_package_share_directory

def plot_all_trajectories(json_file_path):
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    plt.figure(figsize=(12, 8))
    
    # Iterate over each scenario in the data
    for scenario, trajectories in data.items():
        # Plot each trajectory
        for traj_key, traj_data in trajectories.items():
            x = traj_data['x']
            y = traj_data['y']
            plt.plot(x, y)

    # Add labels and legend
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Accumulated Trajectories')
    # plt.legend(loc='upper right')
    plt.grid(True)
    
    # Show the plot
    plt.show()




def main():
    # Read trajectory file
    package_path = get_package_share_directory('lbp_dev')

    # Plot the trajectories
    plot_all_trajectories(package_path + '/config/LBP.json')

if __name__ == '__main__':  
    main()