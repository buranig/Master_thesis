import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the CSV file
data = pd.read_csv('src/seed_simulation/seed_simulation/seed_sim.csv')
filename = 'src/seed_simulation/seed_simulation/seed_sim.csv'
# data = data.to_numpy()
fontsize = 10
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = fontsize

#colors = ['#3D6BB6', '#94A5C4', '#CCD1DD', '#CFB9B7', '#614947', '#4F2020']
colors = np.array(sns.color_palette(None, n_colors=8))[[0, 1, 2, 3, 4, 7]]

class Plotter:
    def __init__(self, data):
        self.data = data
    
    def plot_bars1(self, filename, ylabels):
        # Create scatter plot
        sns.set_style("whitegrid")
        fig, axs = plt.subplots(2, figsize=(10, 6))

        # Plot both cost and violations
        for lab, (ax, affix) in zip(ylabels, zip(axs, ['', '_viol'])):

            # Read the data from the CSV file into a pandas DataFrame
            df = pd.read_csv(filename.split(".")[0] + affix + ".csv", header=None)
            # df = df.drop(columns=1, axis=1)
            # Rename the columns
            # df = df.rename(columns={0: "DRInC", 1: "Empirical", 2: "Robust",
            #                         3: "LQG", 4: "DRLQG",
            #                         5: "Wasserstein distance"})

            # Round the Wasserstein distance to the nearest 0.02
            # The data should be much closer than that
            for r in df.index:
                df.iloc[r, -1] = 0.02 * round(df.iloc[r, -1] / 0.02)

            # Melt the DataFrame to reshape it
            melted_df = melt_and_add_category(df)

            # Create the box plot using Seaborn
            sns.boxplot(x='Method', y='Value', hue='Category',
                        data=melted_df, ax=ax, palette=colors)

            # Set plot labels and legend
            ax.set_ylabel(lab)
            ax.set_ylim(8 if affix == '' else None, None, auto=True)
            ax.legend(title='Wasserstein distance (ordered)',
                    ncol=round(np.max(df.iloc[:, -1].to_numpy() / 0.02)),
                    columnspacing=1.2)

        # Only set xlabel at the bottom
        axs[0].set_xlabel('')

    def plot_bars(self, x, y, hue, title):
        # Create scatter plot
        # sns.set_style("whitegrid")
        fig, ax = plt.subplots(1, figsize=(10, 6))

        # Create the box plot using Seaborn
        sns.boxplot(x=x, y=y, hue=hue,
                    data=self.data, ax=ax, palette=colors)
        
        plt.title(title, fontdict={'size': fontsize, 'family': 'serif'})
        # plt.yscale('log')
        plt.xlabel("Number of Robots",fontdict={'size': fontsize, 'family': 'serif'})
        # plt.ylabel(fontdict={'size': fontsize, 'family': 'serif'})


    def plot(self, x, y, hue, title):
        
        sns.lineplot(data=self.data, x=x, y=y, hue=hue)
        # plt.xlabel("x [m]", fontdict={'size': fontsize, 'family': 'serif'})
        # plt.ylabel("y [m]", fontdict={'size': fontsize, 'family': 'serif'})
        plt.title(title, fontdict={'size': fontsize, 'family': 'serif'})
        plt.show()

quantities = ['Path Length', 'Acceleration Usage', 'Steering Usage', 'Average Speed', 'Avg Computational Time',	'Solver Failure', 'Collision Number']
methods = ['MPC', 'LBP', 'CBF', 'C3BF', 'DWA']
noises = [0.0, 0.1, 0.2, 0.4]

# data_plotter = Plotter(data)
# data_plotter.plot_bars("File Name", "Collision Number", None, "Collision number as a function of the seed file")
# for idx in range(0,58):
#     file_data = data.loc[data["File Name"] == "circular_seed_"+str(idx)+".json"]
#     # plt.scatter(file_data["Robot Number"].iloc[0], max(file_data["Collision Number"]))

# plt.show()


for method in methods:
    for quantity in quantities:
        plt.close()
        mpc_data = data.loc[data["Method"] == method]
        mpc_plotter = Plotter(mpc_data)
        mpc_plotter.plot_bars("Robot Number", quantity, 'Noise Scaling', quantity+" vs Number of robots " + '(Method: ' + method + ')')
        savepath = "/home/giacomo/Documenti/Thesis report/results/"
        if quantity == 'Collision Number':
            pre = "collision_"
        elif quantity == 'Solver Failure':
            pre = "failure_"
        elif quantity == 'Avg Computational Time':
            pre = "comp_"
        elif quantity == 'Average Speed':
            pre = "speed_"
        elif quantity == 'Steering Usage':
            pre = "steer_"
        elif quantity == 'Acceleration Usage':
            pre = "acc_"
        elif quantity == 'Path Length':
            pre = "path_"
        plt.savefig(savepath + pre + "vs_robot_" + method + "_hue_noise"+ ".pdf")
        plt.show(block=False)
        plt.pause(1)
        plt.close()
        # plt.savefig('Figure1.svg')

for noise in noises:
    for quantity in quantities:
        plt.close()
        mpc_data = data.loc[data["Noise Scaling"] == noise]
        mpc_plotter = Plotter(mpc_data)
        mpc_plotter.plot_bars("Robot Number", quantity, 'Method', quantity+" vs Number of robots " + '(Noise Scaling Parameter: ' + str(noise) + ')')

        if quantity == 'Collision Number':
            pre = "collision_"
        elif quantity == 'Solver Failure':
            pre = "failure_"
        elif quantity == 'Avg Computational Time':
            pre = "comp_"
        elif quantity == 'Average Speed':
            pre = "speed_"
        elif quantity == 'Steering Usage':
            pre = "steer_"
        elif quantity == 'Acceleration Usage':
            pre = "acc_"
        elif quantity == 'Path Length':
            pre = "path_"
        
        if noise == 0.0:
            noise_str = "Noise0"
        elif noise == 0.1:
            noise_str = "Noise1"
        elif noise == 0.2:
            noise_str = "Noise2"
        elif noise == 0.4:
            noise_str = "Noise3"
        plt.savefig(savepath + pre + "vs_robot_" + noise_str + "_hue_method"+ ".pdf")