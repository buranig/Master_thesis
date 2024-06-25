import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data plotting
import time
import os
file_path = os.path.dirname(os.path.realpath(__file__))
# file_path += "/../csv/data_0.csv" 
file_path += "/../all_data_collision.csv" 

import seaborn as sns


def main():
    # main_normal()
    # return
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    print(df.head())

    # change name from mpc_gpu to mppi
    df['alg'] = df['alg'].replace('mpc_gpu', 'mppi')

    # carNumber,predHorizon,desiredThrottle,alg,model,scenario,car_i,cum_time,lap_time,lap_number,it_time,isd
    algs = df['alg'].unique()
    pHs = df['predHorizon'].unique()
    desThrottles = df['desiredThrottle'].unique()
    models = df['model'].unique()
    scenarios = df['scenario'].unique()
    laps = df['lap_number'].unique()
    if 'iteration' in df.columns:
        iterations = df['iteration'].unique()
    else:
        iterations = [True]
    # Define colors for each predHorizon
    colors_pHs = sns.color_palette("husl", len(pHs))  # You can also use another palette like "viridis"
    
    #####################################################################################

    # # Computation time vs desired throttle for each prediction horizon 
    # for i in range(0, len(algs)):
    #     plt.figure(i)
    #     mask = (df['alg'] == algs[i])
    #     sns.violinplot(data=df[mask], 
    #                    x="desiredThrottle", y="it_time", 
    #                    hue="predHorizon", inner="quart", 
    #                    palette = 'Set2', legend='full',
    #                    fill=False)
    #     # sns.boxplot(x = df[df['alg']==algs[i]]['desiredThrottle'], 
    #     #             y = df[df['alg']==algs[i]]['it_time'], 
    #     #             hue = df[df['alg']==algs[i]]['predHorizon'],
    #     #             palette = 'Set2',
    #     #             legend='full') 
        
    #     plt.grid(True)
    #     plt.legend(title='Prediction Horizon (s)', loc='upper right')
    #     plt.xlabel("Desired throttle "+ r'$\left[\frac{m}{s^2}\right]$')
    #     plt.ylabel("Control computation time "+ r'$\left[s\right]$ ')
    #     plt.title("Control computation time vs desired throttle for " + algs[i].upper() + " algorithm")
    #     plt.show()
        

    #####################################################################################


    # # Plot the control differece vs lap time
    # for i in range(0, len(algs)):
    #     # plt.figure(i)
    #     # Create subplots: one subplot per pH
    #     fig, axs = plt.subplots(len(pHs), 1, figsize=(8, 2 * len(pHs)), sharex=True)
        
    #     # If there's only one subplot, axs will not be a list, so we make it a list
    #     if len(pHs) == 1:
    #         axs = [axs]

    #     for j, pH in enumerate(pHs):
    #         ax = axs[j]
    #         color = colors_pHs[j]
    #         num_trues = 0
    #         num_possib = 0
    #         lap_times = []
    #         for desThrottle in desThrottles:
    #             for model in models:
    #                 for scenario in scenarios:
    #                     # for iteration in iterations:
    #                         success = 0
    #                         mask = (df['alg'] == algs[i]) & (df['predHorizon'] == pH) & (df['desiredThrottle'] == desThrottle) & (df['model'] == model) & (df['scenario'] == scenario)
    #                         prev_time = 0.0
    #                         for lap_time in df[mask]['lap_time'].values:
    #                             if lap_time < prev_time:
    #                                 lap_times.append(prev_time)
    #                                 success += 1
    #                             prev_time = lap_time

    #                         if success:
    #                             num_trues += 1
    #                             for lap in range(1, success+1):
    #                                 mask = (df['alg'] == algs[i]) & (df['predHorizon'] == pH) & (df['desiredThrottle'] == desThrottle) & (df['model'] == model) & (df['scenario'] == scenario) & (df['lap_number'] == lap)
    #                                 ax.plot(df[mask]['lap_time'].values, df[mask]['isd'].values, color=color, alpha=0.2)
    #                         num_possib +=1
    #         if len(lap_times)==0:
    #             lap_times = [0]
            
    #         avg = sum(lap_times)/len(lap_times)
    #         ax.axvline(x=avg)
    #         ax.text(0.9, 0.7, 'Avg='+str(round(avg,2)), ha='center', va='center', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=1.0))
    #         ax.text(0.9, 0.9, 'N='+str(num_trues)+'/'+str(num_possib), ha='center', va='center', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=1.0))
    #         ax.grid(True)
    #         ax.set_title(f'Prediction Horizon: {pH}s')

    #     fig.text(0.5, 0.04, 'Lap Time [s]', ha='center', va='center', fontsize=12)
    #     fig.text(0.04, 0.5, 'Input Squared Difference [-]', ha='center', va='center', rotation='vertical', fontsize=12)
    #     fig.suptitle(f'Input Squared Difference for {algs[i].upper()} Algorithm')
    #     plt.show()

    #####################################################################################

    # Plot the time for each lap vs prediction horizon
    for i in range(0, len(algs)):
        data_times = {'pH':[], 'time':[], 'desThrottle':[]}
        for j, pH in enumerate(pHs):
            for desThrottle in desThrottles:
                for model in models:
                    for scenario in scenarios:
                        # for iteration in iterations:
                            mask = (df['alg'] == algs[i]) & (df['predHorizon'] == pH) & (df['desiredThrottle'] == desThrottle) & (df['model'] == model) & (df['scenario'] == scenario) 
                            if any(mask):
                                prev_time = 0.0
                                for lap_time in df[mask]['lap_time'].values:
                                    if lap_time < prev_time:
                                        data_times['desThrottle'].append(desThrottle)
                                        data_times['pH'].append(pH)
                                        data_times['time'].append(prev_time)
                                    prev_time = lap_time
               
        df_aux = pd.DataFrame(data_times)
        plt.figure(i)
        sns.boxplot(x = df_aux['desThrottle'], 
                    y = df_aux['time'], 
                    color='black',
                    # palette = 'Set2',
                    # inner="quart", 
                    # hue=df_aux['pH'],
                    legend='full',
                    fill=False
                    ) 

        plt.grid(True)
        plt.xlabel("Desired throttle "+ r'$\left[\frac{m}{s^2}\right]$')
        plt.ylabel("Lap time "+ r'$\left[s\right]$ ')
        plt.title("Lap time vs desired throttle for " + algs[i].upper() + " algorithm")
        plt.show()


def main_normal():
    # Run this function if not ran from the bash script
    header = "carNumber,cum_time,lap_time,lap_number,it_time,isd"
    df = pd.read_csv(file_path,names=header.split(","))

    # Create subplots: one subplot per scene control/compute time
    fig, axs = plt.subplots(2, 1, sharex=True)
    max_lap = max(df['lap_number'])-1 # Assume last lap not complete
    
    colors = sns.color_palette("husl", 2)  # You can also use another palette like "viridis"
    for j in range(len(axs)):
        ax = axs[j]
        color = colors[j]
        for i in range(1,max_lap):
            mask = (df['lap_number'] == i)
            
            if j == 0:
                ax.plot(df[mask]['lap_time'].values, df[mask]['it_time'].values, color=color, alpha=0.1)
                # ax.set_title('Computing time'+r'[ms]')
                ax.set_ylabel('Computing time'+r'[ms]')
                ax.set_ylim([0.011, 0.018])
            else:
                ax.plot(df[mask]['lap_time'].values, df[mask]['isd'].values, color=color, alpha=0.1)
                ax.set_ylabel('Input squared difference'+r'[-]')
                # ax.set_yaxis('log')
        ax.grid(True)

    fig.text(0.5, 0.04, 'Lap Time [s]', ha='center', va='center', fontsize=12)
    # fig.text(0.04, 0.5, 'Input Squared Difference [-]', ha='center', va='center', rotation='vertical', fontsize=12)
    fig.suptitle('Value for computing time and input squared difference through the lap')
    plt.show()


if __name__ == '__main__':
    main()
    pass