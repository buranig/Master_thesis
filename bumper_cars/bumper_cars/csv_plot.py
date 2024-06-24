import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Path to the CSV file
# Data storing
import time
import os
file_path = os.path.dirname(os.path.realpath(__file__))

file_path += "/csv/data.csv" 

def main():
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    print(df.head())

    # Convert the DataFrame to a NumPy array and add offset
    arr = df.to_numpy()
    arr[:,0] = np.where(arr[:,1] != 0, arr[:,0], arr[:,0] - arr[0,0])
    # arrAux = arr[:,0] - arr[0,0]
    # arr[:,0] = arrAux
    mask = arr[:,1].astype(int) != 0
    arr = arr[mask][:]
    print(df.columns)
    col = df.columns[1:]
    for column in range(arr.shape[1]-2):
        for lap in range(1, max(arr[:,1].astype(int))+1):
            mask = arr[:,1].astype(int) == lap
            plt.plot(arr[mask,0], arr[mask,column+2],color='blue', linewidth=1.5, alpha=0.1)
        # plt.plot(arr[:,0], arr[:,column+2], label=col[column], linewidth=1.5, alpha=0.7)
    # Adding title and labels
    # plt.title('All Columns vs First Column')
    plt.xlabel("Time (s)")
    plt.ylabel('Values')

    # # Adding a grid for better readability
    plt.grid(True)

    # # Adding a legend
    # plt.legend()

    # # Save the plot as an image file
    # plt.savefig('all_columns_vs_first_column.png')

    # # Show the plot
    plt.show()

if __name__ == '__main__':
    pass