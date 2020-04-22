"""
TODO
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def combine_results(file_path1, file_path2, file_path3, file_name='/episode_performance.csv'):
    """
    Combine the results of 3 todo
    """
    data1 = pd.read_csv(file_path1+file_name, header=0, names=['episode', 'eval_steps', 'reward', 'done'])
    # data2 = pd.read_csv(file_path2+file_name, header=0, names=['episode', 'eval_steps', 'reward', 'done'])
    # data3 = pd.read_csv(file_path3+file_name, header=0, names=['episode', 'eval_steps', 'reward', 'done'])

    return data1


def plot_2_together(data1, data1_name, data2, data2_name, title):
    """

    """
    data1['algorithm'] = data1_name
    data2['algorithm'] = data2_name
    # print(data2)

    data_merge = pd.concat([data1, data2])
    # print(data_merge)

    fig = sns.relplot(x="episode", y="reward", kind="line", hue='algorithm', hue_order=[data1_name, data2_name],
                      data=data_merge)
    fig.fig.suptitle(title)

    plt.legend(labels=[data1_name, data2_name])
    fig.set(ylim=(-3000, 10))
    plt.show()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide data locations for plotting results')

    # Arguments for the 3 similar locations
    parser.add_argument('--folder1', help='first folder', default='rs_8')
    parser.add_argument('--folder2', help='second folder', default='rs_1964')
    parser.add_argument('--folder3', help='third folder', default='rs_1754')

    args = vars(parser.parse_args())

    ddpg_data = combine_results('./DDPG/'+args['folder1'], './DDPG/'+args['folder2'], './DDPG/'+args['folder3'])
    cbf_data = combine_results('./DDPG-CBF/'+args['folder1'], './DDPG-CBF/'+args['folder2'],
                               './DDPG-CBF/'+args['folder3'])

    plot_2_together(ddpg_data, 'DDPG', cbf_data, 'DDPG-CBF', 'Pendulum-v0')
