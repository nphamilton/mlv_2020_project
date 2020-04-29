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
    data2 = pd.read_csv(file_path2+file_name, header=0, names=['episode', 'eval_steps', 'reward', 'done'])
    data3 = pd.read_csv(file_path3+file_name, header=0, names=['episode', 'eval_steps', 'reward', 'done'])

    data_merge = pd.concat([data1, data2, data3])

    return data_merge


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

    # plt.legend(labels=[data1_name, data2_name])
    fig.set(xlim=(0, 1000), ylim=(-1250, 100))
    # plt.show()
    return


def plot_all(ddpg, ddpg_c, cbf, cbf_w):
    """

    """
    ddpg['algorithm'] = 'DDPG'
    ddpg_c['algorithm'] = 'DDPG-C'
    cbf['algorithm'] = 'CBF-N'
    cbf_w['algorithm'] = 'CBF'

    data_merge = pd.concat([ddpg, ddpg_c, cbf, cbf_w])
    # print(data_merge)

    order = ['CBF', 'CBF-N', 'DDPG-C', 'DDPG']  # ['DDPG', 'DDPG-C', 'CBF-N', 'CBF']
    fig = sns.relplot(x="episode", y="reward", kind="line", hue='algorithm',
                      hue_order=order, data=data_merge)
    fig.fig.suptitle('Pendulum-v0')

    # plt.legend(labels=order)
    fig.set(xlim=(0, 1000), ylim=(-1250, 100))
    # plt.show()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide data locations for plotting results')

    # Arguments for the 3 similar locations
    parser.add_argument('--folder1', help='first folder', default='rs_8')
    parser.add_argument('--folder2', help='second folder', default='rs_1964')
    parser.add_argument('--folder3', help='third folder', default='rs_1754')

    args = vars(parser.parse_args())

    ddpg_data = combine_results('./DDPG/'+args['folder1'], './DDPG/'+args['folder2'], './DDPG/'+args['folder3'])
    ddpg_c_data = combine_results('./DDPG/constrained_'+args['folder1'], './DDPG/constrained_'+args['folder2'],
                                  './DDPG/constrained_'+args['folder3'])
    cbf_data = combine_results('./DDPG-CBF/'+args['folder1'], './DDPG-CBF/'+args['folder2'],
                               './DDPG-CBF/'+args['folder3'])
    cbf_w_data = combine_results('./DDPG-CBF/' + args['folder1'], './DDPG-CBF/' + args['folder2'],
                                 './DDPG-CBF/' + args['folder3'], file_name='/episode_cbf_performance.csv')

    plot_2_together(ddpg_data, 'DDPG', ddpg_c_data, 'DDPG-C', 'DDPG vs DDPG-C')
    plot_2_together(ddpg_data, 'DDPG', cbf_data, 'CBF-N', 'DDPG vs CBF-N')
    plot_2_together(ddpg_data, 'DDPG', cbf_w_data, 'CBF', 'DDPG vs CBF')
    plot_2_together(ddpg_c_data, 'DDPG-C', cbf_data, 'CBF-N', 'DDPG-C vs CBF-N')
    plot_2_together(ddpg_c_data, 'DDPG-C', cbf_w_data, 'CBF', 'DDPG-C vs CBF')
    plot_2_together(cbf_data, 'CBF-N', cbf_w_data, 'CBF', 'CBF-N vs CBF')

    plot_all(ddpg_data, ddpg_c_data, cbf_data, cbf_w_data)

    plt.show()
