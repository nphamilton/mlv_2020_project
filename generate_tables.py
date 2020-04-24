"""
TODO
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def generate_latex_table(rs1, rs2, rs3, target='./table.txt'):
    """

    """
    paths = ['./DDPG/rs_{}/final_eval.csv'.format(rs1), './DDPG/rs_{}/final_eval.csv'.format(rs2),
             './DDPG/rs_{}/final_eval.csv'.format(rs3),
             './DDPG/constrained_rs_{}/final_eval.csv'.format(rs1),
             './DDPG/constrained_rs_{}/final_eval.csv'.format(rs2),
             './DDPG/constrained_rs_{}/final_eval.csv'.format(rs3),
             './DDPG-CBF/rs_{}/final_eval.csv'.format(rs1), './DDPG-CBF/rs_{}/final_eval.csv'.format(rs2),
             './DDPG-CBF/rs_{}/final_eval.csv'.format(rs3),
             './DDPG-CBF/rs_{}/final_cbf_eval.csv'.format(rs1), './DDPG-CBF/rs_{}/final_cbf_eval.csv'.format(rs2),
             './DDPG-CBF/rs_{}/final_cbf_eval.csv'.format(rs3)]
    names = ['DDPG_{}'.format(rs1), 'DDPG_{}'.format(rs2), 'DDPG_{}'.format(rs3),
             'DDPG-C_{}'.format(rs1), 'DDPG-C_{}'.format(rs2), 'DDPG-C_{}'.format(rs3),
             'CBF-N_{}'.format(rs1), 'CBF-N_{}'.format(rs2), 'CBF-N_{}'.format(rs3),
             'CBF_{}'.format(rs1), 'CBF{}'.format(rs2), 'CBF_{}'.format(rs3)]

    # Define the important columns to the table
    cols = ['Method', 'Expected Reward', 'Percent Safe Runs']

    # Fill the data frame
    data = pd.concat([pd.DataFrame([extract_data(file_path=paths[i], name=names[i])], columns=cols)
                      for i in range(len(paths))], ignore_index=True)
    # print(data)
    print(data.to_latex(index=False))
    return


def extract_data(file_path, name):
    """

    """
    #
    data = pd.read_csv(file_path, header=0, names=['reward', 'steps', 'done', 'safe'])

    # Compute the expected reward using the mean and standard deviation
    rewards = data['reward'].to_numpy()
    exp_reward = '{}+-{}'.format(round(np.mean(rewards), 4), round(np.std(rewards), 4))

    # Compute the percent of safe trajectories
    safes = data['safe'].to_numpy()
    # print(safes)
    safes = np.where(safes == ' True', 1, 0)
    safe_percent = str(100 * np.sum(safes) / len(safes)) + '%'

    return [name, exp_reward, safe_percent]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide data locations for plotting results')

    # Arguments for the 3 similar locations
    parser.add_argument('--folder1', help='first folder', default='rs_8')
    parser.add_argument('--folder2', help='second folder', default='rs_1964')
    parser.add_argument('--folder3', help='third folder', default='rs_1754')

    args = vars(parser.parse_args())

    generate_latex_table(8, 1964, 1754, target='./table.txt')

    print('Hello World!')
