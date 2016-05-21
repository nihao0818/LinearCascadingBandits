# generate single file

import numpy as np
import matplotlib.pyplot as plt


def single_plot(Rank_10, Rank_20, Rank_40, Step):

    plt.figure(figsize=(8,6))

    avg_Rank_10 = np.mean(Rank_10, axis=0)
    avg_Rank_20 = np.mean(Rank_20, axis=0)
    avg_Rank_40 = np.mean(Rank_40, axis=0)

    # plt.plot(np.array(range(Step)), avg_Rank_10, label='Rank 10', color='r')
    # plt.plot(np.array(range(Step)), avg_Rank_20, label='Rank 20', color='b')
    # plt.plot(np.array(range(Step)), avg_Rank_40, label='Rank 40', color='g')

    plt.plot(np.array(range(Step)), avg_Rank_10, label='K = 4', color='r')
    plt.plot(np.array(range(Step)), avg_Rank_20, label='K = 8', color='b')
    plt.plot(np.array(range(Step)), avg_Rank_40, label='K = 12', color='g')

    plot_Error_Bar(Step, avg_Rank_10, Rank_10, 'Rank_10')
    plot_Error_Bar(Step, avg_Rank_20, Rank_20, 'Rank_20')
    plot_Error_Bar(Step, avg_Rank_40, Rank_40, 'Rank_40')

    add_Annotation()
    rename_XAxis(Step)


def plot_Error_Bar(Step, avg_Regret, all_Regret, flag):

    err = err_Calculate(all_Regret)

    if Step == 100000:
        x = np.arange(10000, Step, 9999)
    else:
        x = np.arange(100000, Step, 99999)

    y = avg_Regret[x]
    y_Err = err[x]

    if flag == 'Rank_10':
        plt.errorbar(x, y, yerr=y_Err, fmt=None, ecolor='r', lw=2, mew=2)
    elif flag == 'Rank_20':
        plt.errorbar(x, y, yerr=y_Err, fmt=None, ecolor='b', lw=2, mew=2)
    elif flag == 'Rank_40':
        plt.errorbar(x, y, yerr=y_Err, fmt=None, ecolor='g', lw=2, mew=2)

def err_Calculate(all_Regret):

    [a,b] = all_Regret.shape
    err = np.std(all_Regret, axis=0) / np.sqrt(a)

    return err

def add_Annotation():

    plt.xlabel('Step n')
    plt.ylabel('Regret')
    plt.legend(loc=2, frameon=False, fontsize=12)

def rename_XAxis(Step):

    ax = plt.gca()
    if Step == 100000:
        ax.set_xticklabels(('0', '20k', '40k', '60k', '80k', '100k'))
    else:
        ax.set_xticklabels(('0', '200k', '400k', '600k', '800k', '1M'))


if __name__ == '__main__':

    ##############################
    # put all your csv files here
    # assume 10 iterations
    # assume run 100k
    # need to revise if run 1M
    ##############################

    # only accept Step = 100k or Step = 1M
    Step = 100000

    # load UCB related files
    # Rank_10 = np.loadtxt('specific_Rank_10.csv', dtype=np.float64, delimiter=',')
    # Rank_20 = np.loadtxt('specific_Rank_20.csv', dtype=np.float64, delimiter=',')
    # Rank_40 = np.loadtxt('specific_Rank_40.csv', dtype=np.float64, delimiter=',')   

    # Rank_10 = np.loadtxt('Rank_10.csv', dtype=np.float64, delimiter=',')
    # Rank_20 = np.loadtxt('Rank_20.csv', dtype=np.float64, delimiter=',')
    # Rank_40 = np.loadtxt('Rank_40.csv', dtype=np.float64, delimiter=',')   

    # Rank_10 = np.loadtxt('Rank_4.csv', dtype=np.float64, delimiter=',')
    # Rank_20 = np.loadtxt('Rank_8.csv', dtype=np.float64, delimiter=',')
    # Rank_40 = np.loadtxt('Rank_12.csv', dtype=np.float64, delimiter=',')

    Rank_10 = np.loadtxt('Rank_reward_4.csv', dtype=np.float64, delimiter=',')
    Rank_20 = np.loadtxt('Rank_reward_8.csv', dtype=np.float64, delimiter=',')
    Rank_40 = np.loadtxt('Rank_reward_12.csv', dtype=np.float64, delimiter=',')


    single_plot(Rank_10, Rank_20, Rank_40, Step)

    # plt.savefig('reward_single_16_8.eps', dpi=800, bbox_inches='tight')
    plt.savefig('single_16_k_reward.eps', dpi=800, bbox_inches='tight')

    plt.show()
