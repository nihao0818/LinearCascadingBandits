# generate combined 3 plots
# codes are not optimized, you may use this file
# or just copy the related lines to your ownfile
# input: 3*3 .csv file
# output: combine.eps at the same folder (change to your own file name)

import numpy as np
import matplotlib.pyplot as plt

def combine_Three_Plots(L16_UCB, L256_UCB, Lmax_UCB,
                     L16_LinTS, L256_LinTS, Lmax_LinTS,
                     L16_RankTS, L256_RankTS, Lmax_RankTS,
                     L16_Step, L256_Step, Lmax_Step):

    plt.figure(figsize=(20,3))

    ##############################
    # generate the first plot
    ##############################

    plt.subplot(131)

    avg_L16_UCB = np.mean(L16_UCB, axis=0)
    avg_L16_LinTS = np.mean(L16_LinTS, axis=0)
    avg_L16_RankTS = np.mean(L16_RankTS, axis=0)

    plt.plot(np.array(range(L16_Step)), avg_L16_UCB, label='CascadeUCB1', color='r')
    plt.plot(np.array(range(L16_Step)), avg_L16_LinTS, label='CascadeLinTS', color='b')
    plt.plot(np.array(range(L16_Step)), avg_L16_RankTS, label='RankedLinTS', color='g')

    plot_Error_Bar(L16_Step, avg_L16_UCB, L16_UCB, 'UCB1')
    plot_Error_Bar(L16_Step, avg_L16_LinTS, L16_LinTS, 'LinTS')
    plot_Error_Bar(L16_Step, avg_L16_RankTS, L16_RankTS, 'RankTS')

    add_Annotation()
    rename_XAxis(L16_Step)

    ##########################################
    # change max value of y here
    ##########################################
    plt.ylim(0,4000)

    plt.title('$L = 16,\,K = 4$', fontsize=18)

    ##############################
    # generate the second plot
    ##############################

    p2 = plt.subplot(132)

    avg_L256_UCB = np.mean(L256_UCB, axis=0)
    avg_L256_LinTS = np.mean(L256_LinTS, axis=0)
    avg_L256_RankTS = np.mean(L256_RankTS, axis=0)

    plt.plot(np.array(range(L256_Step)), avg_L256_UCB, label='CascadeUCB1', color='r')
    plt.plot(np.array(range(L256_Step)), avg_L256_LinTS, label='CascadeLinTS', color='b')
    plt.plot(np.array(range(L256_Step)), avg_L256_RankTS, label='RankedLinTS', color='g')

    plot_Error_Bar(L256_Step, avg_L256_UCB, L256_UCB, 'UCB1')
    plot_Error_Bar(L256_Step, avg_L256_LinTS, L256_LinTS, 'LinTS')
    plot_Error_Bar(L256_Step, avg_L256_RankTS, L256_RankTS, 'RankTS')

    add_Annotation()
    rename_XAxis(L256_Step)

    ##########################################
    # change max value of y here
    ##########################################
    ymax = 4000
    plt.ylim(0,ymax)

    # p2.text(100000/2,-ymax/2.5,'(a)',fontsize=20,verticalalignment="center",horizontalalignment="center")
    p2.text(100000/2,ymax * 1.25,'Yelp Restaurant Dataset',fontsize=20,verticalalignment="center",horizontalalignment="center")

    plt.title('$L = 256,\, K = 4$', fontsize=18)

    ##############################
    # generate the third plot
    ##############################

    plt.subplot(133)

    avg_Lmax_UCB = np.mean(Lmax_UCB, axis=0)
    avg_Lmax_LinTS = np.mean(Lmax_LinTS, axis=0)
    avg_Lmax_RankTS = np.mean(Lmax_RankTS, axis=0)

    plt.plot(np.array(range(Lmax_Step)), avg_Lmax_UCB, label='CascadeUCB1', color='r')
    plt.plot(np.array(range(Lmax_Step)), avg_Lmax_LinTS, label='CascadeLinTS', color='b')
    plt.plot(np.array(range(Lmax_Step)), avg_Lmax_RankTS, label='RankedLinTS', color='g')

    plot_Error_Bar(Lmax_Step, avg_Lmax_UCB, Lmax_UCB, 'UCB1')
    plot_Error_Bar(Lmax_Step, avg_Lmax_LinTS, Lmax_LinTS, 'LinTS')
    plot_Error_Bar(Lmax_Step, avg_Lmax_RankTS, Lmax_RankTS, 'RankTS')

    add_Annotation()
    rename_XAxis(Lmax_Step)

    ##########################################
    # change max value of y here
    ##########################################
    plt.ylim(0,4000)

    plt.title('$L = 3000,\, K = 4$', fontsize=18)

def plot_Error_Bar(Step, avg_Regret, all_Regret, flag):

    err = err_Calculate(all_Regret)

    if Step == 100000:
        x = np.arange(10000, Step, 9999)
    else:
        x = np.arange(100000, Step, 99999)

    y = avg_Regret[x]
    y_Err = err[x]

    if flag == 'UCB1':
        plt.errorbar(x, y, yerr=y_Err, fmt=None, ecolor='r', lw=2, mew=2)
    elif flag == 'LinTS':
        plt.errorbar(x, y, yerr=y_Err, fmt=None, ecolor='b', lw=2, mew=2)
    elif flag == 'RankTS':
        plt.errorbar(x, y, yerr=y_Err, fmt=None, ecolor='g', lw=2, mew=2)

def err_Calculate(all_Regret):

    [a,b] = all_Regret.shape
    err = np.std(all_Regret, axis=0) / np.sqrt(a)

    return err

def add_Annotation():

    plt.xlabel('Step n', fontsize=16)
    plt.ylabel('Regret', fontsize=16)
    plt.legend(loc=2, frameon=False, fontsize=12)

def rename_XAxis(Step):

    ax = plt.gca()
    if Step == 100000:
        ax.set_xticklabels(('0', '20k', '40k', '60k', '80k', '100k'))
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(16)
    else:
        ax.set_xticklabels(('0', '200k', '400k', '600k', '800k', '1M'))
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(16)

    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(16)


if __name__ == '__main__':

    ##############################
    # put all your csv files here
    # assume 10 iterations
    # assume run 100k
    # need to revise if run 1M
    ##############################

    # only accept Step = 100k or Step = 1M
    L16_Step = 100000
    L256_Step = 100000
    Lmax_Step = 100000

    # load UCB related files
    L16_UCB = np.loadtxt('a_L16_UCB.csv', dtype=np.float64, delimiter=',')
    L256_UCB = np.loadtxt('a_L256_UCB.csv', dtype=np.float64, delimiter=',')
    Lmax_UCB = np.loadtxt('a_Lmax_UCB.csv', dtype=np.float64, delimiter=',')

    # load LinTS related files
    L16_LinTS = np.loadtxt('L16_LinTS_4_20.csv', dtype=np.float64, delimiter=',')
    L256_LinTS = np.loadtxt('L256_LinTS_4_20.csv', dtype=np.float64, delimiter=',')
    Lmax_LinTS = np.loadtxt('L3000_LinTS_4_20.csv', dtype=np.float64, delimiter=',')

    # load Rank related files
    L16_RankTS = np.loadtxt('L16_RankTS_4_20.csv', dtype=np.float64, delimiter=',')
    L256_RankTS = np.loadtxt('L256_RankTS_4_20.csv', dtype=np.float64, delimiter=',')
    Lmax_RankTS = np.loadtxt('L3000_RankTS_4_20.csv', dtype=np.float64, delimiter=',')

    ##############################
    # combine 3 plots
    ##############################

    combine_Three_Plots(L16_UCB, L256_UCB, Lmax_UCB,
                        L16_LinTS, L256_LinTS, Lmax_LinTS,
                        L16_RankTS, L256_RankTS, Lmax_RankTS,
                        L16_Step, L256_Step, Lmax_Step)

    ##############################
    # plot is in the same folder
    ##############################

    plt.savefig('Exp1.eps', dpi=800, bbox_inches='tight')

    # plt.show()
