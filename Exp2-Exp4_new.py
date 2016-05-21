# generate combined 3 plots
# codes are not optimized, you may use this file
# or just copy the related lines to your ownfile
# input: 3*3 .csv file
# output: combine.eps at the same folder (change to your own file name)

import numpy as np
import matplotlib.pyplot as plt

def combine_Exp2Exp4_Plots(L256_LinTS_d10, L256_LinTS_d20, L256_LinTS_d40,
                        Sub_L256_LinTS_d10, Sub_L256_LinTS_d20, Sub_L256_LinTS_d40,
                        L256_LinTS_K4, L256_LinTS_K8, L256_LinTS_K12,
                        L256_LinTS_K4_Rwd, L256_LinTS_K8_Rwd, L256_LinTS_K12_Rwd,
                        Exp2_Step, Exp3_Step, Exp41_Step, Exp42_Step,
                        Exp2_ymax, Exp3_ymax, Exp41_ymax, Exp42_ymax):

    plt.figure(figsize=(28,3))

    ##############################
    # Exp2 plot
    ##############################

    p1 = plt.subplot(141)

    avg_L256_LinTS_d10 = np.mean(L256_LinTS_d10, axis=0)
    avg_L256_LinTS_d20 = np.mean(L256_LinTS_d20, axis=0)
    avg_L256_LinTS_d40 = np.mean(L256_LinTS_d40, axis=0)

    plt.plot(np.array(range(Exp2_Step)), avg_L256_LinTS_d10, label='$d = 10$', color='r')
    plt.plot(np.array(range(Exp2_Step)), avg_L256_LinTS_d20, label='$d = 20$', color='b')
    plt.plot(np.array(range(Exp2_Step)), avg_L256_LinTS_d40, label='$d = 40$', color='g')

    plot_Error_Bar(Exp2_Step, avg_L256_LinTS_d10, L256_LinTS_d10, 'd=10')
    plot_Error_Bar(Exp2_Step, avg_L256_LinTS_d20, L256_LinTS_d20, 'd=20')
    plot_Error_Bar(Exp2_Step, avg_L256_LinTS_d40, L256_LinTS_d40, 'd=40')

    add_Annotation()
    rename_XAxis(Exp2_Step)
    plt.ylim(0,Exp2_ymax)
    # p1.text(Exp2_Step/2,-Exp2_ymax/2,'(a)',fontsize=20,verticalalignment="center",horizontalalignment="center")

    ax = plt.gca()

    ##########################################
    # change y tick lable
    ##########################################
    # ax.set_yticklabels(('0', '1k', '2k', '3k', '4k', '5k', '6k'))

    for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(16)

    ##############################
    # Exp3 plot
    ##############################

    p2 = plt.subplot(142)

    avg_Sub_L256_LinTS_d10 = np.mean(Sub_L256_LinTS_d10, axis=0)
    avg_Sub_L256_LinTS_d20 = np.mean(Sub_L256_LinTS_d20, axis=0)
    avg_Sub_L256_LinTS_d40 = np.mean(Sub_L256_LinTS_d40, axis=0)

    plt.plot(np.array(range(Exp3_Step)), avg_Sub_L256_LinTS_d10, label='$d = 10$', color='r')
    plt.plot(np.array(range(Exp3_Step)), avg_Sub_L256_LinTS_d20, label='$d = 20$', color='b')
    plt.plot(np.array(range(Exp3_Step)), avg_Sub_L256_LinTS_d40, label='$d = 40$', color='g')

    plot_Error_Bar(Exp3_Step, avg_Sub_L256_LinTS_d10, Sub_L256_LinTS_d10, 'd=10')
    plot_Error_Bar(Exp3_Step, avg_Sub_L256_LinTS_d20, Sub_L256_LinTS_d20, 'd=20')
    plot_Error_Bar(Exp3_Step, avg_Sub_L256_LinTS_d40, Sub_L256_LinTS_d40, 'd=40')

    add_Annotation()
    rename_XAxis(Exp3_Step)
    plt.ylim(0,Exp3_ymax)
    # p2.text(Exp3_Step/2,-Exp3_ymax/2,'(b)',fontsize=20,verticalalignment="center",horizontalalignment="center")

    ##########################################
    # Restaurant or what
    ##########################################
    p2.text(110000,1.2 * Exp3_ymax,'Yelp Restaurant Dataset',fontsize=20,verticalalignment="center",horizontalalignment="center")

    ax = plt.gca()

    ##########################################
    # change y tick lable
    ##########################################
    #ax.set_yticklabels(('0', '1k', '2k', '3k', '4k'))

    for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(16)

    ##############################
    # Exp4 regret plot
    ##############################

    p3 = plt.subplot(143)

    avg_L256_LinTS_K4 = np.mean(L256_LinTS_K4, axis=0)
    avg_L256_LinTS_K8 = np.mean(L256_LinTS_K8, axis=0)
    avg_L256_LinTS_K12 = np.mean(L256_LinTS_K12, axis=0)

    plt.plot(np.array(range(Exp41_Step)), avg_L256_LinTS_K4, label='$K = 4$', color='r')
    plt.plot(np.array(range(Exp41_Step)), avg_L256_LinTS_K8, label='$K = 8$', color='b')
    plt.plot(np.array(range(Exp41_Step)), avg_L256_LinTS_K12, label='$K = 12$', color='g')

    plot_Error_Bar(Exp3_Step, avg_L256_LinTS_K4, L256_LinTS_K4, 'd=10')
    plot_Error_Bar(Exp3_Step, avg_L256_LinTS_K8, L256_LinTS_K8, 'd=20')
    plot_Error_Bar(Exp3_Step, avg_L256_LinTS_K12, L256_LinTS_K12, 'd=40')

    add_Annotation()
    rename_XAxis(Exp41_Step)
    plt.ylim(0,Exp41_ymax)
    # p3.text(Exp41_Step/2,-Exp41_ymax/2,'(c)',fontsize=20,verticalalignment="center",horizontalalignment="center")

    ax = plt.gca()

    ##########################################
    # change y tick lable
    ##########################################
    # ax.set_yticklabels(('0', '1k', '2k', '3k', '4k', '5k', '6k'))

    for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(16)

    ##############################
    # Exp4 reward plot
    ##############################

    p4 = plt.subplot(144)

    avg_L256_LinTS_K4_Rwd = np.mean(L256_LinTS_K4_Rwd, axis=0)
    avg_L256_LinTS_K8_Rwd = np.mean(L256_LinTS_K8_Rwd, axis=0)
    avg_L256_LinTS_K12_Rwd = np.mean(L256_LinTS_K12_Rwd, axis=0)

    plt.plot(np.array(range(Exp42_Step)), avg_L256_LinTS_K4_Rwd, label='$K = 4$', color='r')
    plt.plot(np.array(range(Exp42_Step)), avg_L256_LinTS_K8_Rwd, label='$K = 8$', color='b')
    plt.plot(np.array(range(Exp42_Step)), avg_L256_LinTS_K12_Rwd, label='$K = 12$', color='g')

    plot_Error_Bar(Exp42_Step, avg_L256_LinTS_K4_Rwd, L256_LinTS_K4_Rwd, 'd=10')
    plot_Error_Bar(Exp42_Step, avg_L256_LinTS_K8_Rwd, L256_LinTS_K8_Rwd, 'd=20')
    plot_Error_Bar(Exp42_Step, avg_L256_LinTS_K12_Rwd, L256_LinTS_K12_Rwd, 'd=40')

    plt.xlabel('Step n', fontsize=16)
    plt.ylabel('Reward', fontsize=16)
    plt.legend(loc=2, frameon=False, fontsize=14)

    rename_XAxis(Exp42_Step)
    plt.ylim(0,Exp42_ymax)
    # p4.text(Exp42_Step/2,-Exp42_ymax/2,'(d)',fontsize=20,verticalalignment="center",horizontalalignment="center")

    ax = plt.gca()
    ax.set_yticklabels(('0', '20k', '40k', '60k', '80k', '100k'))
    for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(16)

def plot_Error_Bar(Step, avg_Regret, all_Regret, flag):

    err = err_Calculate(all_Regret)

    if Step == 100000:
        x = np.arange(10000, Step, 9999)
    else:
        x = np.arange(100000, Step, 99999)

    y = avg_Regret[x]
    y_Err = err[x]

    if flag == 'd=10':
        plt.errorbar(x, y, yerr=y_Err, fmt=None, ecolor='r', lw=2, mew=2)
    elif flag == 'd=20':
        plt.errorbar(x, y, yerr=y_Err, fmt=None, ecolor='b', lw=2, mew=2)
    elif flag == 'd=40':
        plt.errorbar(x, y, yerr=y_Err, fmt=None, ecolor='g', lw=2, mew=2)

def err_Calculate(all_Regret):

    [a,b] = all_Regret.shape
    err = np.std(all_Regret, axis=0) / np.sqrt(a)

    return err

def add_Annotation():

    plt.xlabel('Step n', fontsize=16)
    plt.ylabel('Regret', fontsize=16)
    plt.legend(loc=2, frameon=False, fontsize=14)

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

if __name__ == '__main__':

    ##############################
    # put all your csv files here
    # assume 10 iterations
    # assume run 100k
    # need to revise if run 1M
    ##############################

    # only accept Step = 100k or Step = 1M
    Exp2_Step= 100000
    Exp3_Step = 100000
    Exp41_Step = 100000
    Exp42_Step = 100000

    ##########################################
    # set max shown value for each experiments
    ##########################################

    Exp2_ymax = 2500
    Exp3_ymax = 4000
    Exp41_ymax = 1600
    Exp42_ymax = 100000

    # load Exp2 related files
    L256_LinTS_d10 = np.loadtxt('Rank_10.csv', dtype=np.float64, delimiter=',')
    L256_LinTS_d20 = np.loadtxt('Rank_20.csv', dtype=np.float64, delimiter=',')
    L256_LinTS_d40 = np.loadtxt('Rank_40.csv', dtype=np.float64, delimiter=',')

    # load Exp3 related files
    # Sub_L256_LinTS_d10 = np.loadtxt('Sub_L256_LinTS_d10_Select.csv', dtype=np.float64, delimiter=',')
    Sub_L256_LinTS_d10 = np.loadtxt('specific_Rank_10.csv', dtype=np.float64, delimiter=',')
    Sub_L256_LinTS_d20 = np.loadtxt('specific_Rank_20.csv', dtype=np.float64, delimiter=',')
    Sub_L256_LinTS_d40 = np.loadtxt('specific_Rank_40.csv', dtype=np.float64, delimiter=',')

    # load Exp4 regret related files
    L256_LinTS_K4 = np.loadtxt('Rank_4.csv', dtype=np.float64, delimiter=',')
    L256_LinTS_K8 = np.loadtxt('Rank_8.csv', dtype=np.float64, delimiter=',')
    L256_LinTS_K12 = np.loadtxt('Rank_12.csv', dtype=np.float64, delimiter=',')

    # load Exp4 reward related files
    L256_LinTS_K4_Rwd = np.loadtxt('Rank_reward_4.csv', dtype=np.float64, delimiter=',')
    L256_LinTS_K8_Rwd = np.loadtxt('Rank_reward_8.csv', dtype=np.float64, delimiter=',')
    L256_LinTS_K12_Rwd = np.loadtxt('Rank_reward_12.csv', dtype=np.float64, delimiter=',')

    combine_Exp2Exp4_Plots(L256_LinTS_d10, L256_LinTS_d20, L256_LinTS_d40,
                        Sub_L256_LinTS_d10, Sub_L256_LinTS_d20, Sub_L256_LinTS_d40,
                        L256_LinTS_K4, L256_LinTS_K8, L256_LinTS_K12,
                        L256_LinTS_K4_Rwd, L256_LinTS_K8_Rwd, L256_LinTS_K12_Rwd,
                        Exp2_Step, Exp3_Step, Exp41_Step, Exp42_Step,
                        Exp2_ymax, Exp3_ymax, Exp41_ymax, Exp42_ymax)

    ##############################
    # plot is in the same folder
    ##############################

    plt.savefig('Exp2-Exp4_2_24.eps', dpi=800, bbox_inches='tight')

    # plt.show()