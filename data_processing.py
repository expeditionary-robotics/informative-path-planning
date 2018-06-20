# !/usr/bin/python

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import math
from matplotlib.colors import LogNorm
from matplotlib import cm
import os

plt.rcParams['xtick.labelsize'] = 32
plt.rcParams['ytick.labelsize'] = 32
plt.rcParams['axes.labelsize'] = 40
plt.rcParams['axes.titlesize'] = 40
plt.rcParams['figure.figsize'] = (17,10)


def make_df(file_names, column_names):
    d = file_names[0]
    data = pd.read_table(d, delimiter = " ", header=None)
    data = data.T
    data.columns = column_names

    for m in file_names[1:]:
        temp_data = pd.read_table(m, delimiter = " ", header=None)
        temp_data = temp_data.T
        temp_data.columns = column_names
        data = data.append(temp_data)

    return data


def make_samples_df(file_names, column_names, max_loc, thresh=1.5):
    prop = []
    d = file_names[0]
    sdata = pd.read_table(d, delimiter = " ", header=None)
    sdata = sdata.T
    sdata.columns = column_names
    sdata.loc[:, 'Distance'] = sdata.apply(lambda x: np.sqrt((x['x']-max_loc[0][0])**2+(x['y']-max_loc[0][1])**2),axis=1)
    prop.append(float(len(sdata[sdata.Distance < thresh]))/len(sdata))
    print len(file_names), len(max_loc)
    for i,m in enumerate(file_names[1:]):
        temp_data = pd.read_table(m, delimiter = " ", header=None)
        temp_data = temp_data.T
        temp_data.columns = column_names
        temp_data.loc[:,'Distance'] = temp_data.apply(lambda x: np.sqrt((x['x']-max_loc[i+1][0])**2+(x['y']-max_loc[i+1][1])**2),axis=1)
        prop.append(float(len(temp_data[temp_data.Distance < thresh]))/len(temp_data))
        sdata = sdata.append(temp_data)

    return sdata, prop


def print_stats(meandf, mesdf, eidf, columns, end_time=174):
    mean_end = meandf[meandf.time == end_time]
    mes_end = mesdf[mesdf.time == end_time]
    if eidf is not None:
        ei_end = eidf[eidf.time == end_time]

    for e in columns:
        print '-------------'
        print str(e)
        print 'MEAN:    ' + str(mean_end[e].mean()) + ', ' + str(mean_end[e].std())
        print 'MES :    ' + str(mes_end[e].mean()) + ', '  + str(mes_end[e].std())
        if eidf is not None:
            print 'EI  :    ' + str(ei_end[e].mean()) + ', ' + str(ei_end[e].std()) 


def make_histograms(mean_sdata, mes_sdata, ei_sdata, figname=''):
    #make the aggregate histograms
    if ei_sdata is not None:
        fig, axes = plt.subplots(1, 3, sharey = True)
    else:
        fig, axes = plt.subplots(1, 2, sharey = True)

    axes[0].hist(mean_sdata['Distance'].values, bins = np.linspace(min(mean_sdata['Distance'].values), max(mean_sdata['Distance'].values), np.floor(max(mean_sdata['Distance'].values)-min(mean_sdata['Distance'].values))), color = 'g')
    axes[0].set_title("UCB")
    axes[1].hist(mes_sdata['Distance'].values, bins = np.linspace(min(mean_sdata['Distance'].values), max(mean_sdata['Distance'].values), np.floor(max(mean_sdata['Distance'].values)-min(mean_sdata['Distance'].values))), color = 'r')
    axes[1].set_title("PLUMES")
    if ei_sdata is not None:
        axes[2].hist(ei_sdata['Distance'].values,bins = np.linspace(min(mean_sdata['Distance'].values), max(mean_sdata['Distance'].values), np.floor(max(mean_sdata['Distance'].values)-min(mean_sdata['Distance'].values))), color = 'b')
        axes[2].set_title("EI")
    axes[1].set_xlabel('Distance ($m$) From Maxima')
    axes[0].set_ylabel('Count')
    plt.savefig(figname+'_agg_samples.png')

    # make the proportional barcharts
    if ei_sdata is not None:
        fig = plt.figure()
        plt.bar(np.arange(3), [sum(m)/len(m) for m in (mean_prop, mes_prop, ei_prop)], yerr=[np.std(m) for m in (mean_prop, mes_prop, ei_prop)], color=['g', 'r', 'b'])
        plt.xticks(np.arange(3),['UCB', 'PLUMES', 'EI'])
        plt.ylabel('Proportion of Samples')
        plt.title('Average Proportion of Samples taken within 1.5m of the True Maxima')
        plt.savefig('my_prop_samples')
    else:
        fig = plt.figure()
        plt.bar(np.arange(2), [sum(m)/len(m) for m in (mean_prop, mes_prop)], yerr=[np.std(m) for m in (mean_prop, mes_prop)], color=['g', 'r'])
        plt.xticks(np.arange(2),['UCB', 'PLUMES'])
        plt.ylabel('Proportion of Samples')
        plt.title('Average Proportion of Samples taken within 1.5m of the True Maxima')
        plt.savefig(figame+'_prop_samples')


def make_plots(mean_data, mes_data, ei_data, param, title, d=20, plot_confidence=False, save_fig=False, lab="Value", fname="fig"):
    #based upon the definition of rate of convergence
    ucb = [0 for m in range(149)]
    mes = [0 for m in range(149)]
    ei = [0 for m in range(149)]
    
    ucb_v = []
    mes_v = []
    ei_v = []
    
    for i in range(d-1):
        sm = []
        sme = []
        se = []
        for j in range(149):
            sm.append((mean_data[mean_data.time == j][param].values[i]))
            sme.append((mes_data[mes_data.time == j][param].values[i]))
            if ei_data is not None:
                se.append((ei_data[ei_data.time == j][param].values[i]))
        ucb = [m+n for m,n in zip(ucb,sm)]
        mes = [m+n for m,n in zip(mes,sme)]
        if ei_data is not None:
            ei = [m+n for m,n in zip(ei,se)]
        
        ucb_v.append(sm)
        mes_v.append(sme)
        ei_v.append(se)

        
    vucb = []
    vmes = []
    vei = []
    for i in range(149):
        t1 = []
        t2 = []
        t3 = []
        if ei_data is not None:
            for m, n, o in zip(ucb_v, mes_v, ei_v):
                t1.append(m[i])
                t2.append(n[i])
                t3.append(o[i])
            vucb.append(np.std(t1))
            vmes.append(np.std(t2))
            vei.append(np.std(t3))
        else:
            for m, n in zip(ucb_v, mes_v):
                t1.append(m[i])
                t2.append(n[i])
            vucb.append(np.std(t1))
            vmes.append(np.std(t2))
    
    fig = plt.figure()
    plt.plot([l/d for l in ucb], 'g', label='UCB')
    plt.plot([l/d for l in mes], 'r', label='PLUMES')
    if ei_data is not None:
        plt.plot([l/d for l in ei], 'b', label='EI')
    
    if plot_confidence:
        x = [i for i in range(149)]
        y1 = [l/d + m for l,m in zip(ucb,vucb)]
        y2 = [l/d - m for l,m in zip(ucb,vucb)]

        y3 = [l/d + m for l,m in zip(mes,vmes)]
        y4 = [l/d - m for l,m in zip(mes,vmes)]

        if ei_data is not None:
            y5 = [l/d + m for l,m in zip(ei,vei)]
            y6 = [l/d - m for l,m in zip(ei,vei)]

        plt.fill_between(x, y1, y2, color='g', alpha=0.2)
        plt.fill_between(x, y3, y4, color='r', alpha=0.2)
        if ei_data is not None:
            plt.fill_between(x, y5, y6, color='b', alpha=0.2)
    
    plt.legend(fontsize=30)
    plt.xlabel("Planning Iteration")
    plt.ylabel(lab)
    
    if save_fig:
        plt.savefig(fname)
    plt.title(title)
    # plt.show()



######### MAIN LOOP ###########
if __name__ == '__main__':
    seeds = ['seed0-', 'seed100-', 'seed200-', 'seed300-', 'seed400-', 'seed500-', 'seed600-', 'seed700-', 'seed800-', 'seed900-',
             'seed1000-', 'seed1100-', 'seed1200-', 'seed1300-', 'seed1400-', 'seed1500-']
    fileparams = 'pathsetdubins-costFalse-nonmyopicTrue-goalFalse'
    path= '/home/vpreston/Documents/IPP/informative-path-planning/experiments/'

    #get the data files
    f_mean = []
    f_mes = []
    f_ei = []
    
    for root, dirs, files in os.walk(path):
        for name in files:
            if 'metric' in name and fileparams in root and 'mean' in root:
                for s in seeds:
                    if str(s) in root:
                        f_mean.append(root + "/" + name)
            # elif 'metric' in name and 'pathsetdubins-costFalse-nonmyopicFalse' in root and 'exp_improve' in dirs:
            #     f_ei.append(root + "/" + name)
            elif 'metric' in name and fileparams in root and 'mes' in root:
                for s in seeds:
                    if str(s) in root:
                        f_mes.append(root + "/" + name)

    # variables for making dataframes
    l = ['time', 'info_gain','aqu_fun', 'MSE', 'hotspot_error','max_loc_error', 'max_val_error', 
                        'simple_regret', 'sample_regret_loc', 'sample_regret_val', 'regret', 'info_regret',
                        'current_highest_obs', 'current_highest_obs_loc_x', 'current_highest_obs_loc_y',
                        'robot_loc_x', 'robot_loc_y', 'robot_loc_a', 'distance']

    mean_data = make_df(f_mean, l)
    # ei_data = make_df(f_ei, l)
    mes_data = make_df(f_mes, l)

    # print_stats(mean_data, mes_data, ei_data, l)
    print_stats(mean_data, mes_data, None, l, 149)


    ######## Looking at Samples ######
    # get the robot log files
    max_val = []
    max_loc = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if 'log' in name and 'mean' in root and fileparams in root:
                for s in seeds:
                    ls = []
                    if str(s) in root:
                        temp = open(root+'/'+name, "r")
                        for l in temp.readlines():
                            if "max value" in l:
                                ls.append(l)
                        max_val.append(float(ls[-1].split(" ")[3]))
                        max_loc.append((float(ls[-1].split(" ")[6].split("[")[1]), float(ls[-1].split(" ")[7].split("]")[0])))

    # get the robot samples list
    mean_samples = []
    mes_samples = []
    # ei_samples = []

    for root, dirs, files in os.walk(path):
        for name in files:
            if 'robot_model' in name and 'mean' in root and fileparams in root:
                for s in seeds:
                    if str(s) in root:
                        mean_samples.append(root+"/"+name)
            # elif 'robot_model' in name and 'exp_improve' in root:
            #     ei_samples.append(root+"/"+name)
            elif 'robot_model' in name and 'mes' in root and fileparams in root:
                for s in seeds:
                    if str(s) in root:
                        mes_samples.append(root+"/"+name)

    mean_sdata, mean_prop = make_samples_df(mean_samples, ['x', 'y', 'a'], max_loc, 1.5)
    mes_sdata, mes_prop = make_samples_df(mes_samples, ['x', 'y', 'a'], max_loc, 1.5)
    # ei_sdata, ei_prop = make_samples_df(ei_samples, ['x', 'y', 'a'], max_loc, 1.5)

    print 'Mean value of sample proportions: ' 
    # print [sum(m)/len(m) for m in (mean_prop, mes_prop, ei_prop)]
    print [sum(m)/len(m) for m in (mean_prop, mes_prop)]
    print 'STD value of sample proportions: '
    # print [np.std(m) for m in (mean_prop, mes_prop, ei_prop)]
    print [np.std(m) for m in (mean_prop, mes_prop)]

    # make_histograms(mean_sdata, mes_sdata, ei_sdata)
    make_histograms(mean_sdata, mes_sdata, None, figname='nonmyopic_dubins')


    # ######### Looking at Mission Progression ######
    make_plots(mean_data, mes_data, None, 'max_val_error', 'Averaged Maximum Value Error, Conf', len(seeds), True, False, fname='nonmyopic_dubins_avg_valerr_conf')
    make_plots(mean_data, mes_data, None, 'max_loc_error', 'Averaged Maximum Location Error, Conf', len(seeds), True, False, fname='nonmyopic_dubins_avg_valloc_conf')
    make_plots(mean_data, mes_data, None, 'info_regret', 'Averaged Information Regret, Conf', len(seeds), True, False, fname='nonmyopic_dubins_avg_reg_conf')
    make_plots(mean_data, mes_data, None, 'MSE', 'Averaged MSE, Conf', len(seeds), True, False, fname='nonmyopic_dubins_avg_mse_conf')
    plt.show()
