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
import pdb

plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28
plt.rcParams['axes.labelsize'] = 32
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['figure.figsize'] = (17,10)


def make_df(file_names, column_names):
    d = file_names[0]
    data = pd.read_table(d, delimiter = " ", header=None)
    data = data.T
    if data.shape[1] > len(column_names):
        data = pd.read_table(d, delimiter = " ", header=None, skipfooter = data.shape[1] - len(column_names))
        data = data.T
        data.columns = column_names
        #data.columns = column_names[0:-2]
        #column_names = column_names[0:-2]
    else:
        data.columns = column_names

    for m in file_names[1:]:
        temp_data = pd.read_table(m, delimiter = " ", header=None)
        temp_data = temp_data.T

        # Added because some data now has mes reward printing; we want the dataframe to have the same dimensions for now
        if temp_data.shape[1] > len(column_names):
            temp_data = pd.read_table(m, delimiter = " ", header=None, skipfooter = temp_data.shape[1] - len(column_names))
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
    prop.append(float(len(sdata[sdata.Distance <= thresh]))/len(sdata))
    for i,m in enumerate(file_names[1:]):
        temp_data = pd.read_table(m, delimiter = " ", header=None)
        temp_data = temp_data.T
        temp_data.columns = column_names
        temp_data.loc[:,'Distance'] = temp_data.apply(lambda x: np.sqrt((x['x']-max_loc[i+1][0])**2+(x['y']-max_loc[i+1][1])**2),axis=1)
        prop.append(float(len(temp_data[temp_data.Distance <= thresh]))/len(temp_data))
        sdata = sdata.append(temp_data)

    return sdata, prop

def generate_stats(dfs, labels, params, end_time=149, fname='stats.txt'):
    f = open(fname, 'a')
    for p in params:
        f.write('-------\n')
        f.write(str(p) + '\n')
        for df,label in zip(dfs, labels):
            df_end = df[df.time == end_time]
            f.write(label + ' ' + str(df_end[p].mean()) + ', ' + str(df_end[p].std()) + '\n')
    f.close()

def generate_histograms(dfs, props, labels, title, figname='', save_fig=False):
    fig, axes = plt.subplots(1, len(dfs), sharey = True)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    for i in range(0, len(dfs)):
        axes[i].hist(dfs[i]['Distance'].values, bins = np.linspace(min(dfs[0]['Distance'].values), max(dfs[0]['Distance'].values), np.floor(max(dfs[0]['Distance'].values)-min(dfs[0]['Distance'].values))), color = colors[i])
        axes[i].set_title(labels[i])

    axes[0].set_ylabel('Count')
    # axes[].set_xlabel('Distance ($m$) from Global Maximizer')
    plt.suptitle(title+': Distance ($m$) from Global Maximizer', va='bottom')

    if save_fig == True:
        plt.savefig(figname+'_agg_samples.png')

    fig = plt.figure()
    plt.bar(np.arange(len(dfs)), [np.mean(m) for m in props], yerr=[np.std(m) for m in props], color=colors[0:len(props)])
    plt.xticks(np.arange(len(dfs)), labels)
    plt.ylabel('Proportion of Samples')
    plt.title(title+': Average Proportion of Samples within $1.5m$ of Global Maximizer')

    if save_fig == True:
        plt.savefig(figname+'_prop_samples.png')

def planning_iteration_plots(dfs, labels, param, title, end_time=149, d=20, plot_confidence=False, save_fig=False, fname=''):
    fig = plt.figure()
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for k,df in enumerate(dfs):
        temp = [0 for m in range(end_time)]
        temp_v = []

        for i in range(d-1):
            stemp = []
            for j in range(end_time):
                stemp.append((df[df.time == j][param].values[i]))
            temp = [m+n for m,n in zip(temp, stemp)]
            temp_v.append(stemp)

        vtemp = []
        for i in range(end_time):
            temp1 = []
            for m in temp_v:
                temp1.append(m[i])
            vtemp.append(np.std(temp1))

        plt.plot([l/d for l in temp], colors[k], label=labels[k])
        if plot_confidence:
            x = [i for i in range(end_time)]
            y1 = [l/d + m for l,m in zip(temp,vtemp)]
            y2 = [l/d - m for l,m in zip(temp,vtemp)]
            plt.fill_between(x, y1, y2, color=colors[k], alpha=0.2)

    plt.legend(fontsize=30)
    plt.xlabel("Planning Iteration")
    plt.ylabel(param)
    
    if save_fig:
        plt.savefig(fname)
    plt.title(title)

def make_dist_dfs(data_dfs, sample_dfs, column_names, max_loc, thresh=1.5, dist_lim=150.0):
    all_dist = pd.DataFrame()
    all_samps = pd.DataFrame()
    all_props = []
    all_statsids = []
    for f,g,m in zip(data_dfs, sample_dfs, max_loc):
        temp_df = make_df([f], column_names)
        temp_sdf, temp_prop = make_samples_df([g], ['x','y','a'], [m], thresh)
        dtemp, dstemp, dprop, stats_id = truncate_by_distance(temp_df, temp_sdf, dist_lim=dist_lim, thresh=thresh)
        all_dist = all_dist.append(dtemp)
        all_samps = all_samps.append(dstemp)
        all_props.append(float(dprop))
        all_statsids.append(stats_id)

    return all_dist, all_samps, all_props, all_statsids

def truncate_by_distance(df, sample_df, dist_lim=250.0, thresh=1.5):
    temp_df = df[df['distance'] < dist_lim]
    last_samp_x = temp_df['robot_loc_x'].values[-1]
    last_samp_y = temp_df['robot_loc_y'].values[-1]

    stats_id = temp_df.index[-1]
    temp_sidx = sample_df[np.isclose(sample_df['x'], last_samp_x)& \
                          np.isclose(sample_df['y'], last_samp_y)].index

    candidates = []
    if len(temp_sidx) == 1:
        candidates = temp_sidx
    else:
        for i in temp_sidx:
            dist = 0
            last = 0
            for j in range(1, i):
                dist += np.sqrt((sample_df['x'].values[last]-sample_df['x'].values[j])**2 + (sample_df['y'].values[last]-sample_df['y'].values[j])**2)
                last = j
            #print "Dist:", dist, "Lim:", dist_lim
            if dist < dist_lim:
                candidates.append(i)
    idx = candidates[-1]
    temp_sdf = sample_df[sample_df.index<idx]

    prop = float(len(temp_sdf[temp_sdf.Distance < thresh]))/len(temp_sdf)
    return temp_df, temp_sdf, prop, stats_id

def generate_dist_stats(dfs, labels, params, ids, fname='stats.txt'):
    f = open(fname, 'a')
    for p in params:
        f.write('-------\n')
        f.write(str(p) + '\n')
        for df,label,idx in zip(dfs, labels,ids):
            df_end = []
            temp_df = df[p].values
            for j,i in enumerate(idx):
                df_end.append(temp_df[i])
                temp_df = temp_df[i+1:]
            f.write(label + ' ' + str(np.mean(df_end)) + ', ' + str(np.std(df_end)) + '\n')
    f.close()


######### MAIN LOOP ###########
if __name__ == '__main__':
    seed_numbers = range(0, 700, 100)
    print seed_numbers
    seeds = ['seed'+ str(x) + '-' for x in seed_numbers]

    fileparams = ['pathsetfully_reachable_goal-costTrue-nonmyopicFalse-goalFalse',
                  'pathsetfully_reachable_goal-costTrue-nonmyopicFalse-goalTrue',
                  'pathsetfully_reachable_goal-costFalse-nonmyopicFalse-goalTrue',
                  #'pathsetfully_reachable_goal-costFalse-nonmyopicFalse-goalFalse',
                  'pathsetdubins-costFalse-nonmyopicFalse-goalFalse',
                  'pathsetdubins-costFalse-nonmyopicTrue-goalFalse']

    #labels = ['frpd', 'frgd', 'frgo', 'frpo', 'my', 'plumes']
    labels = ['frpd', 'frgd', 'frgo', 'my', 'plumes']
    file_start = 'all_mse'

    #path= '/home/vpreston/Documents/IPP/informative-path-planning/experiments/'
    path= '/home/genevieve/mit-whoi/informative-path-planning/experiments/'

    # variables for making dataframes
    column_names = ['time', 'info_gain','aqu_fun', 'MSE', 'hotspot_error','max_loc_error', 'max_val_error', 
                        'simple_regret', 'sample_regret_loc', 'sample_regret_val', 'regret', 'info_regret',
                        'current_highest_obs', 'current_highest_obs_loc_x', 'current_highest_obs_loc_y',
                        'robot_loc_x', 'robot_loc_y', 'robot_loc_a', 'distance']

    #get the data files
    all_dfs = []
    all_sample_dfs = []
    all_props = []
    all_labels = []
    dist_dfs = []
    dist_samples_dfs = []
    dist_props = []
    dist_ids = []

    for param, label in zip(fileparams, labels):
        p_mean = []
        p_mes = []
        p_mean_samples = []
        p_mes_samples = []

        max_val = []
        max_loc = []

        for root, dirs, files in os.walk(path):
            for name in files:
                #if 'metrics' in name and 'mean' in root and param in root and 'old_fully_reachable' not in root:
                #    for s in seeds:
                #        if s in root:
                #            p_mean.append(root+"/"+name)
                if 'metric' in name and 'mes' in root and param in root and 'old_fully_reachable' not in root:
                    for s in seeds:
                        if s in root:
                            p_mes.append(root+"/"+name)
                #elif 'robot_model' in name and 'mean' in root and param in root and 'old_fully_reachable' not in root:
                #    for s in seeds:
                #        if s in root:
                #            p_mean_samples.append(root+"/"+name)
                elif 'robot_model' in name and 'mes' in root and param in root and 'old_fully_reachable' not in root:
                    for s in seeds:
                        if s in root:
                            p_mes_samples.append(root+"/"+name)

                if 'log' in name and 'mes' in root and param in root and 'old_fully_reachable' not in root:
                    for s in seeds:
                        ls = []
                        if str(s) in root:
                            temp = open(root+'/'+name, "r")
                            for l in temp.readlines():
                                if "max value" in l:
                                    ls.append(l)
                            max_val.append(float(ls[0].split(" ")[3]))
                            # For Genevieve
                            max_loc.append((float(ls[-1].split(" ")[7].split("[")[0]), float(ls[-1].split(" ")[9].split("]")[0])))
                            # For Victoria
                            #max_loc.append((float(ls[0].split(" ")[6].split("[")[1]), float(ls[0].split(" ")[7].split("]")[0])))
        

        #mean_data = make_df(p_mean, column_names)
        mes_data = make_df(p_mes, column_names)

        # all_dfs.append(mean_data)
        all_dfs.append(mes_data)

        #mean_sdata, mean_prop = make_samples_df(p_mean_samples, ['x', 'y', 'a'], max_loc, 1.5)
        mes_sdata, mes_prop = make_samples_df(p_mes_samples, ['x', 'y', 'a'], max_loc, 1.5)

        # all_sample_dfs.append(mean_sdata)
        all_sample_dfs.append(mes_sdata)

        # all_props.append(mean_prop)
        all_props.append(mes_prop)

        # all_labels.append('mean_'+label)
        all_labels.append('mes_'+label)

        # def make_dist_dfs(data_dfs, sample_dfs, column_names, max_loc, thresh=1.5, dist_lim=150.0):
        mes_dist_data, mes_dist_sdata, mes_dist_props, mes_ids = make_dist_dfs(p_mes, p_mes_samples, column_names, max_loc, 1.5, 300.0)
        dist_dfs.append(mes_dist_data)
        dist_samples_dfs.append(mes_dist_sdata)
        dist_props.append(mes_dist_props)
        dist_ids.append(mes_ids)


    # def generate_stats(dfs, labels, params, end_time=149, fname='stats.txt'):
    generate_stats(all_dfs, all_labels, ['distance'], 149, file_start + '_stats.txt')
    generate_dist_stats(dist_dfs, all_labels, ['distance', 'MSE'], dist_ids, file_start + '_dist_stats.txt')

    # def generate_histograms(dfs, props, labels, figname='', save_fig=False)
    generate_histograms(all_sample_dfs, all_props, all_labels, title='All Iterations', figname=file_start, save_fig=False)
    generate_histograms(dist_samples_dfs, dist_props, all_labels, title='Dist Budget', figname=file_start, save_fig=False)

    # def planning_iteration_plots(dfs, labels, param, title, end_time=149, d=20, plot_confidence=False, save_fig=False, fname='')
    planning_iteration_plots(all_dfs, all_labels, 'MSE', 'Averaged MSE', 149, len(seeds), True, False, file_start+'_avg_mse.png')

    plt.show()
