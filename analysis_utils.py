# !/usr/bin/python

import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import math
from matplotlib.colors import LogNorm
from matplotlib import cm
import matplotlib.colors as mcolors
import os
import pdb
import copy

import gpmodel_library as gplib 
import aq_library as aqlib
from scipy.spatial import distance

plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['figure.figsize'] = (17,10)

''' Predict the maxima of a GP model '''
def predict_max(xvals, zvals, ranges = [0.0, 10.0, 0.0, 10.0], LEN = 1.0, VAR = 100.0, NOISE = 0.5):
    # If no observations have been collected, return default value
    # if xvals is None or True: # TODO: remeber to change this!
    if xvals is None or True: # TODO: remeber to change this!
        print "Skipping maxima prediction!"
        return np.array([0., 0.]), 0.

    GP = gplib.GPModel(ranges = ranges, lengthscale = LEN, variance = VAR, noise = NOISE)
    GP.add_data(xvals, zvals) 
    ''' First option, return the max value observed so far '''
    #return self.GP.xvals[np.argmax(GP.zvals), :], np.max(GP.zvals)

    ''' Second option: generate a set of predictions from model and return max '''
    # Generate a set of observations from robot model with which to predict mean
    x1vals = np.linspace(ranges[0], ranges[1], 100)
    x2vals = np.linspace(ranges[2], ranges[3], 100)
    x1, x2 = np.meshgrid(x1vals, x2vals, sparse = False, indexing = 'xy') 

    data = np.vstack([x1.ravel(), x2.ravel()]).T
    observations, var = GP.predict_value(data)        
    max_loc, max_val = data[np.argmax(observations), :], np.max(observations)

    # fig2, ax2 = plt.subplots(figsize=(8, 8))
    # plot = ax2.contourf(x1, x2, observations.reshape(x1.shape), 25, cmap = 'viridis')
    # scatter = ax2.scatter(GP.xvals[:, 0], GP.xvals[:, 1], c='k', s = 20.0, cmap = 'viridis')

    # scatter = ax2.scatter(data[:, 0], data[:, 1], c='b', s = 10.0, cmap = 'viridis')
    # scatter = ax2.scatter(max_loc[0], max_loc[1], c='r', s = 20.0, cmap = 'viridis')
    # plt.show()

    return max_loc, max_val

''' Quantify entropy of star distribution and visaulize the star heatmap '''
def star_max_dist(xvals, zvals, true_loc, true_val, PATH, ranges = [0.0, 10.0, 0.0, 10.0], LEN = 1.0, VAR = 100.0, NOISE = 0.5):
    # If no observations have been collected, return default value
    if xvals is None: #TODO: remember to change this
        print "Skipping star analysis prediction!"
        return 0.0, 0.0, 0.0, 0.0

    GP = gplib.GPModel(ranges = ranges, lengthscale = LEN, variance = VAR, noise = NOISE)
    GP.add_data(xvals, zvals) 
  
    # If files already exist, simply read in. Othrewise, sample maxima
    # and create files.
    try:
        sampled_maxes = np.loadtxt(os.path.join(PATH, 'sampled_maxes_dist.csv')).T
        max_locs = sampled_maxes[:, 0:2].reshape((-1, 2))
        max_vals = sampled_maxes[:, 2].reshape((-1, 1))
        SAVE_FLAG = False 

    except:
        max_vals, max_locs, func = aqlib.sample_max_vals(GP, t = 0, nK = 20)
        max_vals = np.array(max_vals).reshape((-1, 1))
        max_locs = np.array(max_locs).reshape((-1, 2))
        np.savetxt(os.path.join(PATH, 'sampled_maxes_dist.csv'), np.vstack((max_locs.T, max_vals.T)))
        SAVE_FLAG = True 

    true_loc = np.array(true_loc).reshape((-1, 2))
    true_val = np.array(true_val).reshape((-1, 1))

    # Compute average distance from stars to true loc
    dist_loc = distance.cdist(max_locs, true_loc, 'euclidean')
    dist_val = distance.cdist(max_vals, true_val, 'euclidean')

    NBINS = 50
    RANGE = np.array([(ranges[0], ranges[1]), (ranges[2], ranges[3])])

    # Create the star heatmap
    if SAVE_FLAG:
        plt.figure(figsize=(8,8))
        # plt.hist2d(max_locs[:, 0], max_locs[:, 1], bins = NBINS, normed = True, range = RANGE, cmap = 'magma', norm=mcolors.LogNorm())
        plt.hist2d(max_locs[:, 0], max_locs[:, 1], bins = NBINS, normed = True, range = RANGE, cmap = 'viridis')
        plt.colorbar()
        plt.savefig(os.path.join(PATH, 'star_heatmap.png'))
        # plt.show()
        plt.close()

    # Compute the histrogram entropy of the star distribution
    ALPHA = 0.99
    hist, xbins, ybins = np.histogram2d(max_locs[:, 0], max_locs[:, 1], bins = NBINS, normed = True, range = RANGE)
    uniform = np.ones(hist.shape) / np.sum(np.ones(hist.shape))
    histnorm = ALPHA * hist + (1. - ALPHA) * uniform 
    histnorm = histnorm / np.sum(histnorm)
    entropy_x = -np.sum(histnorm[histnorm > 0.0] * np.log(histnorm[histnorm > 0.0]))
    plt.imshow(hist)
    plt.show()

    # Uniform santiy check
    # uniform = np.ones(hist.shape) / np.sum(np.ones(hist.shape))
    # unifrom_entropy = -np.sum(uniform[uniform > 0.0] * np.log(uniform[uniform > 0.0]))
    # print "Entropy of a uniform distribution:", unifrom_entropy

    hist_z, xbins_z = np.histogram(max_vals, bins = NBINS, density = True)
    uniform = np.ones(hist_z.shape) / np.sum(np.ones(hist_z.shape))
    hist_z = hist_z / np.sum(hist_z)
    entropy_z = -np.mean(np.log(hist_z[hist_z > 0.0]))

    # Save statistics
    if SAVE_FLAG:
        np.savetxt(os.path.join(PATH, 'star_stats.csv'), np.array([np.mean(dist_loc), np.mean(dist_val), entropy_x, entropy_z]))
        np.savetxt(os.path.join(PATH, 'star_loc_variance.csv'), np.array(dist_val))
        np.savetxt(os.path.join(PATH, 'star_val_variance.csv'), np.array(dist_loc))

    return np.mean(dist_loc), np.mean(dist_val), entropy_x, entropy_z
        
def make_df(file_names, column_names):
    d = file_names[0]
    data = pd.read_table(d, delimiter = " ", header=None)
    data = data.T
    if data.shape[1] > len(column_names):
        data = pd.read_table(d, delimiter = " ", header=None, skipfooter = data.shape[1] - len(column_names))
        data = data.T
        data.columns = column_names
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

def make_samples_df(file_names, column_names, max_loc, max_val, xthresh=1.5, ythresh = 2.50):
    prop = []
    propy = []
    err_x = []
    err_z = []
    dist_loc = []
    dist_val = []
    H_x = []
    H_z = []

    # Read in the first file and compute statistics
    d = file_names[0]

    sdata = pd.read_table(d, delimiter = " ", header = None)
    sdata = sdata.T
    sdata.columns = column_names

    # Compute the nubmer of data points within a threshold of the optmizal x value
    sdata.loc[:, 'Distance'] = sdata.apply(lambda x: np.sqrt((x['x']-max_loc[0][0])**2+(x['y']-max_loc[0][1])**2), axis=1)
    prop.append(float(len(sdata[sdata.Distance <= xthresh]))/len(sdata))

    # Compute the number of data points within a threshold of the optimal z value
    sdata.loc[:, 'YDistance'] = sdata.apply(lambda x: np.sqrt((x['z']-max_val[0])**2), axis=1)
    propy.append(float(len(sdata[sdata.YDistance <= ythresh]))/len(sdata))

    xvals = np.array(sdata[['x', 'y']]).reshape((-1, 2))
    zvals = np.array([sdata['z']]).reshape((-1, 1))
    true_loc = np.array([max_loc[0][0], max_loc[0][1]])
    true_val = max_val[0]

    # Compute the error in x and z from inferred GP
    # For sim trials 
    max_x, max_z = predict_max(xvals, zvals)
    # For car trials 
    # max_x, max_z = predict_max(xvals, zvals, ranges = [-3.15, 3.15, -2.0, 2.0], LEN = 0.8, VAR = 100.0, NOISE = 2.0)
    # TOOD: fix how these parameters are set

    err_x.append(np.linalg.norm(max_x - true_loc))
    err_z.append(np.linalg.norm(max_z - true_val))

    # Compute the star heatmap for the current GP
    # d_loc, d_val, entropy_x, entropy_z = star_max_dist(xvals, zvals, true_loc, true_val, PATH = os.path.split(file_names[0])[0])
    # For now, don't need to compute for the entire dataset, since we only care about truncated valued

    # For sim trials 
    d_loc, d_val, entropy_x, entropy_z = star_max_dist(None, None, true_loc, true_val, PATH = os.path.split(file_names[0])[0])
    # For car trials 
    # d_loc, d_val, entropy_x, entropy_z = star_max_dist(xvals, zvals, true_loc, true_val, PATH = os.path.split(file_names[0])[0], ranges = [-3.15, 3.15, -2.0, 2.0], LEN = 0.8, VAR = 100.0, NOISE = 2.0)
    # TOOD: fix how these parameters are set

    dist_loc.append(d_loc)
    dist_val.append(d_val)
    H_x.append(entropy_x)
    H_z.append(entropy_z)

    for i,m in enumerate(file_names[1:]):
        # Read in the next filename data
        temp_data = pd.read_table(m, delimiter = " ", header=None)
        temp_data = temp_data.T
        temp_data.columns = column_names

        # Compute the average distance of samples from maxima in x and maxima in y
        temp_data.loc[:,'Distance'] = temp_data.apply(lambda x: np.sqrt((x['x']-max_loc[i+1][0])**2+(x['y']-max_loc[i+1][1])**2),axis=1)
        temp_data.loc[:,'YDistance'] = temp_data.apply(lambda x: np.sqrt((x['z']-max_val[i+1])**2),axis=1)

        # Append proporations of samples within epsilon of x and delta of y
        prop.append(float(len(temp_data[temp_data.Distance <= xthresh]))/len(temp_data))
        propy.append(float(len(temp_data[temp_data.YDistance <= ythresh]))/len(temp_data))
    
        # Compute the error in x and z from inferred GP
        xvals = np.array(temp_data[['x', 'y']]).reshape((-1, 2))
        zvals = np.array([temp_data['z']]).reshape((-1, 1))
        true_loc = np.array([max_loc[i+1][0], max_loc[i+1][1]])
        true_val = max_val[i+1]

        # For sim trials 
        max_x, max_z = predict_max(xvals, zvals)
        # For car trials 
        # max_x, max_z = predict_max(xvals, zvals, ranges = [-3.15, 3.15, -2.0, 2.0], LEN = 0.8, VAR = 100.0, NOISE = 2.0)
        # TOOD: fix how these parameters are set

        err_x.append(np.linalg.norm(max_x - true_loc))
        err_z.append(np.linalg.norm(max_z - true_val))

        # For sim trials 
        d_loc, d_val, entropy_x, entropy_z = star_max_dist(None, None, true_loc, true_val, PATH = os.path.split(m)[0])
        # For car trials 
        # d_loc, d_val, entropy_x, entropy_z = star_max_dist(xvals, zvals, true_loc, true_val, PATH = os.path.split(m)[0], ranges = [-3.15, 3.15, -2.0, 2.0], LEN = 0.8, VAR = 100.0, NOISE = 2.0)
        # TOOD: fix how these parameters are set

        dist_loc.append(d_loc)
        dist_val.append(d_val)
        H_x.append(entropy_x)
        H_z.append(entropy_z)

        sdata = sdata.append(temp_data)

    return sdata, prop, propy, err_x, err_z, dist_loc, dist_val, H_x, H_z

def generate_stats(dfs, labels, params, end_time=149, fname='stats.txt'):
    f = open(fname, 'a')
    for p in params:
        f.write('-------\n')
        f.write(str(p) + '\n')
        for df,label in zip(dfs, labels):
            df_end = df[df.time == end_time]
            f.write(label + ' ' + str(df_end[p].mean()) + ', ' + str(df_end[p].std()) + '\n')
    f.close()

def generate_histograms(dfs, props, labels, title, figname='', save_fig=False, ONLY_STATS = False):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y']

    print '\n\n ------------------', title, '------------------'
    print '---- Mean and STD for each proportion ---'
    for q,m in enumerate(props):
        print labels[q] + ': ' + str(np.mean(m)) + ', ' + str(np.std(m)) 
    print '---- Median and IQR for each proportion ---'
    for q,m in enumerate(props):
        print labels[q] + ': ' + str(np.median(m)) + ', ' + str(sp.stats.iqr(m))
    print '---- MIN and MAX for each proportion ---'
    for q,m in enumerate(props):
        print labels[q] + ': ' + str(np.min(m)) + ', ' + str(np.max(m))

    if ONLY_STATS == True:
        return

    print '---- Sig Test, PLUMES v other ----'
    for q,m in enumerate(props):
        print labels[q] + ' v PLUMES: ' + str(stats.ttest_ind(props[0],m, equal_var=False))

    print '---- Convergence % ----'
    for q,m in enumerate(props):
        count = 0
        for pro in m:
            if pro >= 0.10:
                count += 1
        print labels[q] + ': ' + str(float(count)/len(m)) 

    fig, axes = plt.subplots(1, len(dfs), sharey = True)
    hist = [None]*len(dfs)
    bins = [None]*len(dfs)
    for i in range(0, len(dfs)):
        if title == '200$m$ Budget Y Samples':
            hist[i], bins[i], _ = axes[i].hist(dfs[i]['YDistance'].values, bins = np.linspace(min(dfs[0]['YDistance'].values), max(dfs[0]['YDistance'].values), np.floor(max(dfs[0]['YDistance'].values)-min(dfs[0]['YDistance'].values))), color = colors[i], density = False)
            axes[i].set_title(labels[i])
        elif title == '200$m$ Budget X Samples':
            hist[i], bins[i], _ = axes[i].hist(dfs[i]['Distance'].values, bins = np.linspace(min(dfs[0]['Distance'].values), max(dfs[0]['Distance'].values), np.floor(max(dfs[0]['Distance'].values)-min(dfs[0]['Distance'].values))), color = colors[i], density = False)
            axes[i].set_title(labels[i])
        else:
            hist[i], bins[i], _= axes[i].hist(dfs[i]['Distance'].values, bins = np.linspace(min(dfs[0]['Distance'].values), max(dfs[0]['Distance'].values), np.floor(max(dfs[0]['Distance'].values)-min(dfs[0]['Distance'].values))), color = colors[i], density = False)
            axes[i].set_title(labels[i])

    axes[0].set_ylabel('Count')
    # axes[].set_xlabel('Distance ($m$) from Global Maximizer')
    # plt.suptitle(title+': Distance ($m$) from Global Maximizer', va='bottom')

    if save_fig == True:
        plt.savefig(figname+'_agg_samples.png')

    fig, axes = plt.subplots(1, 1, sharey = True)
    for i in range(0, len(dfs)):
        if title == '200$m$ Budget Y Samples':
            axes.plot(np.cumsum(hist[i]), color = colors[i], label = labels[i])
            axes.legend()
        elif title == '200$m$ Budget X Samples':
            axes.plot(np.cumsum(hist[i]), color = colors[i], label = labels[i])
            axes.legend()
        else:
            axes.plot(np.cumsum(hist[i]), color = colors[i], label = labels[i])
            axes.legend()

    axes.set_ylabel('CDF')
    plt.suptitle(title+': Distance ($m$) from Global Maximizer', va='bottom')

    fig = plt.figure()
    plt.boxplot(props, meanline=True, showmeans=True, labels=labels)
    # plt.ylim((0.,1.))

    # plt.bar(np.arange(len(dfs)), [np.mean(m) for m in props], yerr = np.array(yerr).T, color = colors[0:len(props)])#yerr=[np.std(m) for m in props], color=colors[0:len(props)])
    plt.ylabel('Proportion of Samples')
    # plt.title(title+': Proportion of Samples within $1.5m$ of Global Maximizer', fontsize=32)

    if save_fig == True:
        plt.savefig(figname+'_prop_samples.png')

def planning_iteration_plots(dfs, labels, param, title, end_time=149, d=20, plot_confidence=False, save_fig=False, fname=''):
    fig = plt.figure()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y']
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

def make_dist_dfs(data_dfs, sample_dfs, column_names, max_loc, max_val, ythresh = 2.50, xthresh=1.5, dist_lim=150.0, lawnmower = False):
    all_dist = pd.DataFrame()
    all_samps = pd.DataFrame()
    all_props = []
    all_propsy = []
    all_statsids = []

    all_errx = []
    all_errz = []
    all_distx = []
    all_distz = []
    all_hx = []
    all_hz = []

    for f,g,m,v in zip(data_dfs, sample_dfs, max_loc, max_val):
        temp_df = make_df([f], column_names)

        # Make samples dataframe and compute stats
        temp_sdf,  \
        temp_prop, \
        temp_propy, \
        temp_xerr, \
        temp_zerr, \
        temp_distx, \
        temp_distz, \
        temp_hx, \
        temp_hz = make_samples_df(file_names = [g], 
                                  column_names = ['x','y','z'], 
                                  max_loc = [m], 
                                  max_val = [v],  
                                  xthresh = xthresh, 
                                  ythresh = ythresh)

        # Truncate these stats by distance
        dtemp, \
        dstemp, \
        dprop, \
        dpropy, \
        stats_id, \
        d_xerr, \
        d_zerr, \
        d_distx, \
        d_distz, \
        d_hx, \
        d_hz = truncate_by_distance(temp_df, 
                                    temp_sdf, 
                                    max_loc = [m],
                                    max_val = [v], 
                                    dist_lim = dist_lim, 
                                    xthresh = xthresh, 
                                    ythresh = ythresh, 
                                    lawnmower = lawnmower,
                                    file_names = [g])

        all_dist = all_dist.append(dtemp)
        all_samps = all_samps.append(dstemp)
        all_props.append(float(dprop))
        all_propsy.append(float(dpropy))
        all_statsids.append(stats_id)

        all_errx.append(d_xerr)
        all_errz.append(d_zerr)
        all_distx.append(d_distx)
        all_distz.append(d_distz)
        all_hx.append(d_hx)
        all_hz.append(d_hz)


    return all_dist, all_samps, all_props, all_propsy, all_statsids, all_errx, all_errz, all_distx, all_distz, all_hx, all_hz


def truncate_by_distance(df, sample_df, max_loc, max_val, dist_lim=250.0, xthresh=1.5, ythresh = 2.50, lawnmower = False, file_names = ''):
    temp_df = df[df['distance'] < dist_lim]
    last_samp_x = temp_df['robot_loc_x'].values[-1]
    last_samp_y = temp_df['robot_loc_y'].values[-1]

    stats_id = temp_df.index[-1]

    temp_sidx = sample_df[np.isclose(sample_df['x'], last_samp_x)& \
                          np.isclose(sample_df['y'], last_samp_y)].index

    if lawnmower == False:
        candidates = []
        if len(temp_sidx) == 1:
            candidates = temp_sidx
        else:
            for i in temp_sidx:
                dist = 0
                disty = 0
                last = 0
                for j in range(1, i):
                    dist += np.sqrt((sample_df['x'].values[last]-sample_df['x'].values[j])**2 + (sample_df['y'].values[last]-sample_df['y'].values[j])**2)
                    last = j
                if dist <= dist_lim:
                    candidates.append(i)
        idx = candidates[-1]
        temp_sdf = sample_df[sample_df.index<idx]
    else:
        temp_sdf  = sample_df

    prop = float(len(temp_sdf[temp_sdf.Distance < xthresh]))/len(temp_sdf)
    propy = float(len(temp_sdf[temp_sdf.YDistance < ythresh]))/len(temp_sdf)

    # Compute the error in x and z from inferred GP
    xvals = np.array(temp_sdf[['x', 'y']]).reshape((-1, 2))
    zvals = np.array([temp_sdf['z']]).reshape((-1, 1))

    # Predict maxima
    max_x, max_z = predict_max(xvals, zvals)
    err_x = (np.linalg.norm(max_x - np.array(max_loc)))
    print "max val:", max_val
    err_z = (np.linalg.norm(max_z - max_val[0]))

    # Predict star distribution 
    d_loc, d_val, entropy_x, entropy_z = star_max_dist(xvals, zvals, max_loc, max_val[0], PATH = os.path.split(file_names[0])[0])

    return temp_df, temp_sdf, prop, propy, stats_id, err_x, err_z, d_loc, d_val, entropy_x, entropy_z


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

def distance_iteration_plots(dfs, trunids, labels, param, title, dist_lim=150., granularity=300, averager=20, plot_confidence=False, save_fig=False, fname=''):
    fig = plt.figure()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y']
    interp_granularity = float(dist_lim/granularity) #uniform distance measures on which to interpolate the data for averaging
    
    # set interpolation params
    dist_markers = {'distance':[], 'misc':[]}
    for i in range(granularity+1):
        dist_markers['distance'].append(i*interp_granularity)
        dist_markers['misc'].append(0)
    interp = pd.DataFrame.from_dict(dist_markers)
    interp = interp.set_index('distance')

    info = []
    info2 = []
    # iterate through each seed group
    for df,group_sidx in zip(dfs,trunids):
        extracted = []
        averaged = {}
        average = {}
        errd = {}
        df2 = copy.copy(df)
        # seperate out the individual seed
        for ids in group_sidx:
            temp = None
            extracted_temp = None
            d = df2[0:ids]
            # extract stuff we care about
            temp = pd.concat([d['distance'], d[param]],axis=1,keys=['distance',param])
            temp = temp.set_index('distance')
            temp = temp.append(interp)
            temp = temp.sort_index()
            # interpolate over standard index so that we can do averaging and standard deviation
            temp = temp.interpolate('index')
            extract_temp = temp.loc[dist_markers['distance']]
            extracted.append(extract_temp)
            df2 = df2[ids+1:]
        # walk through the interpolation index and find the average and error for each seed at those points
        for dist in dist_markers['distance']:
            average[dist] = []
            for zz,e in enumerate(extracted):
                average[dist].append(e.loc[dist][param])
            avge = np.mean(average[dist])
            stde = np.std(average[dist])
            averaged[dist] = avge
            errd[dist] = stde
        # log the stats for the seed
        info.append(averaged)
        info2.append(errd)

    # plot results
    k = 0
    for m,n in zip(info,info2):
        m_ordered = sorted(m)
        m_vals = [m[v] for v in m_ordered]
        m_errs = [n[v] for v in m_ordered]
        plt.plot(m_ordered, m_vals, color=colors[k], label=labels[k])

        if plot_confidence==True:
            y1 = [l + f for l,f in zip(m_vals,m_errs)]
            y2 = [l - f for l,f in zip(m_vals,m_errs)]
            plt.fill_between(m_ordered, y1, y2, color=colors[k], alpha=0.2)
        k += 1


    plt.legend(fontsize=30)
    plt.xlabel("Distance($m$) Travelled")
    plt.ylabel(param)


    if save_fig:
        plt.savefig(fname)
    plt.title(title)

