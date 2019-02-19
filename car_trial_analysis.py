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
import os
import pdb
import copy
import gpmodel_library as gplib 

from analysis_utils import *

######### MAIN LOOP ###########
if __name__ == '__main__':

    # Define files for globa maxima loc, robot samples, and name. 
    # Lists should be the same length
    # maxima_files = ['/home/genevieve/Downloads/true_maxima.csv',
    #                 '/home/genevieve/Downloads/true_maxima.csv']
    # prefix = '/home/genevieve/mit-whoi/temp/'
    # max_filename = 'true_maxima.csv'
    # samp_filename = 'robot_model_modified.csv'

    # Only have a global max value in the mvi 
    # maxima_files = [prefix + '2019-02-08-17-21-09-nonmyopic_mvi_final/' + max_filename,
    #                 prefix + '2019-02-08-17-21-09-nonmyopic_mvi_final/' + max_filename,
    #                 prefix + '2019-02-08-17-21-09-nonmyopic_mvi_final/' + max_filename]

    # sample_files = [prefix + '2019-02-08-17-21-09-nonmyopic_mvi_final/' + samp_filename,
    #                 prefix + '2019-02-08-17-06-26-nonmyopic_ucb_final/' + samp_filename,
    #                 prefix + '2019-02-08-17-49-54-myopic-ucb_final/' + samp_filename]
    
    trials = ['_nonmyopic_mvi',
              '_nonmyopic_ucb',
              '_myopic_ucb'] 

    labels = ['PLUMES',
              'UCB-NONMYOPIC',
              'UCB-MYOPIC'] 

    # Filename for the logfile
    log_file_start = 'iros_car_trials'

    path = '/home/genevieve/Downloads/processed_bags/'

    # Variables for making dataframes
    all_dfs = []
    all_sample_dfs = []
    all_props = []
    all_propsy = []
    all_labels = []
    all_errx = []
    all_errz = []
    dist_dfs = []
    dist_samples_dfs = []
    dist_props = []
    dist_propsy = []
    dist_ids = []
    dist_err_x = []
    dist_err_z = []
    dist_dist_x = []
    dist_dist_z = []
    dist_entropy_x = []
    dist_entropy_z = []

    for label, trial in zip(labels, trials):
        samples = []
        max_val = []
        max_loc = []

        print "Adding for:", label
        for root, dirs, files in os.walk(path):
            for name in files:
                if 'robot_model' in name and trial in root and 'hold' not in root and 'modified' not in name:
                    samples.append(root+"/"+name)
                    print os.path.join(root, name),

                if 'true_maxima' in name and trial in root and 'hold' not in root:
                    true_maxes = np.loadtxt(os.path.join(root, name)).T
                    if true_maxes.ndim > 1:
                        true_loc = true_maxes[0, 0:2].reshape((-1, 2))
                        true_val = true_maxes[0, 2].reshape((-1, ))
                    else:
                        true_loc = true_maxes[0:2].reshape((-1, 2))
                        true_val = true_maxes[2].reshape((-1, ))

                    # max_val.append(float(ls[0].split(" ")[3]))
                    max_val.append(float(true_val))
                    max_loc.append((float(true_loc[0, 0]), float(true_loc[0, 1])))


        # Generate sample statistics
        print "\n Computing for", label, "with", len(samples), "files."
        sdata, prop, propy, err_x, err_z, d_dist_x, d_dist_z, d_hx, d_hz = make_samples_df(samples, ['x', 'y', 'z'], max_loc = max_loc, max_val = max_val, xthresh = 1.5, ythresh = 3.0)
        all_sample_dfs.append(sdata)
        all_props.append(prop)
        all_propsy.append(propy)
        all_labels.append(label)
        all_errx.append(err_x)
        all_errz.append(err_z)

        dist_dist_x.append(d_dist_x)
        dist_dist_z.append(d_dist_z)
        dist_entropy_x.append(d_hx)
        dist_entropy_z.append(d_hz)

        # Geneate data without distance truncation
        # dist_data, dist_sdata, d_props, d_propsy, ids, d_err_x, d_err_z = make_dist_dfs(values, samples, column_names, max_loc, max_val, ythresh = 3.0, xthresh = 1.5, dist_lim = 200.0, lawnmower = True)

        # Geneate data with distance truncation
        # dist_data, dist_sdata, d_props, d_propsy, ids, d_err_x, d_err_z = make_dist_dfs(values, samples, column_names, max_loc, max_val, ythresh = 3.0, xthresh = 1.5, dist_lim = 200.0)

        # dist_dfs.append(dist_data)
        # dist_samples_dfs.append(dist_sdata)
        # dist_props.append(d_props)
        # dist_propsy.append(d_propsy)
        # dist_ids.append(ids)
        # dist_err_x.append(d_err_x)
        # dist_err_z.append(d_err_z)

    # generate_stats(all_dfs, all_labels, ['distance', 'MSE', 'max_loc_error', 'max_val_error', 'max_value_info', 'info_regret'], 149, log_file_start + '_stats.txt')
    # generate_dist_stats(dist_dfs, labels, ['distance', 'MSE', 'max_loc_error', 'max_val_error', 'max_value_info', 'info_regret'], dist_ids, log_file_start + '_dist_stats.txt')

    # generate_histograms(all_sample_dfs, all_props, labels, title='All Iterations', figname=log_file_start, save_fig=False)
    generate_histograms(all_sample_dfs, all_props, labels, title='200$m$ Budget X Samples', figname=log_file_start, save_fig=False)
    generate_histograms(all_sample_dfs, all_propsy, labels, title='200$m$ Budget Y Samples', figname=log_file_start, save_fig=False)

    generate_histograms(all_sample_dfs, all_errx, labels, title='200$m$ Budget X Dist', figname=log_file_start, save_fig=False, ONLY_STATS = True)
    generate_histograms(all_sample_dfs, all_errz, labels, title='200$m$ Budget Z Dist', figname=log_file_start, save_fig=False, ONLY_STATS = True)
    
    generate_histograms(dist_samples_dfs, dist_dist_x, all_labels, title='200$m$ Budget X Star Dist', figname=log_file_start, save_fig=False, ONLY_STATS = True)
    generate_histograms(dist_samples_dfs, dist_dist_z, all_labels, title='200$m$ Budget Z Star Dist', figname=log_file_start, save_fig=False, ONLY_STATS = True)

    generate_histograms(dist_samples_dfs, dist_entropy_x, all_labels, title='200$m$ Budget X Star Entropy', figname=log_file_start, save_fig=False, ONLY_STATS = True)
    generate_histograms(dist_samples_dfs, dist_entropy_z, all_labels, title='200$m$ Budget Z Star Entropy', figname=log_file_start, save_fig=False, ONLY_STATS = True)

    # # def planning_iteration_plots(dfs, labels, param, title, end_time=149, d=20, plot_confidence=False, save_fig=False, fname='')
    # planning_iteration_plots(all_dfs, labels, 'MSE', 'Averaged MSE', 149, len(seeds), True, False, log_file_start+'_avg_mse.png')
    # planning_iteration_plots(all_dfs, labels, 'max_val_error', 'Val Error', 149, len(seeds), True, False, log_file_start+'_avg_rac.png')
    # planning_iteration_plots(all_dfs, labels, 'max_loc_error', 'Loc Error', 149, len(seeds), True, False, log_file_start+'_avg_ireg.png')

    # (dfs, sdfs, labels, param, title, dist_lim=150., granularity=10, d=20, plot_confidence=False, save_fig=False, fname=''):
    # distance_iteration_plots(dist_dfs, dist_ids, labels, 'MSE', 'Averaged MSE', 200., 100, len(seeds), True, False, '_avg_mse_dist.png' )
    # distance_iteration_plots(dist_dfs, dist_ids, labels, 'max_value_info', 'Reward Accumulation', 200., 100, len(seeds), True, False, '_avg_rac_dist.png' )
    # distance_iteration_plots(dist_dfs, dist_ids, labels, 'info_regret', 'Info Regret', 200., 100, len(seeds), True, False, '_avg_ireg_dist.png' )
    # distance_iteration_plots(dist_dfs, dist_ids, labels, 'max_loc_error', 'Loc Error', 200., 100, len(seeds), True, False, '_avg_locerr_dist.png' )


    plt.show()
