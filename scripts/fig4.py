# %%
## Imports

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matlab.engine
from sklearn.metrics import accuracy_score
import pickle
import itertools
from scipy import stats
from joblib import Parallel, delayed

sys.path.insert(0, '../lib')
from utils import build_marg_arrays, normalize_and_concat, load_trial_data_h5
from analyses import decoder_sweep_parallel, run_decoding_sweep, \
                     extract_windowed_features, cross_temporal_decoding


# %%
## Load data

# select participant
participant = 'T16'
# participant = 'T11'

# whether to (re)compute each decoding analysis and save .pkl, or just use the
# cached .pkl results to plot (warns if a .pkl is missing)
RUN_DECODING_SWEEP = True

# load delay-period data (aligned to the target cue)
filename_delay = f'./data/fig4_{participant}_delay.h5'
trial_data_delay = load_trial_data_h5(filename_delay)

# load movement-period data (aligned to the go cue)
filename_move = f'./data/fig4_{participant}_move.h5'
trial_data_move = load_trial_data_h5(filename_move)

# define save path for plots
save_plot_filepath = './plots/fig4/'
if not os.path.exists(save_plot_filepath):
    os.makedirs(save_plot_filepath)


# %%
## Get arrays from the trialized data dicts

# drop trials with NaNs in either the delay- or move-aligned neural data
nan_mask1 = np.isnan(trial_data_delay['spike_data_all']).any(axis=(1,2))
nan_mask2 = np.isnan(trial_data_move['spike_data_all']).any(axis=(1,2))
nan_mask = nan_mask1 | nan_mask2

# get neural data for the delay and movement periods
neural_data_all = trial_data_delay['spike_data_all'][~nan_mask]
neural_data_move_all = trial_data_move['spike_data_all'][~nan_mask]

# get SBP data for the delay and movement periods
sbp_data_all = trial_data_delay['sbp_data_all'][~nan_mask]
sbp_data_move_all = trial_data_move['sbp_data_all'][~nan_mask]

# get per-trial task parameters and the trial time axis
move_speed_all = trial_data_delay['move_speed_all'][~nan_mask]
move_dist_all = trial_data_delay['move_dist_all'][~nan_mask]
vert_orientation_all = trial_data_delay['vert_orientation_all'][~nan_mask].astype(float)
delay_data_all = trial_data_delay['delay_data_all'][~nan_mask]
block_id_data_all = trial_data_delay['block_id_data_all'][~nan_mask]
no_go_bool_all = trial_data_delay['no_go_bool_all'][~nan_mask]
t_data = trial_data_delay['t_data']

# get timing info (bin width in ms, and the bin index of the cues)
t_data = trial_data_delay['t_data']
TS = t_data[1]-t_data[0]
T_START = int(t_data[0])
GO_CUE = -int(T_START/TS)
T_START_MOVE = int(t_data[0])
GO_CUE_MOVE = -int(T_START_MOVE/TS)

# get channel count and the kept (active) channel indices
N_CHANNELS = neural_data_all.shape[2]
keep_chans_all = trial_data_delay['keep_chans_all']

# convert firing rates to Hz
neural_data_all *= 1000/TS
neural_data_move_all *= 1000/TS


# %%
## Get trial condition variables

# split signed target distance into distance (magnitude) and direction (sign)
dist_data_all = np.abs(move_dist_all)
dir_data_all = np.sign(move_dist_all)
speed_data_all = move_speed_all

# combine direction and screen orientation into a target angle (0 - 2*pi)
angle_data_all = np.pi/2 - dir_data_all * np.pi/2 + vert_orientation_all * np.pi/2
angle_data_all[angle_data_all<0] += 2*np.pi
angle_data_all = np.round(angle_data_all,4)
angle_data_unique = np.unique(angle_data_all)

# get unique levels of each condition variable
dist_data_unique = np.unique(dist_data_all)
speed_data_unique = np.unique(speed_data_all)

# get unique delay durations and the shortest usable delay (>= 0.8 s)
delay_data_unique = np.unique(delay_data_all)
delay_min = delay_data_unique[delay_data_unique >= 0.8].min()

# build condition lists
CONDITIONS = [angle_data_unique, dist_data_unique, speed_data_unique]
CONDITION_ARRAYS = [angle_data_all, dist_data_all, speed_data_all]


# %%
## Plot PSTHs for example channels

# example channels to plot, per participant
if participant == 'T16':
    CHS = [56, 54, 37]
if participant == 'T11':
    CHS = [29, 46, 37]

# delay period plotting window: -200 ms to the end of the shortest delay
T_STA = int(-200/TS)
T_END = int(delay_min*1000/TS)
t_plot = np.arange(-200, 1000, TS)

# generate plot, one column per example channel
_, axs = plt.subplots(3, len(CHS), figsize=(len(CHS)*3.2, 6), sharex=True, sharey='col', constrained_layout=True)

# loop over example channels, one column per channel
for ich, ch in enumerate(CHS):

    # 1. plot PSTHs split by direction
    ax = axs[0, ich]

    # loop over target angles
    for id, angle in enumerate(angle_data_unique):

        # select trials of this direction
        mask = (angle_data_all == angle)

        # compute mean and SEM firing rate across trials
        data_plot = neural_data_all[mask][:,GO_CUE+T_STA:GO_CUE+T_END,ch]
        data_plot_mean = data_plot.mean(axis=0)
        data_plot_sem = data_plot.std(axis=0, ddof=1) / np.sqrt(data_plot.shape[0])

        # darker shade for the second angle
        if id == 1:
            color = (0.3, 0.3, 0.6)
        else:
            color = (0.5, 0.5, 0.9)

        # plot mean ± SEM
        ax.fill_between(t_plot, data_plot_mean - data_plot_sem, data_plot_mean + data_plot_sem,
                         alpha=0.2, color=color, linewidth=0.0)
        ax.plot(t_plot, data_plot_mean, color=color, linewidth=2)

    # 2. plot PSTHs split by distance
    ax = axs[1, ich]

    # loop over target distances
    for id, dist in enumerate(dist_data_unique):

        # select trials of this distance
        mask = (dist_data_all == dist)

        # compute mean and SEM firing rate across trials
        data_plot = neural_data_all[mask][:,GO_CUE+T_STA:GO_CUE+T_END,ch]
        data_plot_mean = data_plot.mean(axis=0)
        data_plot_sem = data_plot.std(axis=0, ddof=1) / np.sqrt(data_plot.shape[0])

        # darker shade for larger distances
        if id == 2:
            color = (0.1, 0.3, 0.1)
        elif id == 1:
            color = (0.3, 0.6, 0.3)
        else:
            color = (0.5, 0.9, 0.5)

        # plot mean ± SEM
        ax.fill_between(t_plot, data_plot_mean - data_plot_sem, data_plot_mean + data_plot_sem,
                         alpha=0.2, color=color, linewidth=0.0)
        ax.plot(t_plot, data_plot_mean, color=color, linewidth=2)

    # 3. plot PSTHs split by speed
    ax = axs[2, ich]

    # loop over instructed speeds
    for iv, speed in enumerate(speed_data_unique):

        # select trials of this speed
        mask = (speed_data_all == speed)

        # compute mean and SEM firing rate across trials
        data_plot = neural_data_all[mask][:,GO_CUE+T_STA:GO_CUE+T_END,ch]
        data_plot_mean = data_plot.mean(axis=0)
        data_plot_sem = data_plot.std(axis=0, ddof=1) / np.sqrt(data_plot.shape[0])

        # darker shade for higher speeds
        if iv == 2:
            color = (0.3, 0.1, 0.1)
        elif iv == 1:
            color = (0.6, 0.3, 0.3)
        else:
            color = (0.9, 0.5, 0.5)

        # plot mean ± SEM
        ax.fill_between(t_plot, data_plot_mean - data_plot_sem, data_plot_mean + data_plot_sem,
                         alpha=0.2, color=color, linewidth=0.0)
        ax.plot(t_plot, data_plot_mean, color=color, linewidth=2)

    ylims = axs[2,ich].get_ylim()
    ylims_range = ylims[1] - ylims[0]

    for i, ax in enumerate(axs[:,ich]):

        # misc. subplot settings
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])

        # scale bar offset
        scale_offset = [-20, -ylims_range*0.1]
        ax.set_xlim(-300+scale_offset[0], 1100)
        ax.set_ylim(ylims[0]+scale_offset[1], ylims[1])

        # draw target cue
        ax.plot([0,0],[ylims[0], ylims[1]], color='y', linestyle='-', linewidth=3)

        # only draw the firing rate and time scale bars on the bottom row
        if i == 2:
            ax.plot([-300+scale_offset[0]/3,-300+scale_offset[0]/3],[ylims[0]+scale_offset[1]/2, ylims[0]+scale_offset[1]/2+2],
                    color='k', linestyle='-', linewidth=5)
            ax.plot([-300+scale_offset[0]/2,-300+200],[ylims[0]+scale_offset[1]/2, ylims[0]+scale_offset[1]/2],
                    color='k', linestyle='-', linewidth=5)

    # report the plotted channel and the trial count
    print(f'ich: {ich}, actual channel: {keep_chans_all[ch]}')
    print(f'trials: {angle_data_all.shape[0]}')

# save pdf
savepath = os.path.join(save_plot_filepath, f'fig4_{participant}_psths.pdf')
plt.savefig(savepath, format='pdf')

plt.show()


# %%
## Per-channel three-way ANOVA — direction, distance, and speed tuning

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

# containers for the per-channel p-value of each main effect and interaction
anova_results_direction = np.zeros((N_CHANNELS))
anova_results_distance = np.zeros((N_CHANNELS))
anova_results_speed = np.zeros((N_CHANNELS))
anova_results_direction_distance = np.zeros((N_CHANNELS))
anova_results_direction_speed = np.zeros((N_CHANNELS))
anova_results_distance_speed = np.zeros((N_CHANNELS))
anova_results_direction_distance_speed = np.zeros((N_CHANNELS))

# loop over channels, testing tuning of the mean delay-period rate
for ch in range(N_CHANNELS):

    # delay window: 400 ms after the target cue to the end of the shortest delay
    T_STA = int(400/TS)
    T_END = int(delay_min*1000/TS)

    # trial table of conditions and the channel's mean rate over the window
    df = pd.DataFrame({
        'direction': angle_data_all,
        'distance': dist_data_all,
        'speed': speed_data_all,
        'neural_data_ch': neural_data_all[:,
                                        int(GO_CUE+T_STA):int(GO_CUE+T_END),
                                        ch].mean(axis=1),
    })

    # run ANOVA
    model_full = ols("""neural_data_ch ~ C(direction) * C(distance) * C(speed)""", data = df).fit()
    anova = sm.stats.anova_lm(model_full, typ = 2)

    # get FDR-corrected p-values
    pvals = anova['PR(>F)'].values[:-1]
    corrected_pvals = multipletests(pvals, method='fdr_bh')[1]

    # save the p-value of each effect for this channel
    anova_results_direction[ch] = corrected_pvals[0]
    anova_results_distance[ch] = corrected_pvals[1]
    anova_results_speed[ch] = corrected_pvals[2]
    anova_results_direction_distance[ch] = corrected_pvals[3]
    anova_results_direction_speed[ch] = corrected_pvals[4]
    anova_results_distance_speed[ch] = corrected_pvals[5]
    anova_results_direction_distance_speed[ch] = corrected_pvals[6]


# %%
## Plot percentage of channels tuned to each task feature

# significance threshold
sig = 0.05

plt.figure(figsize=(5, 4.2))

# 1. plot direction tuning (main effect alone, and with its interactions)
anova_sig = anova_results_direction < sig
anova_sig2 = ((anova_results_direction < sig) +
             (anova_results_direction_distance < sig) +
                (anova_results_direction_speed < sig) +
             (anova_results_direction_distance_speed < sig))
sig_percent = anova_sig.sum(axis=0) / N_CHANNELS * 100
sig_percent2 = anova_sig2.sum(axis=0) / N_CHANNELS * 100
plt.bar(0, sig_percent2, color=(0.1, 0.1, 0.6), label='Direction', width=1)
plt.bar(0, sig_percent, color=(0.2, 0.2, 0.9), label='Combined', width=1)

# report the tuned channel percentages and indices
print(f'\nDirection ANOVA: {sig_percent:.1f}% / {sig_percent2:.1f}% of channels')
print(np.where(anova_sig)[0])
print(np.where(anova_sig2)[0])

# 2. plot distance tuning (main effect alone, and with its interactions)
anova_sig = anova_results_distance < sig
anova_sig2 = ((anova_results_distance < sig) +
               (anova_results_direction_distance < sig) +
                (anova_results_distance_speed < sig) +
                (anova_results_direction_distance_speed < sig))
sig_percent = anova_sig.sum(axis=0) / N_CHANNELS * 100
sig_percent2 = anova_sig2.sum(axis=0) / N_CHANNELS * 100
plt.bar(2.5, sig_percent2, color=(0.1, 0.6, 0.1), label='Distance', width=1)
plt.bar(2.5, sig_percent, color=(0.2, 0.9, 0.2), label='Combined', width=1)

# report the tuned channel percentages and indices
print(f'\nDistance ANOVA: {sig_percent:.1f}% / {sig_percent2:.1f}% of channels')
print(np.where(anova_sig)[0])
print(np.where(anova_sig2)[0])

# 3. plot speed tuning (main effect alone, and with its interactions)
anova_sig = anova_results_speed < sig
anova_sig2 = ((anova_results_speed < sig) +
               (anova_results_direction_speed < sig) +
                (anova_results_distance_speed < sig) +
                (anova_results_direction_distance_speed < sig))
sig_percent = anova_sig.sum(axis=0) / N_CHANNELS * 100
sig_percent2 = anova_sig2.sum(axis=0) / N_CHANNELS * 100
plt.bar(5, sig_percent2, color=(0.6, 0.1, 0.1), label='Speed', width=1)
plt.bar(5, sig_percent, color=(0.9, 0.2, 0.2), label='Combined', width=1)

# report the tuned channel percentages and indices
print(f'\nSpeed ANOVA: {sig_percent:.1f}% / {sig_percent2:.1f}% of channels')
print(np.where(anova_sig)[0])
print(np.where(anova_sig2)[0])

# annotate participant and channel count
plt.text(x=2.5, y=100, s=f'{participant}',  fontsize=20,
            color='k', ha='center', va='top')
plt.text(x=2.5, y=90, s=f'{N_CHANNELS} channels',  fontsize=16,
            color='k', ha='center', va='top') #, fontweight='bold')

# misc. plot settings
plt.xticks([0, 2.5, 5], ['Direction', 'Distance', 'Speed'],
           rotation=0, fontsize=16)
plt.yticks(np.arange(0, 101, 20), fontsize=14)
plt.ylabel('% of channels\ntuned to feature', fontsize=16)
plt.gca().axes.spines[['top', 'right']].set_visible(False)
plt.xlim(-1.5, 6.5)
plt.ylim(0, 100)

# save pdf
savepath = os.path.join(save_plot_filepath, f'fig4_{participant}_tuned_chans.pdf')
plt.savefig(savepath, format='pdf')

plt.show()


# %%
## Normalize neural data

# soft-normalization constants for the spike and SBP features
SOFT_NORM_SPIKES = 1e-9
SOFT_NORM_SBP = 1e-9

# soft-normalize per block and optionally concatenate the SBP features
neural_data_norm_all, neural_data_move_norm_all = normalize_and_concat(
    neural_data_all, sbp_data_all, neural_data_move_all, sbp_data_move_all,
    block_id_data_all, GO_CUE, TS,
    USE_SBP=True, SOFT_NORM_SPIKES=SOFT_NORM_SPIKES, SOFT_NORM_SBP=SOFT_NORM_SBP,
)


# %%
## Build dPCA arrays

# build condition-averaged and per-trial arrays
X, trialsX, trialNum = build_marg_arrays(neural_data_norm_all,
                                        CONDITIONS, CONDITION_ARRAYS,)

# delay window: target cue to the end of the shortest delay
T_STA = int(000/TS)
T_END = int(delay_min*1000/TS)

# single-trial and trial-averaged firing rates handed to dPCA
firingRates = trialsX[:,:,:,:,GO_CUE+T_STA:GO_CUE+T_END,:]
firingRatesAverage = X[:,:,:,:,GO_CUE+T_STA:GO_CUE+T_END]
margNames = ['Direction', 'Distance', 'Speed', 'Condition-independent', 'Dir./dist./speed interactions']

# plotting order of the marginalizations, with titles and colors
marg_order = [3, 0, 1, 2, 4]
marg_titles = [margNames[3], margNames[0], margNames[1], margNames[2], margNames[4]]
color_list = [(0.6, 0.6, 0.6),
              (0.45, 0.45, 0.9),
              (0.32, 0.8, 0.32),
              (0.9, 0.35, 0.35),
              (0.64, 0.36, 0.9)]


# %%
## Run dPCA

# start MATLAB engine
eng = matlab.engine.start_matlab()

# add paths to libraries
eng.addpath(eng.genpath('../lib/dPCA/matlab'), nargout=0)
eng.addpath(eng.genpath('../lib/utils_dpca'), nargout=0)

# convert numpy arrays to MATLAB arrays
firingRates_matlab = matlab.double(firingRates.tolist())
firingRatesAverage_matlab = matlab.double(firingRatesAverage.tolist())
trialNum_matlab = matlab.double(trialNum.tolist())
margNames_matlab = margNames

# run dpca script for 3 marginalizations
output = eng.dpca_analysis_3d(firingRates_matlab,
                            firingRatesAverage_matlab,
                            trialNum_matlab,
                            margNames_matlab,)

# close MATLAB engine
eng.quit()

# get the dPCA encoder (V) and decoder (W) axes, and which marginalization each belongs to
dpcaV = np.array(output['V'])
dpcaW = np.array(output['W'])
whichMarg = np.array(output['whichMarg']).astype(int)


# %%
## Compute variance in each dPCA component

# mean-center the trial-averaged firing rates
Xfull = firingRatesAverage - np.nanmean(firingRatesAverage, axis=(1,2,3,4), keepdims=True)

# flatten mean-centered trial-averaged firing rates
Xfull_flat = Xfull.reshape(Xfull.shape[0], -1)

# container for the per-component R2
comp_variance = []

# loop over components, computing the variance each one accounts for
for i in range(dpcaW.shape[-1]):
    SSerr = np.sum((Xfull_flat - dpcaV[:,i].reshape(-1,1) @ dpcaW[:,i].reshape(1,-1) @ Xfull_flat)**2)
    SStot = np.sum(Xfull_flat**2)
    R2 = (SStot - SSerr) / (SStot)
    comp_variance.append(R2)

comp_variance = np.array(comp_variance)


# %%
## Plot variance explained by dPC type

# generate plot
plt.figure(figsize=(6, 3))

# loop over marginalizations, plotting the variance their components jointly explain
for marg in range(5):
    # compute the variance explained jointly by this marginalization's components
    marg_idx = np.where(whichMarg == marg_order[marg]+1)[1]
    SSerr = np.sum((Xfull_flat - dpcaV[:,marg_idx] @ dpcaW[:,marg_idx].T @ Xfull_flat)**2)
    SStot = np.sum(Xfull_flat**2)
    R2 = (SStot - SSerr) / (SStot)
    var = R2
    print(var)

    # draw the bar and label it with the variance explained
    plt.barh(5-marg, var*100, color=color_list[marg], height=0.6)
    plt.text(var*100+0.25, 5-marg, f'{var*100:.1f}%', va='center', ha='left', fontsize=16, color=color_list[marg], fontweight='bold')

# misc. plot settings
plt.gca().spines[['top', 'right']].set_visible(False)
plt.gca().yaxis.set_ticks([])
if participant == 'T16':
    plt.gca().xaxis.set_ticks(np.arange(0, 25+1, 5))
else:
    plt.gca().xaxis.set_ticks(np.arange(0, 35+1, 5))
plt.gca().xaxis.set_tick_params(labelsize=14)
plt.gca().yaxis.set_tick_params(labelsize=16)
plt.gca().set_yticks(np.arange(1, 6),
                     ['Dir./dist./speed\ninteractions', 'Speed', 'Distance', 'Direction', 'Condition\nindepedent'],
                     ha='right') #, fontweight='bold')
plt.ylim(0+0.3, 6-0.3)
for marg in range(5):
    plt.gca().get_yticklabels()[4-marg].set_color(color_list[marg])
plt.xlabel('% of total variance\nexplained by dPC type', fontsize=16)
plt.title(f'{participant}', fontsize=20)

# save pdf
savepath = os.path.join(save_plot_filepath, f'fig4_{participant}_dpca_var_bars.pdf')
plt.savefig(savepath, format='pdf')

plt.show()


# %%
## Plot dPCA projections

# plotting window relative to the target cue (ms)
T0 = -200
TF = 1000

# convert the plotting window to bin indices
T_STA = int(GO_CUE+(T0/TS))
T_END = int(GO_CUE+(TF/TS) + 0)

# get single-trial data over the plotting window
Xdpca = trialsX[:,:,:,:,T_STA:T_END]
t_pca = t_data[T_STA:T_END]

# mean-center on the delay-period mean across conditions
xdpca_mean = np.nanmean(np.nanmean(trialsX[:,:,:,:,GO_CUE:T_END], axis=-1, keepdims=True),
                        axis=(1,2,3,4), keepdims=True)
Xcen = Xdpca - xdpca_mean

# project onto the dPCA decoder axes
Z = Xcen.T @ dpcaW

# generate 2x4 grid of subplots
_, axs = plt.subplots(nrows=2, ncols=4, figsize=(4*5,2*4*1.1), sharex=True, sharey=False, facecolor='w', constrained_layout=True)

# y limits per participant
if participant == 'T16':
    ylims = [-1, 1]
    yticks = [-1, -0.5, 0, 0.5, 1]
elif participant == 'T11':
    ylims = [-2, 2]
    yticks = [-2, -1, 0, 1, 2]

# loop over the subplot grid (one marginalization per column, its first two dPCs per row)
for col in range(4):

    for row in range(2):

        # map each grid position to a marginalization and component index
        if col == 0 and row == 0:
            marg = 0
            pc = 0
        elif col == 0 and row == 1:
            marg = 0
            pc = 1
        elif col == 1 and row == 0:
            marg = 1
            pc = 0
        elif col == 1 and row == 1:
            marg = 1
            pc = 1
        elif col == 2 and row == 0:
            marg = 2
            pc = 0
        elif col == 2 and row == 1:
            marg = 2
            pc = 1
        elif col == 3 and row == 0:
            marg = 3
            pc = 0
        elif col == 3 and row == 1:
            marg = 3
            pc = 1
        else:
            continue

        # components belonging to this marginalization
        marg_i = np.where(whichMarg == marg_order[marg]+1)[1]

        ax = axs[row,col]

        # loop over conditions
        for cond_set in itertools.product(*CONDITIONS):

            # select the trials of this condition
            masks = [cond_set[i] == CONDITION_ARRAYS[i] for i in range(len(CONDITIONS))]
            mask = np.all(masks, axis=0)

            # index of this condition along each grid axis
            i0 = np.where(cond_set[0] == angle_data_unique)[0][0]
            i1 = np.where(cond_set[1] == dist_data_unique)[0][0]
            i2 = np.where(cond_set[2] == speed_data_unique)[0][0]

            # compute mean and SEM of the projection across trials
            Zmarg = np.nanmean(Z[:,:,i2,i1,i0,marg_i[pc]], axis=0)
            Zmarg_sem = np.nanstd(Z[:,:,i2,i1,i0,marg_i[pc]], axis=0, ddof=1) / np.sqrt(np.sum(mask))

            linestyle = '-'
            linewidth = 2

            # color by this marginalization's condition variable
            if marg == 0:
                color = (0.4, 0.4, 0.4)
            elif marg == 1:
                c1 = (np.pi - cond_set[0])/np.pi * 0.5
                c2 = c1 + 0.5
                c1 = np.clip(c1, 0, 1)
                c2 = np.clip(c2, 0, 1)
                color = (c1, c1, c2)
            elif marg == 2:
                c1 = (600 - cond_set[1])/600 * 3/2 * 0.4 + 0.1
                c2 = (600 - cond_set[1])/600 * 3/2 * 0.6 + 0.3
                c1 = np.clip(c1, 0, 1)
                c2 = np.clip(c2, 0, 1)
                color = (c1, c2, c1)
            elif marg == 3:
                c1 = (600 - cond_set[2])/600 * 3/2 * 0.4 + 0.1
                c2 = (600 - cond_set[2])/600 * 3/2 * 0.6 + 0.3
                c1 = np.clip(c1, 0, 1)
                c2 = np.clip(c2, 0, 1)
                color = (c2, c1, c1)

            # plot mean ± SEM
            ax.fill_between(t_pca, Zmarg - Zmarg_sem, Zmarg + Zmarg_sem,
                            color=color, alpha=0.15, linewidth=0)
            ax.plot(t_pca, Zmarg,color=color, linewidth=linewidth, linestyle=linestyle, alpha=1)

        # draw target cue
        ax.axvline(0, color='y', linestyle='-', linewidth=4)

        # only label the projection axis on the leftmost column
        if col == 0:
            ax.set_ylabel(f'Projection onto dPC {row+1}' , fontsize=22)
            ax.set_yticks(yticks)

        # draw amplitude scale bar (must follow set_yticks to re-trigger the y autoscale)
        ax.plot([-250, -250], [-0.05, 0.05], c='k', linewidth=6)

        # widen the y range of near-flat components
        ylims = ax.get_ylim()
        if ylims[1] - ylims[0] < 0.25:
            ax.set_ylim(-0.125, 0.125)
            ylims = ax.get_ylim()

        # only draw the time scale bar under the bottom-left subplot
        if col == 0 and row == 1:
            ax.plot([100, 300], [ylims[0], ylims[0]], c='k', linewidth=6)

        # annotate the variance explained by this component
        marg_variance_i = comp_variance[marg_i[pc]] * 100
        ax.text(t_pca[-1], ylims[0] + (ylims[1]-ylims[0])*0.01*0, f'{marg_variance_i:2.2f}%',
                color='k', fontsize=20, ha='right', va='bottom', fontweight='normal')

        # title each column with its marginalization
        if row == 0:
            ax.set_title(f'{marg_titles[marg]}', fontsize=24, color=np.array(color_list[marg])*0.8, fontweight='bold')

        # misc. subplot settings
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlim(T0-100, TF)

# save pdf
savepath = os.path.join(save_plot_filepath, f'fig4_{participant}_dpca_traces.pdf')
plt.savefig(savepath, format='pdf')

plt.show()


# %%
## Run cross-validated decoding sweep — prep and move periods

# run the decoding sweep and cache it
if RUN_DECODING_SWEEP:

    # loop over the prep and move epochs
    for epoch in ['prep', 'move']:

        np.random.seed(42)

        # time window for decoding
        T_STA = int(GO_CUE+400/TS)
        T_END = int(GO_CUE+delay_min*1000/TS)

        # loop over the decoded variables
        for yi, y_var in enumerate([angle_data_all, dist_data_all, speed_data_all]):

            # select trials for this epoch and decoded variable
            trial_mask = (delay_data_all >= 0.8)
            if epoch == 'move':
                trial_mask = trial_mask & (no_go_bool_all == 0)
            if yi == 1:  # distance: only extreme levels, all 3 speeds
                trial_mask = trial_mask & ((dist_data_all == dist_data_unique[0]) | (dist_data_all == dist_data_unique[-1]))
            elif yi == 2:  # speed: only extreme levels, all 3 distances
                trial_mask = trial_mask & ((speed_data_all == speed_data_unique[0]) | (speed_data_all == speed_data_unique[-1]))

            # use move-aligned data for the move epoch
            X_data = neural_data_move_norm_all if epoch == 'move' else neural_data_norm_all

            # run the decoding sweep and unpack its results
            res = run_decoding_sweep(
                X_data, y_var, CONDITIONS, CONDITION_ARRAYS,
                trial_mask, T_STA, T_END,
            )
            true_all, pred_all = res['true_all'], res['pred_all']
            true_all_split, pred_all_split = res['true_all_split'], res['pred_all_split']
            acc_chance_all, acc_test_all = res['acc_chance_all'], res['acc_test_all']
            sorted_idxs = res['sorted_idxs']

            # save the decoding sweep results
            savepath = os.path.join(save_plot_filepath, f'fig4_{participant}_decoding_{epoch}_{yi}.pkl')
            with open(savepath, 'wb') as f:
                pickle.dump({'true_all_split': true_all_split,
                                'pred_all_split': pred_all_split,
                                'true_all': true_all,
                                'pred_all': pred_all,
                                'acc_chance_all': acc_chance_all,
                                'acc_test_all': acc_test_all,
                                'sorted_idxs': sorted_idxs
                                }, f)


# %%
## Plot decoding accuracy for both participants

from scipy.stats import norm

# one light/dark color pair per decoded feature (T16, T11)
colors2 = [(0.45, 0.45, 0.9), (0.22, 0.22, 0.6),
            (0.32, 0.8, 0.32), (0.15, 0.55, 0.15),
            (0.9, 0.35, 0.35), (0.6, 0.18, 0.18),]

# loop over epochs and decoded features, one figure per combination
for epoch in ['prep', 'move']:

    for i, feature in enumerate(['Direction', 'Distance', 'Speed']):

        # index of this feature in the saved sweep results
        if feature == 'Direction':
            dec = 0
            var_idx_j = 0
        elif feature == 'Distance':
            dec = 1
            var_idx_j = 1
        elif feature == 'Speed':
            dec = 2
            var_idx_j = 2

        plt.figure(figsize=(2.5, 6))

        print(f'\n{epoch} period')

        # loop over participants, one bar (or bar pair) each
        for j, part in enumerate(['T16', 'T11']):

            # load the separate decoder's cached sweep results
            savepath = os.path.join(save_plot_filepath, f'fig4_{part}_decoding_{epoch}_{dec}.pkl')
            # only plot participants whose sweep .pkl exists
            if not os.path.exists(savepath):
                print(f'WARNING: {savepath} not found. Set RUN_DECODING_SWEEP = True to (re)generate it.')
                continue
            with open(savepath, 'rb') as f:
                decoding_data = pickle.load(f)
                acc_test_all  = decoding_data['acc_test_all']
                acc_chance_all = decoding_data['acc_chance_all']

            # compute the binomial 95% CI on the mean accuracy, and the chance level
            sem = np.sqrt(np.mean(acc_test_all) * (1 - np.mean(acc_test_all))) / np.sqrt(acc_test_all.shape[0] - 1)
            acc_test_CI  = norm.ppf(0.975) * sem
            chance_folds = np.mean(acc_chance_all)
            print(f'{part} {feature} — Acc.: {acc_test_all.mean():.3f}  Chance: {chance_folds:.3f}')

            # plot accuracy ± CI with the chance level drawn over it
            pos_sep = j * 2.5
            plt.bar([pos_sep], acc_test_all.mean(), yerr=acc_test_CI, width=1, color=colors2[2*i+j])
            plt.plot([pos_sep-0.8, pos_sep+0.8], [chance_folds, chance_folds], 'k--')

        # misc. plot settings
        plt.xticks([0, 2.5], ['T16', 'T11'], rotation=0, va='top', fontsize=20)
        plt.xlim([-1.5, 4])
        plt.yticks(fontsize=16)
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.ylim([0, 1])
        plt.ylabel('Accuracy', fontsize=20)
        plt.title(f'{epoch} — {feature}', fontsize=9)

        # save pdf
        savepath = os.path.join(save_plot_filepath, f'fig4_decoding_bars_{epoch}_{i}.pdf')
        plt.savefig(savepath, format='pdf')

        plt.show()


# %%
## Decoding generalization across time — direction, distance, and speed

np.random.seed(42)

# sweep parameters: step size, window length, C values, and number of CV splits
STEP = 40
WINDOW_LEN = 100
C_RANGE = np.logspace(-3.5, -1, 11)
CV_SPLITS = 10

# decoded variables, their names, and the color of their diagonal traces
y_vars  = [angle_data_all, dist_data_all, speed_data_all]
y_names = ['Direction', 'Distance', 'Speed']
DIAG_COLORS = [(0.45, 0.45, 0.9), (0.32, 0.8, 0.32), (0.9, 0.35, 0.35)]

# loop over the decoded variables, one pair of figures each
for y_var, y_name, diag_color in zip(y_vars, y_names, DIAG_COLORS):

    # per-variable trial mask: extreme levels only for dist/speed
    base_mask = (delay_data_all >= 0.8) & (no_go_bool_all == 0)
    if y_name == 'Distance':
        trial_mask = base_mask & ((dist_data_all == dist_data_unique[0]) | (dist_data_all == dist_data_unique[-1]))
    elif y_name == 'Speed':
        trial_mask = base_mask & ((speed_data_all == speed_data_unique[0]) | (speed_data_all == speed_data_unique[-1]))
    else:
        trial_mask = base_mask

    # relabel the variable's levels as class indices
    y_all = np.unique(y_var, return_inverse=True)[1][trial_mask]
    n_trials = y_all.shape[0]

    # only decode variables with at least two trials
    if n_trials < 2:
        print(f'skipping { y_name }: insufficient trials')
        continue

    # shuffle labels and then sort by class so every K-th index is a balanced fold
    idxs = np.arange(n_trials)
    np.random.shuffle(idxs)
    sorted_idxs = idxs[np.argsort(y_all[idxs])]

    # sampled time points (ms) and the width of the averaging window, in bins
    t_ms_delay_end = 1000
    t_ms = np.arange(-200, t_ms_delay_end + STEP, STEP)
    win_samp = int(WINDOW_LEN / TS)

    # run the cross-temporal decoding and cache it, or load the cached results
    savepath_pkl = os.path.join(save_plot_filepath, f'fig4_{participant}_{y_name.lower()}_tgen_4block.pkl')
    if RUN_DECODING_SWEEP:

        # extract window-averaged delay-period features at each time point
        T_end_delay = GO_CUE + (t_ms / TS).astype(int)
        X_delay     = extract_windowed_features(neural_data_norm_all[trial_mask], T_end_delay, win_samp)

        # same for the movement period
        T_end_move  = GO_CUE_MOVE + (t_ms / TS).astype(int)
        X_move      = extract_windowed_features(neural_data_move_norm_all[trial_mask], T_end_move,  win_samp)

        # train and test at every pair of time steps within and across the two epochs
        acc_dd, acc_dm, acc_md, acc_mm, chance_empirical = cross_temporal_decoding(
            X_delay, X_move, y_all, sorted_idxs,
            cv_splits=CV_SPLITS, c_range=C_RANGE, label=y_name,
        )
        n_delay_t = acc_dd.shape[0]
        n_move_t  = acc_mm.shape[0]

        # save the cross-temporal decoding results
        with open(savepath_pkl, 'wb') as f:
            pickle.dump({'acc_dd': acc_dd, 'acc_mm': acc_mm,
                         'acc_dm': acc_dm, 'acc_md': acc_md,
                         't_ms_delay': t_ms, 't_ms_move': t_ms,
                         'step': STEP, 'WINDOW_LEN': WINDOW_LEN,
                         'chance_empirical': chance_empirical}, f)
    else:
        # if results are not cached, warn the user and skip plotting
        if not os.path.exists(savepath_pkl):
            print(f'WARNING: {savepath_pkl} not found. Set RUN_DECODING_SWEEP = True to (re)generate it.')
            continue

        # load the cached results from the pickle file
        with open(savepath_pkl, 'rb') as f:
            pkl = pickle.load(f)
        acc_dd = pkl['acc_dd']
        acc_mm = pkl['acc_mm']
        chance_empirical = pkl['chance_empirical']
        n_delay_t = acc_dd.shape[0]
        n_move_t  = acc_mm.shape[0]

    # number of time points in each epoch
    n_tgt = n_delay_t
    n_mov = n_move_t

    # get the per-split diagonals (train time == test time) for delay and move
    diag_dd = np.array([np.diag(acc_dd[:,:,s]) for s in range(acc_dd.shape[2])])
    diag_mm = np.array([np.diag(acc_mm[:,:,s]) for s in range(acc_mm.shape[2])])
    diag_dd_mean = diag_dd.mean(axis=0)
    diag_mm_mean = diag_mm.mean(axis=0)

    # compute the 95% CI across CV splits
    n_splits = diag_dd.shape[0]
    t_crit = stats.t.ppf(0.975, df=n_splits - 1)
    diag_dd_CI = t_crit * diag_dd.std(ddof=1, axis=0) / np.sqrt(n_splits)
    diag_mm_CI = t_crit * diag_mm.std(ddof=1, axis=0) / np.sqrt(n_splits)

    # generate 1x2 grid of subplots (delay, move) for the time-resolved decoding
    fig = plt.figure(figsize=(5.5, 2.4), facecolor='w')
    fig.subplots_adjust(top=0.72)
    gs_diag = fig.add_gridspec(1, 2, width_ratios=[n_tgt, n_mov], wspace=100/1200)
    ax_d = fig.add_subplot(gs_diag[0,0])
    ax_m = fig.add_subplot(gs_diag[0,1], sharey=ax_d)

    DIAG_COLOR = diag_color

    # 1. plot delay panel: mean ± 95% CI, with the target cue marked in yellow
    ax_d.plot(t_ms, diag_dd_mean, color=DIAG_COLOR, linewidth=2.5)
    ax_d.fill_between(t_ms, diag_dd_mean - diag_dd_CI, diag_dd_mean + diag_dd_CI, color=DIAG_COLOR, alpha=0.25)
    ax_d.axvline(0, color='y', lw=2.5, zorder=3)

    # misc. subplot settings
    ax_d.set_xlabel('Delay period time (ms)', fontsize=11)
    ax_d.set_xlim(t_ms[0]-STEP/2, t_ms[-1]+STEP/2)
    ax_d.set_xticks([0, t_ms_delay_end//2, t_ms_delay_end])
    ax_d.tick_params(axis='x', labelsize=10)
    ax_d.tick_params(axis='y', labelsize=10)
    ax_d.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_d.set_ylim(0, 1.02)
    ax_d.spines[['top', 'right']].set_visible(False)
    ax_d.spines['left'].set_position(('outward', 10))
    ax_d.set_axisbelow(True)
    ax_d.grid(True, color='lightgray', lw=0.6, zorder=0)

    # 2. plot move panel: mean ± 95% CI, with the go cue marked in green
    ax_m.plot(t_ms, diag_mm_mean, color=DIAG_COLOR, linewidth=2.5)
    ax_m.fill_between(t_ms, diag_mm_mean - diag_mm_CI, diag_mm_mean + diag_mm_CI, color=DIAG_COLOR, alpha=0.25)
    ax_m.axvline(0, color='g', lw=2.5, zorder=3)

    # misc. subplot settings
    ax_m.set_xlabel('Movement period time (ms)', fontsize=11)
    ax_m.set_xlim(t_ms[0]-STEP/2, t_ms[-1]+STEP/2)
    ax_m.set_xticks([0, t_ms_delay_end//2, t_ms_delay_end])
    ax_m.tick_params(axis='x', labelsize=10)
    ax_m.tick_params(axis='y', left=False, labelleft=False)
    ax_m.spines[['top', 'right', 'left']].set_visible(False)
    ax_m.set_axisbelow(True)
    ax_m.grid(True, color='lightgray', lw=0.6, zorder=0)
    fig.suptitle(f'{participant}\nDecoding accuracy through time', fontsize=13, y=0.95)

    # draw the empirical chance line spanning both panels
    fig.canvas.draw()
    from matplotlib.transforms import blended_transform_factory
    from matplotlib.lines import Line2D as Line2D
    trans = blended_transform_factory(fig.transFigure, ax_d.transData)
    x0 = ax_d.get_position().x0
    x1 = ax_m.get_position().x1
    fig.add_artist(Line2D([x0, x1], [chance_empirical, chance_empirical], transform=trans,
                           color='k', lw=1.5, ls='--', zorder=10, clip_on=False))

    # save pdf
    savepath = os.path.join(save_plot_filepath, f'fig4_{participant}_tgen_diagonal_{y_name.lower()}.pdf')
    plt.savefig(savepath, format='pdf', bbox_inches='tight')

    plt.show()

# %%
