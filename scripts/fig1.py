# %%
## Imports

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch
import pickle
import itertools
import matlab.engine
import sklearn
import seaborn as sns
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from scipy import stats
from joblib import Parallel, delayed

sys.path.insert(0, '../lib')
from utils import normalize_data, build_marg_arrays, normalize_and_concat, load_trial_data_h5
from analyses import compute_crossnobis_matrix, decoder_sweep_parallel, run_decoding_sweep, \
                     extract_windowed_features, cross_temporal_decoding, alignment_index


# %%
## Load data

# select participant
participant = 'T16'
# participant = 'T11'
# participant = 'T5'

# whether to (re)compute the SVM decoding sweeps and save .pkl, or just use the
# cached .pkl results to plot (warns if a .pkl is missing)
RUN_DECODING_SWEEP = True

# load delay-period data (aligned to the target cue)
filename_delay = f'./data/fig1_{participant}_delay.h5'
trial_data_delay = load_trial_data_h5(filename_delay)

# load movement-period data (aligned to the go cue)
filename_move = f'./data/fig1_{participant}_move.h5'
trial_data_move = load_trial_data_h5(filename_move)

# define save path for plots
save_plot_filepath = './plots/fig1/'
if not os.path.exists(save_plot_filepath):
    os.makedirs(save_plot_filepath)


# %%
## Get arrays from the trialized data dicts

# identify trials with NaNs in either the delay or the movement period, and drop them below
nan_mask1 = np.isnan(trial_data_delay['spike_data_all']).any(axis=(1,2))
nan_mask2 = np.isnan(trial_data_move['spike_data_all']).any(axis=(1,2))
nan_mask = nan_mask1 | nan_mask2

# get neural data for the delay and movement periods (firing rates & SBP)
neural_data_all = trial_data_delay['spike_data_all'][~nan_mask]
neural_data_move_all = trial_data_move['spike_data_all'][~nan_mask]
sbp_data_all = trial_data_delay['sbp_data_all'][~nan_mask]
sbp_data_move_all = trial_data_move['sbp_data_all'][~nan_mask]

# get EMG data if available
if 'emg_data_all' in trial_data_delay:
    emg_data_all = trial_data_delay['emg_data_all'][~nan_mask]
    emg_data_move_all = trial_data_move['emg_data_all'][~nan_mask]

# get trial info (target and start positions, delay duration, block id, no-go bool)
target_data_all = trial_data_delay['target_data_all'][~nan_mask]
start_data_all = trial_data_delay['start_data_all'][~nan_mask]
delay_data_all = trial_data_delay['delay_data_all'][~nan_mask]
block_id_data_all = trial_data_delay['block_id_data_all'][~nan_mask]
no_go_bool_all = trial_data_delay['no_go_bool_all'][~nan_mask]

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

# compute movement direction for each trial (0 - 2π) and the unique directions
angle_data_all = np.arctan2(target_data_all[:,1]-start_data_all[:,1],
                            target_data_all[:,0]-start_data_all[:,0])
angle_data_all[angle_data_all<0] += 2*np.pi
angle_data_all = np.round(angle_data_all,4)
angle_data_unique = np.unique(angle_data_all)

# get unique delay durations and the shortest usable delay (>= 0.8 s)
delay_data_unique = np.unique(delay_data_all)
delay_min = delay_data_unique[delay_data_unique >= 0.8].min()

# build condition lists
CONDITIONS = [angle_data_unique]
CONDITION_ARRAYS = [angle_data_all]


# %%
## Plot PSTHs for example channels

# example channels to plot, per participant
if participant == 'T16':
    CHS = [1, 7]
elif participant == 'T11':
    CHS = [29, 31, 54]
elif participant == 'T5':
    CHS = [29, 18, 1, 9]

# loop over example channels, one figure per channel
for ich, ch in enumerate(CHS):

    _, axs = plt.subplots(2,1, figsize=(6,3.5), sharex=True, sharey=True, constrained_layout=True)

    # 1. plot move trials
    ax = axs[0]

    # loop over conditions
    for cond_set in itertools.product(*CONDITIONS):

        # select go trials for this direction with a long enough delay
        masks = [cond_set[i] == CONDITION_ARRAYS[i] for i in range(len(CONDITIONS))]
        mask = np.all(masks, axis=0)
        mask = mask & (delay_data_all >= 0.8) & (no_go_bool_all == 0)

        # delay period plotting window: -200 ms to the end of the shortest delay
        T_STA = int(GO_CUE+(-200/TS))
        if participant == 'T5':
            T_END = int(GO_CUE+(800/TS))
        else:
            T_END = int(GO_CUE+(delay_min*1000/TS))
        t_plot = t_data[T_STA:T_END]

        # compute mean and SEM firing rate across trials
        data_plot = neural_data_all[mask][:,T_STA:T_END,ch]
        data_plot_mean = data_plot.mean(axis=0)
        data_plot_std = data_plot.std(axis=0, ddof=1) / np.sqrt(data_plot.shape[0])

        # color by movement direction
        color = colors.hsv_to_rgb([cond_set[0]/(2*np.pi), 1, 1])
        linestyle = '-'

        # plot mean ± SEM
        ax.fill_between(t_plot,
                        data_plot_mean-data_plot_std,
                        data_plot_mean+data_plot_std,
                        alpha=0.2, color=color, linewidth=0.0)
        ax.plot(t_plot, data_plot_mean, color=color, linewidth=2)

        # movement period plotting window: -200 to 1000 ms,
        # shifted right by t_offset so both epochs share one axis
        t_offset = 1300
        T_STA = int(GO_CUE+(-200/TS))
        T_END = int(GO_CUE+(1000/TS))
        t_plot = t_data[T_STA:T_END] + t_offset

        # compute mean and SEM firing rate across trials
        data_plot = neural_data_move_all[mask][:,T_STA:T_END,ch]
        data_plot_mean = data_plot.mean(axis=0)
        data_plot_std = data_plot.std(axis=0, ddof=1) / np.sqrt(data_plot.shape[0])

        # color by movement direction
        color = colors.hsv_to_rgb([cond_set[0]/(2*np.pi), 1, 1])
        linestyle = '-'

        # plot mean ± SEM
        ax.fill_between(t_plot,
                        data_plot_mean-data_plot_std,
                        data_plot_mean+data_plot_std,
                        alpha=0.2, color=color, linewidth=0.0)
        ax.plot(t_plot, data_plot_mean, color=color, linewidth=2)

    # misc. subplot settings
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])

    # offset for scale bar
    scale_offset = [-20, -2]
    ax.set_xlim(-300+scale_offset[0], 2400)
    ylims = ax.get_ylim()

    # draw target cue (yellow) and go cue (green)
    ax.plot([0,0],[ylims[0], ylims[1]], color='y', linestyle='-', linewidth=2)
    ax.plot([t_offset,t_offset],[ylims[0], ylims[1]], color='g', linestyle='-', linewidth=2)

    # 2. plot no-move (catch) trials (T5 had no catch trials)
    if participant != 'T5':

        ax = axs[1]

        # loop over conditions
        for cond_set in itertools.product(*CONDITIONS):

            # select catch trials of this direction with a long enough delay
            masks = [cond_set[i] == CONDITION_ARRAYS[i] for i in range(len(CONDITIONS))]
            mask = np.all(masks, axis=0)
            mask = mask & (delay_data_all >= 0.8) & (no_go_bool_all == 1)

            # delay period plotting window: -200 ms to the end of the shortest delay
            T_STA = int(GO_CUE+(-200/TS))
            T_END = int(GO_CUE+(delay_min*1000/TS))
            t_plot = t_data[T_STA:T_END]

            # compute mean and SEM firing rate across trials
            data_plot = neural_data_all[mask][:,T_STA:T_END,ch]
            data_plot_mean = data_plot.mean(axis=0)
            data_plot_std = data_plot.std(axis=0, ddof=1) / np.sqrt(data_plot.shape[0])

            # color by movement direction
            color = colors.hsv_to_rgb([cond_set[0]/(2*np.pi), 1, 1])
            linestyle = (0, (5, 1))

            # plot mean ± SEM
            ax.fill_between(t_plot,
                            data_plot_mean-data_plot_std,
                            data_plot_mean+data_plot_std,
                            alpha=0.2, color=color, linewidth=0.0)
            ax.plot(t_plot, data_plot_mean, color=color, linewidth=2, linestyle=linestyle)

            # (no-move) movement period plotting window, shifted right by t_offset
            t_offset = 1300
            T_STA = int(GO_CUE+(-200/TS))
            T_END = int(GO_CUE+(1000/TS))
            t_plot = t_data[T_STA:T_END] + t_offset

            # compute mean and SEM firing rate across trials
            data_plot = neural_data_move_all[mask][:,T_STA:T_END,ch]
            data_plot_mean = data_plot.mean(axis=0)
            data_plot_std = data_plot.std(axis=0, ddof=1) / np.sqrt(data_plot.shape[0])

            # color by movement direction
            color = colors.hsv_to_rgb([cond_set[0]/(2*np.pi), 1, 1])
            linestyle = (0, (5, 1))

            # plot mean ± SEM
            ax.fill_between(t_plot,
                            data_plot_mean-data_plot_std,
                            data_plot_mean+data_plot_std,
                            alpha=0.1, color=color, linewidth=0.0)
            ax.plot(t_plot, data_plot_mean, color=color, linewidth=2, linestyle=linestyle)

    # misc. subplot settings
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])

    # scale bar offset
    scale_offset = [-20, -2]
    ax.set_xlim(-300+scale_offset[0], 2400)
    ylims = ax.get_ylim()
    ax.set_ylim(ylims[0]+scale_offset[1], ylims[1])

    # draw target cue (yellow) and go cue (green)
    ax.plot([0,0],[ylims[0], ylims[1]], color='y', linestyle='-', linewidth=2)
    ax.plot([t_offset,t_offset],[ylims[0], ylims[1]], color='g', linestyle='-', linewidth=2)

    # draw firing rate (vertical) and time (horizontal) scale bars
    ax.plot([-300+scale_offset[0]/3,-300+scale_offset[0]/3],[ylims[0]+scale_offset[1]/2, ylims[0]+scale_offset[1]/2+5],
            color='k', linestyle='-', linewidth=5)
    ax.plot([-300+scale_offset[0]/2,-300+200],[ylims[0]+scale_offset[1]/2, ylims[0]+scale_offset[1]/2],
            color='k', linestyle='-', linewidth=5)

    # save pdf
    savepath = os.path.join(save_plot_filepath, f'fig1_{participant}_psth_ch{keep_chans_all[ch]}.pdf')
    plt.savefig(savepath, format='pdf')

    # report the plotted channel and the trial counts behind each subplot
    print(f'ich: {ch}, actual channel: {keep_chans_all[ch]}')
    print(f'move trials: {((no_go_bool_all == 0)).sum()}')
    print(f'no move trials: {((no_go_bool_all == 1)).sum()}')

    plt.show()


# %%
## ANOVA for single-channel tuning

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

# container for the per-channel direction p-value
anova_results_direction = np.zeros((N_CHANNELS))

# loop over channels, testing whether delay activity depends on direction
for ich in range(N_CHANNELS):

    # delay window: 400 ms after the target cue to the end of the shortest delay
    T_STA = int(GO_CUE+(400/TS))
    T_END = int(GO_CUE+(delay_min*1000/TS))

    # one row per trial: direction and the window-averaged firing rates for this channel
    df = pd.DataFrame({
        'direction': angle_data_all,
        'neural_data_ch': neural_data_all[:,
                                          T_STA:T_END,
                                          ich].mean(axis=1),
    })

    # run ANOVA
    model = ols("""neural_data_ch ~ C(direction)""", data = df).fit()
    anova = sm.stats.anova_lm(model, typ = 2)

    # get FDR-corrected p-values
    pvals = anova['PR(>F)'].values[:-1]
    corrected_pvals = multipletests(pvals, method='fdr_bh')[1]

    # save the direction p-value for this channel
    p = corrected_pvals[0]
    anova_results_direction[ich] = p

# save ANOVA results
savepath = os.path.join(save_plot_filepath, f'fig1_{participant}_anova.pkl')
with open(savepath, 'wb') as f:
    pickle.dump({'anova_results_direction': anova_results_direction,
                    }, f)


# %%
## Plot single-channel tuning results

# significance threshold
sig = 0.01

plt.figure(figsize=(6, 5.5))

# loop over participants, using the cached ANOVA results of each
for i, part in enumerate(['T11', 'T16', 'T5']):

    savepath = os.path.join(save_plot_filepath, f'fig1_{part}_anova.pkl')

    # only plot participants whose ANOVA .pkl exists
    if os.path.exists(savepath):

        # load p-values
        with open(savepath, 'rb') as f:
            anova_results = pickle.load(f)

        # compute % of significant channels
        anova_sig = anova_results['anova_results_direction'] < sig
        sig_percent = anova_sig.sum(axis=0) / anova_sig.shape[0] * 100

        print(f'\nParticipant {part} - Direction ANOVA: ')
        print(np.where(anova_sig)[0])

        # plot % of tuned channels, annotated with the channel count
        pos = 2*i
        plt.bar(pos, sig_percent, color=(0.45, 0.45, 0.9), width=0.5)
        plt.text(x=pos, y=sig_percent+2, s=f'{anova_sig.sum(axis=0)}/{anova_sig.shape[0]}\nchannels',  fontsize=16,
                    color='k', ha='center', va='bottom')

# misc. plot settings
plt.xticks([0, 2, 4], ['T11', 'T16', 'T5'],
           rotation=0, fontsize=18)
plt.yticks(np.arange(0, 101, 20), fontsize=14)
plt.ylabel('% of channels tuned\nto movement direction', fontsize=16)
plt.gca().axes.spines[['top', 'right']].set_visible(False)
plt.xlim(-1.25, 5.25)
plt.ylim(0, 100)

# save pdf
savepath = os.path.join(save_plot_filepath, 'fig1_tuned_chans.pdf')
plt.savefig(savepath, format='pdf')

plt.show()


# %%
## Normalize neural data

# soft-normalization constants for the spike and SBP features
SOFT_NORM_SPIKES = 1e-9
SOFT_NORM_SBP = 1e-9

# soft-normalize per block and concatenate the spike and SBP features
neural_data_norm_all, neural_data_move_norm_all = normalize_and_concat(
    neural_data_all, sbp_data_all, neural_data_move_all, sbp_data_move_all,
    block_id_data_all, GO_CUE, TS,
    USE_SBP=True, SOFT_NORM_SPIKES=SOFT_NORM_SPIKES, SOFT_NORM_SBP=SOFT_NORM_SBP,
)


# %%
## Crossnobis distance matrix (delay period)

# build condition-averaged and per-trial arrays
X, trialsX, trialNum = build_marg_arrays(neural_data_norm_all,
                                         CONDITIONS, CONDITION_ARRAYS,)

# delay window
T_STA_corr = int(400/TS)
T_END_corr = int(delay_min*1000/TS)

# compute cross-validated Mahalanobis (crossnobis) distances between directions
crossnobis_delay = compute_crossnobis_matrix(trialsX, trialNum, GO_CUE, T_STA_corr, T_END_corr)

# color scale (shared with the movement-period matrix below)
vmax = 30
cbar_norm = colors.PowerNorm(gamma=0.55, vmin=0, vmax=vmax)

# mean over off-diagonal entries only; diagonal (self-distance) is trivially zero
off_diag_mask = ~np.eye(crossnobis_delay.shape[0], dtype=bool)
mean_crossnobis_delay = np.mean(crossnobis_delay[off_diag_mask])
print(f'Mean crossnobis (off-diagonal): {mean_crossnobis_delay:.3f}')

# plot the distance matrix as an annotated heatmap
ax = sns.heatmap(crossnobis_delay,
                 vmin=0, vmax=vmax,
                 norm=cbar_norm,
                 annot=True, annot_kws={"size": 12}, fmt='.1f',
                 cbar=True)

# misc. plot settings
ax.set_xlabel('Direction', fontsize=14)
ax.set_ylabel('Direction', fontsize=14)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(axis='both', which='both', length=0)
plt.gca().invert_yaxis()
plt.title(f'{participant}\nMean crossnobis distance: {mean_crossnobis_delay:.2f}', fontsize=16)

# save pdf
savepath = os.path.join(save_plot_filepath, f'fig1_{participant}_crossnobis_delay.pdf')
plt.savefig(savepath, format='pdf')

plt.show()


# %%
## Crossnobis distance matrix (movement period)

# build condition-averaged and per-trial arrays, go trials only
X_move, trialsX_move, trialNum_move = build_marg_arrays(neural_data_move_norm_all,
                                                        CONDITIONS, CONDITION_ARRAYS,
                                                        trial_mask=(no_go_bool_all == 0))

# movement window
if participant == 'T5':
    T_STA_move_corr = int(200/TS)
    T_END_move_corr = int(600/TS)
else:
    T_STA_move_corr = int(400/TS)
    T_END_move_corr = int(1000/TS)

# compute cross-validated Mahalanobis (crossnobis) distances between directions
crossnobis_move = compute_crossnobis_matrix(trialsX_move, trialNum_move, GO_CUE_MOVE, T_STA_move_corr, T_END_move_corr)

# mean over off-diagonal entries only; diagonal (self-distance) is trivially zero
off_diag_mask = ~np.eye(crossnobis_move.shape[0], dtype=bool)
mean_crossnobis_move = np.mean(crossnobis_move[off_diag_mask])
print(f'Mean crossnobis move (off-diagonal): {mean_crossnobis_move:.3f}')

# plot the distance matrix as an annotated heatmap
ax = sns.heatmap(crossnobis_move,
                 vmin=0, vmax=vmax,
                 norm=cbar_norm,
                 annot=True, annot_kws={"size": 12}, fmt='.1f',
                 cbar=True)

# misc. plot settings
ax.set_xlabel('Direction', fontsize=14)
ax.set_ylabel('Direction', fontsize=14)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(axis='both', which='both', length=0)
plt.gca().invert_yaxis()
plt.title(f'{participant}\nMean crossnobis distance: {mean_crossnobis_move:.2f}', fontsize=16)

# save pdf
savepath = os.path.join(save_plot_filepath, f'fig1_{participant}_crossnobis_move.pdf')
plt.savefig(savepath, format='pdf')

plt.show()


# %%
## Alignment index — prep vs. move subspace (Elsayed et al. 2016)

# delay/move epoch windows
T_STA_PREP = int(GO_CUE + 400/TS)
T_END_PREP = int(GO_CUE + 800/TS) if participant == 'T5' else int(GO_CUE + delay_min*1000/TS)
if participant == 'T5':
    T_STA_MOVE = int(GO_CUE_MOVE + 200/TS)
    T_END_MOVE = int(GO_CUE_MOVE + 600/TS)
else:
    T_STA_MOVE = int(GO_CUE_MOVE + 400/TS)
    T_END_MOVE = int(GO_CUE_MOVE + 1000/TS)

# trial mask: go trials with sufficient delay and valid condition label
ai_mask = (delay_data_all >= 0.8) & (no_go_bool_all == 0)

# compute alignment indices and VAFs
ai_1to2, ai_2to1, ai_mean, vaf_1to2, vaf_2to1, _, _, _, _ = alignment_index(
    neural_data_norm_all[ai_mask],
    neural_data_move_norm_all[ai_mask],
    angle_data_all[ai_mask],
    angle_data_all[ai_mask],
    t_start=T_STA_PREP,
    t_end=T_END_PREP,
    t_start2=T_STA_MOVE,
    t_end2=T_END_MOVE,
    n_pcs=10,
    subtract_ci=True,
)

# within delay alignment index: self-consistency check and yields D_prep (delay-period PC axes)
ai_1to1, _, _, vaf_1to1, _, D_prep, _, scale_prep, _ = alignment_index(
    neural_data_norm_all[ai_mask],
    neural_data_norm_all[ai_mask],
    angle_data_all[ai_mask],
    angle_data_all[ai_mask],
    t_start=T_STA_PREP,
    t_end=T_END_PREP,
    t_start2=T_STA_PREP,
    t_end2=T_END_PREP,
    n_pcs=10,
    subtract_ci=True,
)

# within movement alignment index: self-consistency check and yields D_move (movement-period PC axes)
ai_2to2, _, _, vaf_2to2, _, D_move, _, scale_move, _ = alignment_index(
    neural_data_move_norm_all[ai_mask],
    neural_data_move_norm_all[ai_mask],
    angle_data_all[ai_mask],
    angle_data_all[ai_mask],
    t_start=T_STA_MOVE,
    t_end=T_END_MOVE,
    t_start2=T_STA_MOVE,
    t_end2=T_END_MOVE,
    n_pcs=10,
    subtract_ci=True,
)

print(f'Alignment index (prep→prep): {ai_1to1:.3f}')
print(f'Alignment index (prep→move): {ai_1to2:.3f}')
print(f'Alignment index (move→prep): {ai_2to1:.3f}')
print(f'Alignment index (move→move): {ai_2to2:.3f}')
print(f'Alignment index (mean):      {ai_mean:.3f}')


# generate alignment index VAF bar plots
_, axes2 = plt.subplots(1, 2, figsize=(10, 4))

# bar width for the VAF plots
bar_width = 0.42
# bar positions and labels
n_pcs = len(vaf_1to2)
x = np.arange(n_pcs)
pc_labels = [str(i + 1) for i in range(n_pcs)]

# 1. plot variance of both epochs in the delay period subspace
axes2[0].bar(x - bar_width/2, vaf_1to1*scale_prep*100, bar_width, color='y', label='Delay period data')
axes2[0].bar(x + bar_width/2, vaf_2to1*scale_move*100, bar_width, color='g', label='Movement period data')

# misc. subplot settings
axes2[0].set_xticks(x)
axes2[0].set_xticklabels(pc_labels, fontsize=15)
axes2[0].set_xlabel('PC', fontsize=17)
axes2[0].set_ylabel('% of total variance', fontsize=17)
axes2[0].set_title('Delay period subspace', fontweight='bold', color='y', fontsize=18, pad=-20)
axes2[0].text(0.5, 0.9, f'Alignment index: {ai_2to1:.2f}',
              transform=axes2[0].transAxes, ha='center', va='bottom', fontsize=16)
axes2[0].tick_params(axis='y', labelsize=15)
axes2[0].legend(frameon=True, loc='center right', bbox_to_anchor=(1.0, 0.67), fontsize=15)
axes2[0].spines[['top', 'right']].set_visible(False)

# 2. plot variance of both epochs in the movement period subspace
axes2[1].bar(x - bar_width/2, vaf_1to2*scale_prep*100, bar_width, color='y', label='Delay period data')
axes2[1].bar(x + bar_width/2, vaf_2to2*scale_move*100, bar_width, color='g', label='Movement period data')

# misc. subplot settings
axes2[1].set_xticks(x)
axes2[1].set_xticklabels(pc_labels, fontsize=15)
axes2[1].set_xlabel('PC', fontsize=17)
axes2[1].set_title('Movement period subspace', fontweight='bold', color='g', fontsize=18, pad=-20)
axes2[1].text(0.5, 0.9, f'Alignment index: {ai_1to2:.2f}',
              transform=axes2[1].transAxes, ha='center', va='bottom', fontsize=16)
axes2[1].tick_params(axis='y', labelsize=15)
axes2[1].legend(frameon=True, loc='center right', bbox_to_anchor=(1.0, 0.67), fontsize=15)
axes2[1].spines[['top', 'right']].set_visible(False)

# keep only the right panel's legend
axes2[0].legend().set_visible(False)

# use one common y limit across both panels
vaf_max = max(np.max(vaf_1to1*scale_prep*100), np.max(vaf_2to1*scale_move*100),
              np.max(vaf_1to2*scale_prep*100), np.max(vaf_2to2*scale_move*100))
for a in axes2:
    a.set_ylim(0, vaf_max * 1.1)
plt.tight_layout()

# save pdf
savepath = os.path.join(save_plot_filepath, f'fig1_{participant}_alignment_index.pdf')
plt.savefig(savepath, format='pdf')

plt.show()


# %%
## 2D projections — delay & move epochs onto delay PCs vs. move PCs

# use same trial mask as alignment_index for consistency
ai_trial_mask = (delay_data_all >= 0.8) & (no_go_bool_all == 0)

# build delay-period condition averages and subtract the condition-independent signal
Xdel_proj, _, _ = build_marg_arrays(
    neural_data_norm_all, CONDITIONS, CONDITION_ARRAYS)
Xdel_cis_proj  = Xdel_proj.mean(axis=1)
Xdel_nocis     = Xdel_proj - Xdel_cis_proj[:, None]

# same for the movement period
Xmov_proj, _, _ = build_marg_arrays(
    neural_data_move_norm_all, CONDITIONS, CONDITION_ARRAYS,
    trial_mask=ai_trial_mask)
Xmov_cis_proj  = Xmov_proj.mean(axis=1)
Xmov_nocis_proj = Xmov_proj - Xmov_cis_proj[:, None]

# two epochs: delay (0–800/1000 ms depending on participant), movement (0–1200 ms post-go)
t_end_del_samp = int(800/TS) if participant == 'T5' else int(delay_min*1000/TS)
t_end_del_ms   = 800         if participant == 'T5' else int(delay_min*1000)
epoch_slices = [
    (Xdel_nocis,      GO_CUE       + int(0 / TS), GO_CUE      + t_end_del_samp),
    (Xmov_nocis_proj, GO_CUE_MOVE  + int(0 / TS), GO_CUE_MOVE + int(1200 / TS)),
]
group_labels = [f'Delay period\n(0–{t_end_del_ms} ms)', 'Movement period\n(0–1200 ms)']

# define the two PC spaces: delay PCs and movement PCs
pc_spaces    = [D_prep, D_move]
pc_labels    = ['Delay period subspace', 'Move period subspace']
pc_colors    = ['y', 'g']

# trajectory start marker per epoch (● delay, ▲ movement)
start_marker = ['o', '^']

# project all epochs onto both PC spaces
Z_proj = [
    [Xdata[:, :, sta:end].transpose(1, 2, 0) @ D for D in pc_spaces]
    for Xdata, sta, end in epoch_slices
]

# generate plot
fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True,
                        gridspec_kw={'hspace': 0.55, 'wspace': -0.1})
fig.subplots_adjust(top=0.84)

# single shared limit across all panels and PC spaces
plot_lim = 1.10 * max(np.abs(Z[:, :, :2]).max() for epoch in Z_proj for Z in epoch)
ticks_pos = max(1, round(plot_lim * 0.75 / 1.1))  # extent of the partial spines and ticks

# loop over panels, one per (PC space, epoch) pair
for row in range(2):  # PC space
    for col in range(2):  # epoch

        ax = axs[row, col]

        # plot one trajectory per direction, with a start marker and an arrowhead at the end
        for ia, angle in enumerate(angle_data_unique):
            traj = Z_proj[col][row][ia]
            c = colors.hsv_to_rgb([angle / (2 * np.pi), 1, 1])
            ax.plot(traj[:, 0], traj[:, 1], color=c, lw=1.5)
            ax.scatter(traj[0, 0], traj[0, 1], color=c,
                       marker=start_marker[col], s=25, zorder=5)
            ax.arrow(traj[-2, 0], traj[-2, 1],
                     traj[-1, 0] - traj[-2, 0], traj[-1, 1] - traj[-2, 1],
                     head_width=0.1*plot_lim, head_length=0.1*plot_lim, overhang=0.3,
                     fc=c, ec=c, lw=0, length_includes_head=True, clip_on=True)

        # misc. subplot settings
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_bounds(-ticks_pos, ticks_pos)
        ax.spines[['left', 'bottom']].set_position(('outward', 4))
        ax.set_xticks([-ticks_pos, 0, ticks_pos])
        ax.set_yticks([-ticks_pos, 0, ticks_pos])
        ax.set_aspect('equal')
        ax.tick_params(labelbottom=True, labelleft=True, labelsize=10)
        ax.set_xlabel('PC1 (a.u.)', fontsize=11)
        ax.set_ylabel('PC2 (a.u.)', fontsize=11, labelpad=-5)
        ax.text(0.5, 1, group_labels[col], transform=ax.transAxes,
                ha='center', va='top', fontsize=12)

# apply single global limit to all panels
axs[0, 0].set_xlim(-plot_lim, plot_lim)
axs[0, 0].set_ylim(-plot_lim, plot_lim)

# draw figure
fig.canvas.draw()

# annotate each panel with the VAF in the first 2 PCs
vaf_grid = [
    [(vaf_1to1[0] + vaf_1to1[1]) * scale_prep * 100,
     (vaf_1to2[0] + vaf_1to2[1]) * scale_prep * 100],
    [(vaf_2to1[0] + vaf_2to1[1]) * scale_move * 100,
     (vaf_2to2[0] + vaf_2to2[1]) * scale_move * 100],
]
for row in range(2):
    for col in range(2):
        axs[row, col].text(0.02, 0.015, f'{vaf_grid[col][row]:.1f}% of\ntotal var.',
                           transform=axs[row, col].transAxes,
                           ha='left', va='bottom', fontsize=11, color='black')

# subspace labels centered across both columns, one per row
for row in range(2):
    bbox0 = axs[row, 0].get_position()
    bbox1 = axs[row, 1].get_position()
    fig.text((bbox0.x0 + bbox1.x1) / 2, bbox0.y1 + 0.02,
             pc_labels[row], ha='center', va='bottom', fontsize=14,
             fontweight='bold', color=pc_colors[row])

# save pdf
savepath = os.path.join(save_plot_filepath, f'fig1_{participant}_pca_proj2d.pdf')
plt.savefig(savepath, format='pdf')

plt.show()


# %%
## Build dPCA arrays

# build condition-averaged and per-trial arrays
X, trialsX, trialNum = build_marg_arrays(neural_data_norm_all,
                                        CONDITIONS, CONDITION_ARRAYS,)

# delay window: target cue to the end of the shortest delay
T_STA = int(GO_CUE+(0/TS))
if participant == 'T5':
    T_END = int(GO_CUE+(800/TS))
else:
    T_END = int(GO_CUE+(delay_min*1000/TS))

# per-trial and trial-averaged firing rates over the delay window, plus the
# marginalization names expected by the MATLAB dPCA code
firingRates = trialsX[:,:,T_STA:T_END,:]
firingRatesAverage = X[:,:,T_STA:T_END]
margNames = ['Direction dependent', 'Condition independent',]


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

# run dpca script for 1 variable
output = eng.dpca_analysis_1d(firingRates_matlab,
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

# mean-center trial-averaged firing rates
Xcen = firingRatesAverage - np.nanmean(firingRatesAverage, axis=(1,2), keepdims=True)

# flatten mean-centered trial-averaged firing rates
Xfull_flat = Xcen.reshape(Xcen.shape[0], -1)

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
## Plot 3D dPCA projections

# build condition-averaged arrays, one condition per direction
X, trialsX, trialNum = build_marg_arrays(neural_data_norm_all,
                                        CONDITIONS, CONDITION_ARRAYS,)

# delay window: target cue to the end of the shortest delay
T_STA = int(GO_CUE+(0/TS))
if participant == 'T5':
    T_END = int(GO_CUE+(800/TS))
else:
    T_END = int(GO_CUE+(delay_min*1000/TS))

# mean-center trial-averaged firing rates and compute dPCA projections
Xdpca = X[:,:,T_STA:T_END]
Xcen = Xdpca - Xdpca.mean(axis=(1,2), keepdims=True)
Z = Xcen.T @ dpcaW

# generate 3D plot
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
fig.patch.set_alpha(0.0)

# plot the first 2 direction-dependent components and the first condition-independent one
PCS = [
    np.where(whichMarg == 0+1)[1][0],
    np.where(whichMarg == 0+1)[1][1],
    np.where(whichMarg == 1+1)[1][0],
]

# participant-specific axis limits, scale bar placement, and camera angle
if participant == 'T16':
    limits = [0.75,0.75,0.75]
    center = np.array([-0.3,0.25,-0.3])
    axes_length = [0.8,-0.7,0.7]
    elev = 25
    azim = -40
elif participant == 'T11':
    limits = [2,2,2]
    center = np.array([-0.8,-1,-1.5])
    axes_length = [1.75,1.75,2.5]
    elev = 40
    azim = 50
elif participant == 'T5':
    limits = [3.5,3.5,3.5]
    center = np.array([-1.5,-1,-4])
    axes_length = [2.5,2.5,6]
    elev = 50
    azim = 30

# draw axes scale
ax.plot([center[0],center[0]+axes_length[0]],
        [center[1],center[1]],
        [center[2],center[2]], color=(0.66,0.66,0.66), linestyle='-', linewidth=2.5)
ax.plot([center[0],center[0]],
        [center[1],center[1]+axes_length[1]],
        [center[2],center[2]], color=(0.43,0.43,0.43), linestyle='-', linewidth=2.5)
ax.plot([center[0],center[0]],
        [center[1],center[1]],
        [center[2],center[2]+axes_length[2]], color=(0.20,0.20,0.20), linestyle='-', linewidth=2.5)

# plot one trajectory per condition, marking the start (circle) and end (triangle)
for ic, cond_set in enumerate(itertools.product(*CONDITIONS)):
    color = colors.hsv_to_rgb([cond_set[0]/(2*np.pi), 1, 1])
    linestyle = '-'
    linewidth=3
    ax.plot(Z[:,ic,PCS[0]],
            Z[:,ic,PCS[1]],
            Z[:,ic,PCS[2]],
            color=color, linewidth=linewidth, linestyle=linestyle, alpha=0.8, zorder=10)
    ax.scatter(Z[0,ic,PCS[0]],
                Z[0,ic,PCS[1]],
                Z[0,ic,PCS[2]],
                color=color, s=50, marker='o', alpha=1, zorder=10)
    ax.scatter(Z[-1,ic,PCS[0]],
                Z[-1,ic,PCS[1]],
                Z[-1,ic,PCS[2]],
                color=color, s=100, marker='^', alpha=1, zorder=10)

# misc. plot settings
ax.set_xlim([-limits[0], limits[0]])
ax.set_ylim([-limits[1], limits[1]])
ax.set_zlim([-limits[2], limits[2]])
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)
ax.set_axis_off()
ax.view_init(elev=elev, azim=azim)
ax.set_xlabel(f'PC{PCS[0]+1}')
ax.set_ylabel(f'PC{PCS[1]+1}')
ax.set_zlabel(f'PC{PCS[2]+1}')

# save pdf
savepath = os.path.join(save_plot_filepath, f'fig1_{participant}_dpca3D.pdf')
plt.savefig(savepath, format='pdf')

plt.show()
plt.close(fig)  # release the 3D axes so later heatmaps get a fresh 2D figure (linear run)

# report the variance accounted for by the 3 plotted components
print(f'CD1 VAF: {comp_variance[PCS[0]]}%')
print(f'CD2 VAF: {comp_variance[PCS[1]]}%')
print(f'CI1 VAF: {comp_variance[PCS[2]]}%')


# %%
## Cross-validated decoding sweep (prep / move / no-move epochs)

# epoch windows, relative to the target cue (prep) or the go cue (move)
T_END_prep = int(GO_CUE+800/TS) if participant == 'T5' else int(GO_CUE+delay_min*1000/TS)
T_START_move = int(GO_CUE_MOVE+200/TS) if participant == 'T5' else int(GO_CUE_MOVE+400/TS)
T_END_move = int(GO_CUE_MOVE+600/TS) if participant == 'T5' else int(GO_CUE_MOVE+1000/TS)

# each dict specifies the time window, trial subset, and output filenames for one epoch
epochs = [
    dict(
        label='prep',
        T_STA=int(GO_CUE+400/TS),
        T_END=T_END_prep,
        trial_mask=(delay_data_all >= 0.8),
        X_data=neural_data_norm_all,
        pdf_suf='_decoding_svm',
        pkl_suf='_decoding',
    ),
    dict(
        label='move',
        T_STA=T_START_move,
        T_END=T_END_move,
        trial_mask=(delay_data_all >= 0.8) & (no_go_bool_all == 0),
        X_data=neural_data_move_norm_all,
        pdf_suf='_decoding_svm_move',
        pkl_suf='_decoding_move',
    ),
    dict(
        label='nomove',
        T_STA=T_START_move,
        T_END=T_END_move,
        trial_mask=(delay_data_all >= 0.8) & (no_go_bool_all == 1),
        X_data=neural_data_move_norm_all,
        pdf_suf='_decoding_svm_nomove',
        pkl_suf='_decoding_nomove',
    ),
]

# loop over epochs, running or loading the decoding sweep and plotting its confusion matrix
for ep in epochs:

    # get epoch trial mask
    trial_mask = ep['trial_mask']

    # only decode epochs that have trials (e.g. T5 has no catch trials)
    if trial_mask.sum() == 0:
        print(f'No valid trials. Skipping {ep["label"]} decoding.')
        continue

    # pkl path for caching the decoding sweep results
    pkl_path = os.path.join(save_plot_filepath, f'fig1_{participant}{ep["pkl_suf"]}.pkl')

    # run the sweep and cache it, or load the cached results
    if RUN_DECODING_SWEEP:
        np.random.seed(42)
        sweep_results = run_decoding_sweep(
            ep['X_data'], angle_data_all, CONDITIONS, CONDITION_ARRAYS,
            trial_mask, ep['T_STA'], ep['T_END'],
        )
        with open(pkl_path, 'wb') as f:
            pickle.dump(sweep_results, f)
    else:
        # if results are not cached, warn the user and skip this epoch  
        if not os.path.exists(pkl_path):
            print(f'WARNING: {pkl_path} not found. Set RUN_DECODING_SWEEP = True to (re)generate it.')
            continue
        with open(pkl_path, 'rb') as f:
            sweep_results = pickle.load(f)

    # get true and predicted labels, pooled and per CV split, plus the chance accuracies
    true_all, pred_all = sweep_results['true_all'], sweep_results['pred_all']
    true_all_split, pred_all_split = sweep_results['true_all_split'], sweep_results['pred_all_split']
    acc_chance_all = sweep_results['acc_chance_all']
    cv_splits = len(acc_chance_all)

    # compute fold-mean and pooled accuracy
    acc_folds = 100*np.mean([accuracy_score(true_all_split[i], pred_all_split[i]) for i in range(cv_splits)])
    acc_overall = 100*accuracy_score(true_all, pred_all)
    # compute fold-mean and pooled chance accuracy
    chance_folds = 100*np.mean(acc_chance_all)
    chance_overall = 100*np.bincount(true_all.ravel()).max()/np.bincount(true_all.ravel()).sum()

    # report the decoding results
    print(f'Fold accuracies: {acc_folds}')
    print(f'Overall accuracy: {acc_overall}')
    print('-----')
    print(f'Fold chance: {chance_folds}')
    print(f'Overall chance: {chance_overall}')

    # plot the SVM confusion matrix
    ax = sns.heatmap(sklearn.metrics.confusion_matrix(true_all, pred_all, normalize='true'),
                     cmap='viridis',
                     annot=True,
                     annot_kws={"size": 14},
                     fmt='.1g',
                     vmin=0,
                     vmax=1, cbar=True)

    # misc. plot settings
    ax.set_xlabel('Predicted direction', fontsize=14)
    ax.set_ylabel('True direction', fontsize=14)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    plt.gca().invert_yaxis()
    plt.title(f'{participant}\nDecoding accuracy: {acc_folds:2.1f}%', fontsize=16)

    # save pdf
    savepath = os.path.join(save_plot_filepath, f'fig1_{participant}{ep["pdf_suf"]}.pdf')
    plt.savefig(savepath, format='pdf')

    plt.show()


# %%
## Cross-epoch decoding — train on one epoch, evaluate once on each other epoch

np.random.seed(42)

# same three epochs as the sweep above, without the .pdf suffixes
epochs_ce = [
    dict(
        label='prep',
        T_STA=int(GO_CUE + 400/TS),
        T_END=int(GO_CUE + 800/TS) if participant == 'T5' else int(GO_CUE + delay_min*1000/TS),
        trial_mask=(delay_data_all >= 0.8),
        X_data=neural_data_norm_all,
        pkl_suf='_decoding',
    ),
    dict(
        label='move',
        T_STA=T_START_move,
        T_END=T_END_move,
        trial_mask=(delay_data_all >= 0.8) & (no_go_bool_all == 0),
        X_data=neural_data_move_norm_all,
        pkl_suf='_decoding_move',
    ),
    dict(
        label='nomove',
        T_STA=T_START_move,
        T_END=T_END_move,
        trial_mask=(delay_data_all >= 0.8) & (no_go_bool_all == 1),
        X_data=neural_data_move_norm_all,
        pkl_suf='_decoding_nomove',
    ),
]

# build condition labels
y_all_ce = np.expand_dims(np.unique(angle_data_all, return_inverse=True, axis=0)[1], axis=1).flatten()

# loop over epochs, labels, epoch-mean features, and whether any trials exist
for ep in epochs_ce:
    mask = ep['trial_mask']
    ep['y'] = y_all_ce[mask]
    ep['X'] = ep['X_data'][mask, ep['T_STA']:ep['T_END'], :].mean(axis=1)
    ep['has'] = mask.sum() > 0

# 3×3 accuracy matrix: acc_cross_epoch[i_train, i_test]  (values in %)
if RUN_DECODING_SWEEP:

    # containers for the accuracy and chance matrices
    acc_cross_epoch    = np.full((3, 3), np.nan)
    chance_cross_epoch = np.full((3, 3), np.nan)

    # off-diagonal: train on all of epoch i, test once on all of epoch j
    for i_tr in range(3):
        for i_te in range(3):

            # diagonal computed separately below
            if i_tr == i_te:
                continue

            # only run if both epochs have trials
            if not epochs_ce[i_tr]['has'] or not epochs_ce[i_te]['has']:
                continue

            # flatten features to (n_trials, n_features)
            X_train_ce = epochs_ce[i_tr]['X'].reshape(epochs_ce[i_tr]['X'].shape[0], -1)
            y_train_ce = epochs_ce[i_tr]['y']
            X_test_ce  = epochs_ce[i_te]['X'].reshape(epochs_ce[i_te]['X'].shape[0], -1)
            y_test_ce  = epochs_ce[i_te]['y']

            # fit SVM with inner 10-fold CV
            pipe = Pipeline([('decode', SVC(kernel='linear', cache_size=1000, class_weight='balanced'))])
            param_grid = {
                'decode__C': np.logspace(-3.5, -2, 7),
            }
            search = GridSearchCV(pipe, param_grid, scoring=make_scorer(accuracy_score), cv=10)
            search.fit(X_train_ce, y_train_ce)

            # evaluate once on all test-epoch data (fully held-out, never used during training)
            acc_cross_epoch[i_tr, i_te]    = 100 * accuracy_score(y_test_ce, search.predict(X_test_ce))
            chance_cross_epoch[i_tr, i_te] = 100 * np.bincount(y_test_ce).max() / len(y_test_ce)

    # container for per-epoch fold accuracies
    acc_test_diag_folds = [None, None, None]

    # diagonal: leave-one-out CV using within-epoch features
    for i_diag, ep in enumerate(epochs_ce):

        # only run for epochs that have trials
        if not ep['has']:
            continue

        # features are already epoch mean
        X_diag = ep['X']
        y_diag = ep['y']
        n_diag = X_diag.shape[0]

        # build condition codes for each trial
        trial_mask_diag = ep['trial_mask']
        cond_codes_diag = np.full(n_diag, np.nan)
        for i, cond_set in enumerate(itertools.product(*CONDITIONS)):
            mask = np.all([cond_set[j] == CONDITION_ARRAYS[j] for j in range(len(CONDITIONS))], axis=0)
            cond_codes_diag[mask[trial_mask_diag]] = i

        # shuffle then sort by condition so every K-th index is a balanced fold
        idxs_diag = np.arange(n_diag)
        np.random.shuffle(idxs_diag)
        sorted_idxs_diag = idxs_diag[np.argsort(cond_codes_diag[idxs_diag])]

        # containers for the per-fold test and chance accuracies
        cv_splits_diag = n_diag
        acc_test_diag   = np.zeros(cv_splits_diag)
        acc_chance_diag = np.zeros(cv_splits_diag)

        # run folds in parallel (leave-one-out with inner 10-fold CV)
        with Parallel(n_jobs=32, require='sharedmem') as parallel:
            tasks = [
                delayed(decoder_sweep_parallel)(X_diag, y_diag, sorted_idxs_diag, i, cv_splits_diag)
                for i in range(cv_splits_diag)
            ]
            results_diag = parallel(tasks)

        # collect per-fold test and chance accuracies
        for i, (acc_chance, acc_test, _, _) in enumerate(results_diag):
            acc_chance_diag[i] = acc_chance
            acc_test_diag[i]   = acc_test

        # store the fold-mean accuracy and chance, and keep the folds for error bars
        acc_cross_epoch[i_diag, i_diag]    = 100 * acc_test_diag.mean()
        chance_cross_epoch[i_diag, i_diag] = 100 * acc_chance_diag.mean()
        acc_test_diag_folds[i_diag]        = 100 * acc_test_diag

    # save cross-epoch decoding results
    savepath = os.path.join(save_plot_filepath, f'fig1_{participant}_cross_epoch_decoding.pkl')
    with open(savepath, 'wb') as f:
        pickle.dump({'acc_cross_epoch':      acc_cross_epoch,
                     'chance_cross_epoch':  chance_cross_epoch,
                     'acc_test_diag_folds': acc_test_diag_folds,
                     }, f)


# %%
## Plot cross-epoch decoding accuracy

# bar plot parameters
bar_pitch   = 0.22
bar_width   = bar_pitch * 0.88
group_pitch = 3 * bar_pitch + 0.5
x_min       = -bar_width/2 - 0.1

# group labels/colors and bar colors for all 3 epochs: delay, move, no-move
group_labels = ['Delay', 'Movement', 'No movement']
group_label_colors = ['y', 'g', 'r']  # tick and legend text colors
epoch_bar_colors = [
    (115/255, 115/255, 229/255),
    ( 54/255,  54/255, 179/255),
    ( 13/255,  13/255, 128/255),
]

# container for each participant's ID, results, tested-on epochs, and panel size
panel_data = []

# load each participant's cached data
for participant in ['T11', 'T16', 'T5']:

    pkl_path = os.path.join(save_plot_filepath, f'fig1_{participant}_cross_epoch_decoding.pkl')

    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)

        acc = results['acc_cross_epoch']

        # get indices of tested-on epochs (columns with at least one decoded value)
        test_epochs_idxs = [group for group in range(3) if not np.all(np.isnan(acc[:, group]))]

        # calculate the position of the rightmost bar
        x_right = max((i * group_pitch + t * bar_pitch
                       for i, g in enumerate(test_epochs_idxs) for t in range(3)
                       if not np.isnan(acc[t, g])), default=2 * bar_pitch)

    # leave a full-width empty panel as a placeholder if no pkl found
    else:
        print(f'WARNING: {pkl_path} not found. Set RUN_DECODING_SWEEP = True to (re)generate it.')
        results, test_epochs_idxs, x_right = None, list(range(3)), 2 * group_pitch + 2 * bar_pitch

    # append participant ID, results, tested-on epochs, and panel size
    panel_data.append((participant, results, test_epochs_idxs, x_right + bar_width/2 + 0.1))

# scale each panel's width to the computed panel size
fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True,
                        gridspec_kw={'width_ratios': [xm - x_min for *_, xm in panel_data]})

# draw bars for each participant (per-train/tested-on epoch pair) + chance-level lines
for ax, (participant, results, test_epochs_idxs, x_max) in zip(axs, panel_data):

    # skip and placeholder text if no data was found for this participant
    if results is None:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        continue

    # load accuracy and chance
    acc_mat    = results['acc_cross_epoch']
    chance_mat = results['chance_cross_epoch']

    # compute 95% CI error bars (for diagonal only)
    err_mat = np.full((3, 3), np.nan)
    for i, folds in enumerate(results.get('acc_test_diag_folds', [])):
        if folds is not None:
            folds = np.asarray(folds)
            err_mat[i, i] = (stats.t.ppf(0.975, df=len(folds) - 1)
                             * folds.std(ddof=1) / np.sqrt(len(folds)))

    # loop over tested-on epochs (groups)
    xtick_pos = []
    for i_group, g in enumerate(test_epochs_idxs):

        # compute group position and x-tick positions for the tested-on epoch labels
        group_x      = i_group * group_pitch
        train_epochs_idxs = [t for t in range(3) if not np.isnan(acc_mat[t, g])]
        xtick_pos.append(np.mean([group_x + t * bar_pitch for t in train_epochs_idxs]))

        # loop over trained-on epochs, drawing a bar for each
        for t in train_epochs_idxs:

            # draw the accuracy bar, with error bars if available
            bar_x = group_x + t * bar_pitch
            ax.bar(bar_x, acc_mat[t, g], width=bar_pitch * 0.88,
                   color=epoch_bar_colors[t], alpha=0.85,
                   yerr=None if np.isnan(err_mat[t, g]) else err_mat[t, g],
                   capsize=0, error_kw=dict(ecolor='k', elinewidth=1.5))

            # draw chance level over the bar
            if not np.isnan(chance_mat[t, g]):
                ax.plot([bar_x - bar_width/2, bar_x + bar_width/2], [chance_mat[t, g]] * 2,
                        color='k', linestyle='--', linewidth=1.0)

    # misc. subplot settings
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels([group_labels[g] for g in test_epochs_idxs], fontsize=11)
    for tick, g in zip(ax.get_xticklabels(), test_epochs_idxs):
        tick.set_color(group_label_colors[g])
        tick.set_fontweight('bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title(participant, fontsize=14)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Tested on:', fontsize=13)
    ax.tick_params(axis='y', labelsize=11)
    if ax is axs[0]:
        ax.set_ylabel('Decoding accuracy (%)', fontsize=12)

# legend: trained-on epoch colors, with colored text
leg = fig.legend(handles=[Patch(facecolor=c, alpha=0.85, label=lbl)
                          for c, lbl in zip(epoch_bar_colors, group_labels)],
                 title='Trained on:', fontsize=11, title_fontsize=13,
                 loc='center left', bbox_to_anchor=(0.01, 0.5), frameon=True)
for text, color in zip(leg.get_texts(), group_label_colors):
    text.set_color(color)
    text.set_fontweight('bold')
fig.get_layout_engine().set(rect=(0.13, 0, 1, 1))

# save pdf
savepath = os.path.join(save_plot_filepath, 'fig1_cross_epoch_decoding_bars.pdf')
plt.savefig(savepath, format='pdf', bbox_inches='tight')

plt.show()


# %%
## Decoding generalization across time

np.random.seed(42)

# sweep parameters: window length, step size, C values, and number of CV splits
WINDOW_LEN = 100
STEP       = 40
C_RANGE    = np.logspace(-3.5, -1, 11)
CV_SPLITS  = 10

# select go trials with a long enough delay, and build their direction labels
trial_mask = (delay_data_all >= 0.8) & (no_go_bool_all == 0)
y_tgen = np.expand_dims(np.unique(angle_data_all, return_inverse=True, axis=0)[1], axis=1).flatten()
y_tgen = y_tgen[trial_mask]

# shuffle labels and then sort by condition so every K-th index is a balanced fold
cond_codes = np.full(y_tgen.shape[0], np.nan)
for i, cond_set in enumerate(itertools.product(*CONDITIONS)):
    mask = np.all([cond_set[j] == CONDITION_ARRAYS[j] for j in range(len(CONDITIONS))], axis=0)
    cond_codes[mask[trial_mask]] = i
idxs = np.arange(y_tgen.shape[0])
np.random.shuffle(idxs)
sorted_idxs = idxs[np.argsort(cond_codes[idxs])]

# sampled time points and the width of the averaging window, in bins
t_ms_sweep_end = 800 if participant == 'T5' else 1000
t_ms     = np.arange(-200, t_ms_sweep_end + STEP, STEP)
t_ms_delay, t_ms_move = t_ms, t_ms
win_samp = int(WINDOW_LEN / TS)

# extract window-averaged delay-period features at each time point
T_end_delay = GO_CUE      + (t_ms / TS).astype(int)
X_delay     = extract_windowed_features(neural_data_norm_all[trial_mask], T_end_delay, win_samp)

# same for the movement period
T_end_move  = GO_CUE_MOVE + (t_ms / TS).astype(int)
X_move      = extract_windowed_features(neural_data_move_norm_all[trial_mask], T_end_move,  win_samp)

# flag to indicate whether the cross-temporal decoding results are available for plotting
tgen_ok_to_plot = True

# run the cross-temporal decoding and cache it, or load the cached results
savepath_pkl = os.path.join(save_plot_filepath, f'fig1_{participant}_tgen_4block.pkl')
if RUN_DECODING_SWEEP:

    # run the cross-temporal decoding
    acc_dd, acc_dm, acc_md, acc_mm, chance_empirical = cross_temporal_decoding(
        X_delay, X_move, y_tgen, sorted_idxs, cv_splits=CV_SPLITS, c_range=C_RANGE,
    )
    # save the results to a pickle file
    with open(savepath_pkl, 'wb') as f:
        pickle.dump({'acc_dd': acc_dd, 'acc_mm': acc_mm,
                     'acc_dm': acc_dm, 'acc_md': acc_md,
                     'chance_empirical': chance_empirical}, f)
else:
    # if results are not cached, warn the user and skip plotting
    if not os.path.exists(savepath_pkl):
        print(f'WARNING: {savepath_pkl} not found. Set RUN_DECODING_SWEEP = True to (re)generate it.')
        tgen_ok_to_plot = False
    else:
        # load the cached results from the pickle file
        with open(savepath_pkl, 'rb') as f:
            pkl = pickle.load(f)
        acc_dd = pkl['acc_dd']
        acc_dm = pkl['acc_dm']
        acc_md = pkl['acc_md']
        acc_mm = pkl['acc_mm']
        chance_empirical = pkl['chance_empirical']


# %%
## Plot 4-block cross-temporal decoding

if tgen_ok_to_plot:

    # compute the mean accuracy across CV folds for each train/test time point pair
    acc_dd_mean = acc_dd.mean(axis=2)
    acc_dm_mean = acc_dm.mean(axis=2)
    acc_md_mean = acc_md.mean(axis=2)
    acc_mm_mean = acc_mm.mean(axis=2)

    # number of time points in each epoch
    n_tgt    = len(t_ms_delay)
    n_mov    = len(t_ms_move)

    # imshow extents: [x_left, x_right, y_bottom, y_top]; x = train axis, y = test axis
    ext_dd = [t_ms_delay[0]-STEP/2, t_ms_delay[-1]+STEP/2, t_ms_delay[0]-STEP/2, t_ms_delay[-1]+STEP/2]
    ext_dm = [t_ms_delay[0]-STEP/2, t_ms_delay[-1]+STEP/2, t_ms_move[0]-STEP/2,  t_ms_move[-1]+STEP/2]
    ext_md = [t_ms_move[0]-STEP/2,  t_ms_move[-1]+STEP/2,  t_ms_delay[0]-STEP/2, t_ms_delay[-1]+STEP/2]
    ext_mm = [t_ms_move[0]-STEP/2,  t_ms_move[-1]+STEP/2,  t_ms_move[0]-STEP/2,  t_ms_move[-1]+STEP/2]

    # generate 2x2 grid of subplots
    fig = plt.figure(figsize=(5.5, 4.5), facecolor='w')
    gs  = fig.add_gridspec(2, 2,
                           width_ratios=[n_tgt, n_mov],
                           height_ratios=[n_mov, n_tgt],
                           hspace=100/1200, wspace=100/1200)

    # ticks at cue onset, mid-window, and window end
    CUE_TICKS = [0, t_ms_sweep_end // 2, t_ms_sweep_end]

    # 1. plot delay → move ([0, 0]: train=delay, test=move)
    ax_dm = fig.add_subplot(gs[0, 0])
    ax_dm.imshow(acc_dm_mean.T, origin='lower', aspect='auto', extent=ext_dm,
                        cmap='viridis', vmin=chance_empirical, vmax=1.0)
    ax_dm.axhline(0, color='g', lw=2.5, zorder=2)
    ax_dm.axvline(0, color='y', lw=2.5, zorder=3)
    ax_dm.set_ylabel('Movement period time (ms)', fontsize=9)
    ax_dm.set_xticks([])
    ax_dm.set_yticks(CUE_TICKS)
    ax_dm.tick_params(axis='y', labelrotation=90, labelsize=8)

    # 2. plot move × move ([0, 1]: train=move, test=move)
    ax_mm = fig.add_subplot(gs[0, 1])
    ax_mm.imshow(acc_mm_mean.T, origin='lower', aspect='auto', extent=ext_mm,
                        cmap='viridis', vmin=chance_empirical, vmax=1.0)
    ax_mm.axhline(0, color='g', lw=2.5, zorder=2)
    ax_mm.axvline(0, color='g', lw=2.5, zorder=3)
    ax_mm.set_xticks([])
    ax_mm.set_yticks([])

    # 3. plot delay × delay ([1, 0]: train=delay, test=delay)
    ax_dd = fig.add_subplot(gs[1, 0])
    im_dd = ax_dd.imshow(acc_dd_mean.T, origin='lower', aspect='auto', extent=ext_dd,
                        cmap='viridis', vmin=chance_empirical, vmax=1.0)
    ax_dd.axhline(0, color='y', lw=2.5, zorder=2)
    ax_dd.axvline(0, color='y', lw=2.5, zorder=3)
    ax_dd.set_xlabel('Delay period time (ms)', fontsize=9)
    ax_dd.set_ylabel('Delay period time (ms)', fontsize=9)
    ax_dd.set_xticks(CUE_TICKS)
    ax_dd.set_yticks(CUE_TICKS)
    ax_dd.tick_params(axis='y', labelrotation=90, labelsize=8)
    ax_dd.tick_params(axis='x', labelsize=8)

    # 4. plot move → delay ([1, 1]: train=move, test=delay)
    ax_md = fig.add_subplot(gs[1, 1])
    ax_md.imshow(acc_md_mean.T, origin='lower', aspect='auto', extent=ext_md,
                        cmap='viridis', vmin=chance_empirical, vmax=1.0)
    ax_md.axhline(0, color='y', lw=2.5, zorder=2)
    ax_md.axvline(0, color='g', lw=2.5, zorder=3)
    ax_md.set_xlabel('Movement period time (ms)', fontsize=9)
    ax_md.set_xticks(CUE_TICKS)
    ax_md.set_yticks([])
    ax_md.tick_params(axis='x', labelsize=8)

    # add one colorbar spanning all 4 panels
    cb = plt.colorbar(im_dd, ax=[ax_dm, ax_mm, ax_dd, ax_md], label='', shrink=0.8, pad=0.04)
    cb.set_ticks(np.arange(0.2, 1.01, 0.2))
    cb.ax.tick_params(labelsize=10)
    cb.outline.set_visible(False)

    # other misc. subplot settings
    for ax in [ax_dm, ax_mm, ax_dd, ax_md]:
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(f'{participant}\nDecoding accuracy across time', fontsize=11, x=0.438)
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    # save pdf
    savepath = os.path.join(save_plot_filepath, f'fig1_{participant}_tgen_4block.pdf')
    plt.savefig(savepath, format='pdf', bbox_inches='tight')

    plt.show()


# %%
## Plot time-resolved decoding (diagonal of cross-temporal decoding)

if tgen_ok_to_plot:

    # get the per-split diagonals (train time == test time) for delay and move
    diag_dd = np.array([np.diag(acc_dd[:, :, s]) for s in range(acc_dd.shape[2])])
    diag_mm = np.array([np.diag(acc_mm[:, :, s]) for s in range(acc_mm.shape[2])])
    n_splits = diag_dd.shape[0]

    # compute mean and 95% CI across CV splits
    diag_dd_mean = diag_dd.mean(axis=0)
    diag_mm_mean = diag_mm.mean(axis=0)
    t_crit = stats.t.ppf(0.975, df=n_splits - 1)
    diag_dd_CI = t_crit * diag_dd.std(ddof=1, axis=0) / np.sqrt(n_splits)
    diag_mm_CI = t_crit * diag_mm.std(ddof=1, axis=0) / np.sqrt(n_splits)

    # generate 1x2 grid of subplots (delay, move) for the time-resolved decoding
    fig = plt.figure(figsize=(5.5, 2.4), facecolor='w')
    fig.subplots_adjust(top=0.72)
    gs_diag = fig.add_gridspec(1, 2, width_ratios=[n_tgt, n_mov], wspace=100/1200)
    ax_d = fig.add_subplot(gs_diag[0, 0])
    ax_m = fig.add_subplot(gs_diag[0, 1], sharey=ax_d)

    DIAG_COLOR = (0.25, 0.45, 0.80)

    # 1. plot delay panel: mean ± 95% CI, with the target cue marked in yellow
    ax_d.plot(t_ms_delay, diag_dd_mean, color=DIAG_COLOR, linewidth=2.5)
    ax_d.fill_between(t_ms_delay, diag_dd_mean - diag_dd_CI, diag_dd_mean + diag_dd_CI, color=DIAG_COLOR, alpha=0.25)
    ax_d.axvline(0, color='y', lw=2.5, zorder=3)

    # misc. subplot settings
    ax_d.set_xlabel('Delay period time (ms)', fontsize=11)
    ax_d.set_xlim(t_ms_delay[0] - STEP/2, t_ms_delay[-1] + STEP/2)
    ax_d.set_xticks(CUE_TICKS)
    ax_d.tick_params(axis='x', labelsize=10)
    ax_d.tick_params(axis='y', labelsize=10)
    ax_d.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_d.set_ylim(0, 1)
    ax_d.spines[['top', 'right']].set_visible(False)
    ax_d.spines['left'].set_position(('outward', 10))
    ax_d.set_axisbelow(True)
    ax_d.grid(True, color='lightgray', lw=0.6, zorder=0)

    # 2. plot move panel: mean ± 95% CI, with the go cue marked in green
    ax_m.plot(t_ms_move, diag_mm_mean, color=DIAG_COLOR, linewidth=2.5)
    ax_m.fill_between(t_ms_move, diag_mm_mean - diag_mm_CI, diag_mm_mean + diag_mm_CI, color=DIAG_COLOR, alpha=0.25)
    ax_m.axvline(0, color='g', lw=2.5, zorder=3)

    # misc. subplot settings
    ax_m.set_xlabel('Movement period time (ms)', fontsize=11)
    ax_m.set_xlim(t_ms_move[0] - STEP/2, t_ms_move[-1] + STEP/2)
    ax_m.set_xticks(CUE_TICKS)
    ax_m.tick_params(axis='x', labelsize=10)
    ax_m.tick_params(axis='y', left=False, labelleft=False)
    ax_m.spines[['top', 'right', 'left']].set_visible(False)
    ax_m.set_axisbelow(True)
    ax_m.grid(True, color='lightgray', lw=0.6, zorder=0)
    fig.suptitle(f'{participant}\nDecoding accuracy through time', fontsize=13, y=0.95)

    # draw chance line spanning both panels
    from matplotlib.transforms import blended_transform_factory
    from matplotlib.lines import Line2D
    fig.canvas.draw()
    fig.add_artist(Line2D([ax_d.get_position().x0, ax_m.get_position().x1],
                          [chance_empirical, chance_empirical],
                           transform=blended_transform_factory(fig.transFigure, ax_d.transData),
                           color='k', lw=1.5, ls='--',
                           zorder=10, clip_on=False))

    # save pdf
    savepath = os.path.join(save_plot_filepath, f'fig1_{participant}_tgen_diagonal.pdf')
    plt.savefig(savepath, format='pdf', bbox_inches='tight')

    plt.show()


# %%
## Plot EMG

# only plot if EMG data is available
if 'emg_data_all' in trial_data_delay.keys():

    # normalize EMG per block, baseline-centered, using both epochs
    emg_data_norm_all, emg_data_move_norm_all = normalize_data(emg_data_all, block_id_data_all,
                                                                    neural_data_move_all=emg_data_move_all,
                                                                    SOFT_NORM=1e-9, center_baseline=True, use_move_data=True,
                                                                    GO_CUE=GO_CUE, TS=TS)

    # generate plot: 2 rows (go/no-go) for each of the 2 EMG channels, 1 column per delay duration
    _, axs = plt.subplots(nrows=2*2, ncols=len(delay_data_unique), figsize=(10,4), sharex=True, sharey=True, facecolor='w')

    t_offset = np.max(delay_data_unique) * 1000
    t_lag = 40  # compensate for lag in recorded EMG data

    # loop over go/no-go trials, delay durations, and EMG channels
    for no_go in [0,1]:
        for id, delay in enumerate(delay_data_unique):
            for ich in range(2):

                ax = axs[2*no_go+ich,id]

                # loop over directions
                for angle in angle_data_unique:

                    # select trials of this direction, delay and go/no-go condition
                    mask = (angle_data_all == angle) & (delay_data_all == delay) & (no_go_bool_all == no_go)
                    color = colors.hsv_to_rgb([angle/(2*np.pi), 0.8, 0.8])

                    # get delay-period EMG and its SEM across trials
                    T_STA = int(GO_CUE-10)
                    T_END = int(GO_CUE+delay*(1000/TS))+1
                    emg = (emg_data_norm_all[mask][:,T_STA:T_END,ich])#.mean(axis=0)
                    sem = emg.std(axis=0, ddof=1) / np.sqrt(emg.shape[0])

                    # right-align the delay periods of different lengths
                    t_offset_2 = 1500 - delay*1000

                    # plot EMG mean ± SEM during the delay period
                    ax.fill_between(t_data[T_STA:T_END] + t_offset_2 - t_lag,
                                    emg.mean(axis=0)-sem,
                                    emg.mean(axis=0)+sem,
                                    alpha=0.2, color=color, linewidth=0.0)
                    ax.plot(t_data[T_STA:T_END] + t_offset_2 - t_lag,
                            emg.mean(axis=0),color=color,alpha=1, linewidth=2)

                    # get movement-period EMG and its SEM across trials
                    T_STA = int(GO_CUE_MOVE-0)
                    T_END = int(GO_CUE_MOVE+1.5*(1000/TS))
                    emg = (emg_data_move_norm_all[mask][:,T_STA:T_END,ich])#.mean(axis=0)
                    sem = emg.std(axis=0, ddof=1) / np.sqrt(emg.shape[0])

                    # plot EMG mean ± SEM during the movement period
                    ax.fill_between(t_data[T_STA:T_END]+t_offset- t_lag,
                                    emg.mean(axis=0)-sem,
                                    emg.mean(axis=0)+sem,
                                    alpha=0.2, color=color, linewidth=0.0)
                    ax.plot(t_data[T_STA:T_END]+t_offset- t_lag,
                            emg.mean(axis=0),color=color,alpha=1, linewidth=2)

                    # draw target cue (yellow) and go cue (green)
                    ax.axvline(t_offset_2, color='y', linestyle='-', linewidth=2)
                    ax.axvline(t_offset, color='g', linestyle='-', linewidth=2)

                    # misc. subplot settings
                    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
                    ax.set_yticks([])
                    ax.set_xticks([])

    plt.tight_layout()

    # save pdf
    savepath = os.path.join(save_plot_filepath, f'fig1_{participant}_emg.pdf')
    plt.savefig(savepath, format='pdf')

    plt.show()


# %%
