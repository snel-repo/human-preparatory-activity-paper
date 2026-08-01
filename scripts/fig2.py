# %%
## Imports

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pickle
import itertools
import matlab.engine
import sklearn
import seaborn as sns
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

sys.path.insert(0, '../lib')
from utils import normalize_data, build_marg_arrays, normalize_and_concat, load_trial_data_h5
from analyses import decoder_sweep_parallel, cv_distance, alignment_index


# %%
## Load data

# select participant
participant = 'T16'
# participant = 'T11'

# whether to (re)compute the SVM decoding sweep and save .pkl, or just use the
# cached .pkl results to plot (warns if a .pkl is missing)
RUN_DECODING_SWEEP = True

# load wrist-task delay-period data
filename_delay = f'./data/fig1_{participant}_delay.h5'
trial_data_delay = load_trial_data_h5(filename_delay)

# load wrist-task movement-period data
filename_move = f'./data/fig1_{participant}_move.h5'
trial_data_move = load_trial_data_h5(filename_move)

# load finger-task delay-period data
filename_delay2 = f'./data/fig2_{participant}_delay.h5'
trial_data_delay2 = load_trial_data_h5(filename_delay2)

# load finger-task movement-period data
filename_move2 = f'./data/fig2_{participant}_move.h5'
trial_data_move2 = load_trial_data_h5(filename_move2)

# define save path for plots
save_plot_filepath = './plots/fig2/'
if not os.path.exists(save_plot_filepath):
    os.makedirs(save_plot_filepath)


# %%
## Get arrays from the trialized data dicts — wrist

# identify trials with NaNs in either the delay or the movement period
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

# get trial info (target and start positions, delay duration, block/trial ids, no-go bool)
target_data_all = trial_data_delay['target_data_all'][~nan_mask]
start_data_all = trial_data_delay['start_data_all'][~nan_mask]
delay_data_all = trial_data_delay['delay_data_all'][~nan_mask]
block_id_data_all = trial_data_delay['block_id_data_all'][~nan_mask]
trial_num_data_all = trial_data_delay['trial_num_data_all'][~nan_mask]
trial_duration_data_all = trial_data_delay['trial_duration_data_all'][~nan_mask]
no_go_bool_all = trial_data_delay['no_go_bool_all'][~nan_mask]

# get timing info (bin width in ms, and the bin index of the cues)
t_data = trial_data_delay['t_data']
TS = t_data[1]-t_data[0]
T_START = int(t_data[0])
GO_CUE = -int(T_START/TS)
T_START_MOVE = int(t_data[0])
GO_CUE_MOVE = -int(T_START_MOVE/TS)

# get the kept (active) channel indices
keep_chans_all1 = trial_data_delay['keep_chans_all']

# convert firing rates to Hz
neural_data_all *= 1000/TS
neural_data_move_all *= 1000/TS


# %%
## Get trial condition variables — wrist

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
## Get arrays from the trialized data dicts — finger

# identify trials with NaNs in either the delay or the movement period
nan_mask1 = np.isnan(trial_data_delay2['spike_data_all']).any(axis=(1,2))
nan_mask2 = np.isnan(trial_data_move2['spike_data_all']).any(axis=(1,2))
nan_mask = nan_mask1 | nan_mask2

# get neural data for the delay and movement periods (firing rates & SBP)
neural_data_all2 = trial_data_delay2['spike_data_all'][~nan_mask]
neural_data_move_all2 = trial_data_move2['spike_data_all'][~nan_mask]
sbp_data_all2 = trial_data_delay2['sbp_data_all'][~nan_mask]
sbp_data_move_all2 = trial_data_move2['sbp_data_all'][~nan_mask]

# get EMG data if available
if 'emg_data_all' in trial_data_delay2:
    emg_data_all2 = trial_data_delay2['emg_data_all'][~nan_mask]
    emg_data_move_all2 = trial_data_move2['emg_data_all'][~nan_mask]

# get trial info (target and start positions, delay duration, block/trial ids, no-go bool)
target_data_all2 = trial_data_delay2['target_data_all'][~nan_mask]
start_data_all2 = trial_data_delay2['start_data_all'][~nan_mask]
delay_data_all2 = trial_data_delay2['delay_data_all'][~nan_mask]
block_id_data_all2 = trial_data_delay2['block_id_data_all'][~nan_mask]
trial_num_data_all2 = trial_data_delay2['trial_num_data_all'][~nan_mask]
trial_duration_data_all2 = trial_data_delay2['trial_duration_data_all'][~nan_mask]
no_go_bool_all2 = trial_data_delay2['no_go_bool_all'][~nan_mask]

# get timing info (bin width in ms, and the bin index of the cues)
t_data2 = trial_data_delay2['t_data']
TS2 = t_data2[1]-t_data2[0]
T_START_MOVE2 = int(t_data2[0])
GO_CUE_MOVE2 = -int(T_START_MOVE2/TS2)

# get the kept (active) channel indices
keep_chans_all2 = trial_data_delay2['keep_chans_all']

# convert firing rates to Hz
neural_data_all2 *= 1000/TS2
neural_data_move_all2 *= 1000/TS2


# %%
## Get trial condition variables — finger

# compute movement direction for each trial (0 - 2π)
angle_data_all2 = np.arctan2(target_data_all2[:,1]-start_data_all2[:,1],
                             target_data_all2[:,0]-start_data_all2[:,0])
angle_data_all2[angle_data_all2<0] += 2*np.pi
angle_data_all2 = np.round(angle_data_all2,4)


# %%
## Combine wrist and finger trial data

# keep only the channels shared by both tasks
keep_chans_all = np.intersect1d(keep_chans_all1, keep_chans_all2)
ch_mask1 = np.isin(keep_chans_all1, keep_chans_all)
ch_mask2 = np.isin(keep_chans_all2, keep_chans_all)

# concatenate wrist and finger trials for the shared channels
neural_data_all = np.concatenate([neural_data_all[:,:,ch_mask1],
                                  neural_data_all2[:,:,ch_mask2]], axis=0)
neural_data_move_all = np.concatenate([neural_data_move_all[:,:,ch_mask1],
                                       neural_data_move_all2[:,:,ch_mask2]], axis=0)
sbp_data_all = np.concatenate([sbp_data_all[:,:,ch_mask1],
                               sbp_data_all2[:,:,ch_mask2]], axis=0)
sbp_data_move_all = np.concatenate([sbp_data_move_all[:,:,ch_mask1],
                                    sbp_data_move_all2[:,:,ch_mask2]], axis=0)

# build the effector identifier (0 = wrist, 1 = finger)
modality_data_all1 = np.zeros_like(angle_data_all)
modality_data_all2 = np.ones_like(angle_data_all2)
modality_data_all = np.concatenate([modality_data_all1, modality_data_all2], axis=0)
modality_data_unique = np.unique(modality_data_all)

# concatenate the trial condition fields
angle_data_all = np.concatenate([angle_data_all, angle_data_all2], axis=0)
delay_data_all = np.concatenate([delay_data_all, delay_data_all2], axis=0)
block_id_data_all = np.concatenate([block_id_data_all, block_id_data_all2], axis=0)
trial_num_data_all = np.concatenate([trial_num_data_all, trial_num_data_all2], axis=0)
trial_duration_data_all = np.concatenate([trial_duration_data_all, trial_duration_data_all2], axis=0)
no_go_bool_all = np.concatenate([no_go_bool_all, no_go_bool_all2], axis=0)

# only concatenate EMG if it is available for both tasks
if 'emg_data_all' in trial_data_delay.keys() and 'emg_data_all' in trial_data_delay2.keys():
    emg_data_all = np.concatenate([emg_data_all, emg_data_all2], axis=0)
    emg_data_move_all = np.concatenate([emg_data_move_all, emg_data_move_all2], axis=0)

# get unique directions and delay durations, and the shortest usable delay
angle_data_unique = np.unique(angle_data_all)
delay_data_unique = np.unique(delay_data_all)
delay_min = delay_data_unique[delay_data_unique >= 0.8].min()

# build condition arrays (direction x effector)
CONDITIONS = [angle_data_unique, modality_data_unique]
CONDITION_ARRAYS = [angle_data_all, modality_data_all]


# %%
## Plot PSTHs for example channels

# example channels to plot, per participant
if participant == 'T16':
    CHS = [7, 45]
if participant == 'T11':
    CHS = [28, 30]

# generate plot (rows = effector, cols = example channel)
_, axs = plt.subplots(2,len(CHS), figsize=(len(CHS)*3,4), sharex=True, sharey='col', constrained_layout=True)

# loop over example channels
for ich, ch in enumerate(CHS):

    # loop over conditions
    for cond_set in itertools.product(*CONDITIONS):

        # wrist on the top row (solid), finger on the bottom row (dashed)
        if cond_set[1] == 0:
            ax = axs[0, ich]
            linestyle = '-'
        else:
            ax = axs[1, ich]
            linestyle = (0, (5, 1))

        # select trials of this condition
        masks = [cond_set[i] == CONDITION_ARRAYS[i] for i in range(len(CONDITIONS))]
        mask = np.all(masks, axis=0)
        mask = mask

        # delay period plotting window: -200 ms to the end of the shortest delay
        T_STA = int(GO_CUE-200/TS)
        T_END = int(GO_CUE+delay_data_all[mask].min()*(1000/TS))
        t_plot = t_data[T_STA:T_END]

        # compute mean and SEM firing rate across trials
        data_plot = neural_data_all[mask][:,T_STA:T_END,ch]
        data_plot_mean = data_plot.mean(axis=0)
        data_plot_sem = data_plot.std(axis=0, ddof=1) / np.sqrt(data_plot.shape[0])

        # color by movement direction
        color = colors.hsv_to_rgb([cond_set[0]/(2*np.pi), 1, 1])

        # plot mean ± SEM
        ax.fill_between(t_plot,
                        data_plot_mean-data_plot_sem,
                        data_plot_mean+data_plot_sem,
                        alpha=0.2, color=color, linewidth=0.0)
        ax.plot(t_plot, data_plot_mean, color=color, linewidth=2, linestyle=linestyle)

    # share the wrist y-limits across both rows
    ylims = axs[0,ich].get_ylim()

    # loop over effector rows
    for i, ax in enumerate(axs[:,ich]):

        # misc. subplot settings
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])

        # offset for scale bar
        scale_offset = [-20, -2]
        ax.set_xlim(-300+scale_offset[0], 1100)
        ax.set_ylim(ylims[0]+scale_offset[1], ylims[1])

        # draw target cue (yellow)
        ax.plot([0,0],[ylims[0], ylims[1]], color='y', linestyle='-', linewidth=2)

        # draw firing rate (vertical) and time (horizontal) scale bars
        if i == 1:
            ax.plot([-300+scale_offset[0]/3,-300+scale_offset[0]/3],
                    [ylims[0]+scale_offset[1]/2, ylims[0]+scale_offset[1]/2+5],
                    color='k', linestyle='-', linewidth=5)
            ax.plot([-300+scale_offset[0]/3,-300+scale_offset[0]/3+200],
                    [ylims[0]+scale_offset[1]/2, ylims[0]+scale_offset[1]/2],
                    color='k', linestyle='-', linewidth=5)

    # report the plotted channel and the trial count per effector
    print(f'ich: {ch}, actual channel: {keep_chans_all[ch]}')
    print(f'modality 0 trials: {((modality_data_all == 0)).sum()}')
    print(f'modality 1 trials: {((modality_data_all == 1)).sum()}')

# save pdf
savepath = os.path.join(save_plot_filepath, f'fig2_{participant}_psths.pdf')
plt.savefig(savepath, format='pdf')

plt.show()


# %%
## Normalize neural data

# soft-normalization constants for the spike and SBP features
SOFT_NORM_SPIKES = 1e-9
SOFT_NORM_SBP = 1e-9

# soft-normalize per block and concatenate the spike and SBP features
neural_data_norm_all, _ = normalize_and_concat(
    neural_data_all, sbp_data_all, neural_data_move_all, sbp_data_move_all,
    block_id_data_all, GO_CUE, TS,
    USE_SBP=True, SOFT_NORM_SPIKES=SOFT_NORM_SPIKES, SOFT_NORM_SBP=SOFT_NORM_SBP,
)


# %%
## Compute cross-validated pairwise correlations

# build condition-averaged and per-trial arrays
X, trialsX, trialNum = build_marg_arrays(neural_data_norm_all,
                                        CONDITIONS, CONDITION_ARRAYS,)

# delay window
T_STA = int(400/TS)
T_END = int(delay_min*1000/TS)

# reshape to (effector, direction, time, channel, trial)
Xcen = trialsX[:,:,:,GO_CUE+T_STA:GO_CUE+T_END:].transpose(2,1,3,0,4)

# subtract the average across directions (and trials)
Xcen = Xcen - np.nanmean(Xcen, axis=(1,4))[:,None,:,:,None]

# average across time and flatten the condition axes
Xcorr = np.nanmean(Xcen, axis=2).reshape((Xcen.shape[0]*Xcen.shape[1],-1,Xcen.shape[-1]))
trialNum_flat = trialNum[0,:,:].T.flatten()

# container for the cross-validated correlation of each condition pair
corr_delay = np.zeros((Xcorr.shape[0], Xcorr.shape[0]))

# loop over condition pairs
for cond1 in range(Xcorr.shape[0]):
    for cond2 in range(Xcorr.shape[0]):

        # select the trials available for each condition
        trialNum1 = int(trialNum_flat[cond1])
        trialNum2 = int(trialNum_flat[cond2])
        X1 = Xcorr[cond1,:,:trialNum1].T
        X2 = Xcorr[cond2,:,:trialNum2].T

        # compute the noise-corrected magnitude of each condition mean
        unbiased_mag1 = cv_distance(X1, np.zeros(X1.shape), subtract_mean=True)[1]
        unbiased_mag2 = cv_distance(X2, np.zeros(X2.shape), subtract_mean=True)[1]

        # get each condition's mean across trials
        Xm1 = np.nanmean(X1, axis=0)
        Xm2 = np.nanmean(X2, axis=0)

        # compute the correlation (mean-centered dot product normalized by the unbiased magnitudes)
        corr_delay[cond1, cond2] = np.dot(Xm1 - np.nanmean(Xm1), Xm2 - np.nanmean(Xm2)) / (unbiased_mag1 * unbiased_mag2)


# %%
## Plot pairwise correlation matrix and wrist-finger quadrant

# color for the quadrant boundaries and cell annotations
c = (0.3,0.3,0.3)

# 1. plot full 16 x 16 correlation matrix
sns.heatmap(corr_delay, cmap=sns.diverging_palette(205, 25, s=90, as_cmap=True),
            vmin=-1, vmax=1, cbar_kws={"ticks": [-1, -0.5, 0, 0.5, 1]}, square=True)

# draw the wrist/finger quadrant boundaries
plt.axvline(8, color=c, linestyle='-', linewidth=3)
plt.axhline(8, color=c, linestyle='-', linewidth=3)

# misc. plot settings
ax = plt.gca()
ax.set_axis_off()
plt.gca().invert_yaxis()
cax = ax.figure.get_children()[-1]
cax.spines['outline'].set_linewidth(0.75)
plt.title('Pairwise neural correlations', fontsize=16)

# save pdf
savepath = os.path.join(save_plot_filepath, f'fig2_{participant}_correlation.pdf')
plt.savefig(savepath, format='pdf')

plt.show()

# 2. plot zoom in on the wrist vs. finger quadrant
sns.heatmap(corr_delay[0:8,8:16], cmap=sns.diverging_palette(205, 25, s=90, as_cmap=True),
            annot=False, square=True, annot_kws={"size": 10, "weight": "normal", "color":c}, fmt= '.2f',
            vmin=-1, vmax=1, cbar_kws={"ticks": [-1, -0.5, 0, 0.5, 1]}, cbar=False)

# annotate every cell, emphasising the matched-direction diagonal
for i in range(8):
    for j in range(8):
        if i == j:
            fontweight = 'bold'
            fontsize = 15
        else:
            fontweight = 'normal'
            fontsize = 13
        plt.text(j+0.5, i+0.5, f'{corr_delay[i,j+8]:.1f}', ha='center', va='center', c=c, fontweight=fontweight, fontsize=fontsize)

# collect the matched-direction correlations and draw a box around each diagonal cell
off_diag = np.zeros(8)
for i in range(8):
    off_diag[i] = corr_delay[i,i+8]
    # draw box
    plt.gca().add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor=c, linewidth=2.5))

print(f'mean off-diagonal: {off_diag.mean()}')

# misc. plot settings
ax = plt.gca()
ax.set_axis_off()
plt.gca().invert_yaxis()
plt.xlim(-0.05,8.05)
plt.ylim(-0.05,8.05)

# save pdf
savepath = os.path.join(save_plot_filepath, f'fig2_{participant}_correlation_zoom.pdf')
plt.savefig(savepath, format='pdf')

plt.show()


# %%
## Alignment index — wrist vs. finger subspace (Elsayed et al. 2016)

# delay window
T_STA_PREP = int(GO_CUE + 400/TS)
T_END_PREP = int(GO_CUE + 800/TS) if participant == 'T5' else int(GO_CUE + delay_min*1000/TS)

# trial masks: long-delay go trials of each effector
ai_mask_wrist  = (delay_data_all >= 0.8) & (no_go_bool_all == 0) & (modality_data_all == 0)
ai_mask_finger = (delay_data_all >= 0.8) & (no_go_bool_all == 0) & (modality_data_all == 1)

# compute alignment indices and VAFs
ai_1to2, ai_2to1, ai_mean, vaf_1to2, vaf_2to1, _, _, _, _ = alignment_index(
    neural_data_norm_all[ai_mask_wrist],
    neural_data_norm_all[ai_mask_finger],
    angle_data_all[ai_mask_wrist],
    angle_data_all[ai_mask_finger],
    t_start=T_STA_PREP,
    t_end=T_END_PREP,
    t_start2=T_STA_PREP,
    t_end2=T_END_PREP,
    n_pcs=10,
    subtract_ci=True,
)

# within wrist alignment index: self-consistency check and yields D_wrist (wrist-subspace PC axes)
ai_1to1, _, _, vaf_1to1, _, D_wrist, _, scale_wrist, _ = alignment_index(
    neural_data_norm_all[ai_mask_wrist],
    neural_data_norm_all[ai_mask_wrist],
    angle_data_all[ai_mask_wrist],
    angle_data_all[ai_mask_wrist],
    t_start=T_STA_PREP,
    t_end=T_END_PREP,
    t_start2=T_STA_PREP,
    t_end2=T_END_PREP,
    n_pcs=10,
    subtract_ci=True,
)

# within finger alignment index: self-consistency check and yields D_finger (finger-subspace PC axes)
ai_2to2, _, _, vaf_2to2, _, D_finger, _, scale_finger, _ = alignment_index(
    neural_data_norm_all[ai_mask_finger],
    neural_data_norm_all[ai_mask_finger],
    angle_data_all[ai_mask_finger],
    angle_data_all[ai_mask_finger],
    t_start=T_STA_PREP,
    t_end=T_END_PREP,
    t_start2=T_STA_PREP,
    t_end2=T_END_PREP,
    n_pcs=10,
    subtract_ci=True,
)

print(f'Alignment index (wrist→wrist):  {ai_1to1:.3f}')
print(f'Alignment index (wrist→finger): {ai_1to2:.3f}')
print(f'Alignment index (finger→wrist): {ai_2to1:.3f}')
print(f'Alignment index (finger→finger):{ai_2to2:.3f}')
print(f'Alignment index (mean):         {ai_mean:.3f}')


# generate alignment index VAF bar plots
_, axes_vaf = plt.subplots(1, 2, figsize=(10, 4))

# bar width for the VAF plots
bar_width = 0.42
# bar positions and labels
n_pcs_ai = len(vaf_1to2)
x_vaf = np.arange(n_pcs_ai)
pc_labels_vaf = [str(i + 1) for i in range(n_pcs_ai)]

# 1. plot VAF of both effectors' data in the wrist subspace
axes_vaf[0].bar(x_vaf - bar_width/2, vaf_1to1*scale_wrist*100, bar_width, color=(0.2, 0.2, 0.9), label='Wrist data')
axes_vaf[0].bar(x_vaf + bar_width/2, vaf_2to1*scale_finger*100, bar_width, color=(0.75, 0, 0), label='Finger data')

# misc. subplot settings
axes_vaf[0].set_xticks(x_vaf)
axes_vaf[0].set_xticklabels(pc_labels_vaf, fontsize=15)
axes_vaf[0].set_xlabel('PC', fontsize=17)
axes_vaf[0].set_ylabel('% of total variance', fontsize=17)
axes_vaf[0].set_title('Wrist subspace', fontweight='bold', color=(0.2, 0.2, 0.9), fontsize=18, pad=-20)
axes_vaf[0].text(0.5, 0.9, f'Alignment index: {ai_2to1:.2f}',
                 transform=axes_vaf[0].transAxes, ha='center', va='bottom', fontsize=16)
axes_vaf[0].tick_params(axis='y', labelsize=15)
axes_vaf[0].legend(frameon=True, loc='center right', bbox_to_anchor=(1.0, 0.67), fontsize=15)
axes_vaf[0].spines[['top', 'right']].set_visible(False)

# 2. plot VAF of both effectors' data in the finger subspace
axes_vaf[1].bar(x_vaf - bar_width/2, vaf_1to2*scale_wrist*100, bar_width, color=(0.2, 0.2, 0.9), label='Wrist data')
axes_vaf[1].bar(x_vaf + bar_width/2, vaf_2to2*scale_finger*100, bar_width, color=(0.75, 0, 0), label='Finger data')

# misc. subplot settings
axes_vaf[1].set_xticks(x_vaf)
axes_vaf[1].set_xticklabels(pc_labels_vaf, fontsize=15)
axes_vaf[1].set_xlabel('PC', fontsize=17)
axes_vaf[1].set_title('Finger subspace', fontweight='bold', color=(0.75, 0, 0), fontsize=18, pad=-20)
axes_vaf[1].text(0.5, 0.9, f'Alignment index: {ai_1to2:.2f}',
                 transform=axes_vaf[1].transAxes, ha='center', va='bottom', fontsize=16)
axes_vaf[1].tick_params(axis='y', labelsize=15)
axes_vaf[1].legend(frameon=True, loc='center right', bbox_to_anchor=(1.0, 0.67), fontsize=15)
axes_vaf[1].spines[['top', 'right']].set_visible(False)

# keep only the right panel's legend
axes_vaf[0].legend().set_visible(False)

# use one common y limit across both panels
vaf_max = max(np.max(vaf_1to1*scale_wrist*100), np.max(vaf_2to1*scale_finger*100),
               np.max(vaf_1to2*scale_wrist*100), np.max(vaf_2to2*scale_finger*100))
for a in axes_vaf:
    a.set_ylim(0, vaf_max * 1.1)
plt.tight_layout()

# save pdf
savepath = os.path.join(save_plot_filepath, f'fig2_{participant}_alignment_index.pdf')
plt.savefig(savepath, format='pdf')

plt.show()


# %%
## 2D projections — wrist and finger data onto wrist PCs vs. finger PCs

# select long-delay go trials of each effector
ai_trial_mask_wrist  = (delay_data_all >= 0.8) & (no_go_bool_all == 0) & (modality_data_all == 0)
ai_trial_mask_finger = (delay_data_all >= 0.8) & (no_go_bool_all == 0) & (modality_data_all == 1)

# average wrist trials per direction and subtract the condition-independent signal
Xdel_wrist, _, _ = build_marg_arrays(
    neural_data_norm_all[ai_trial_mask_wrist],
    [angle_data_unique], [angle_data_all[ai_trial_mask_wrist]])
Xdel_wrist_cis   = Xdel_wrist.mean(axis=1)
Xdel_wrist_nocis = Xdel_wrist - Xdel_wrist_cis[:, None]

# same for the finger trials
Xdel_finger, _, _ = build_marg_arrays(
    neural_data_norm_all[ai_trial_mask_finger],
    [angle_data_unique], [angle_data_all[ai_trial_mask_finger]])
Xdel_finger_cis   = Xdel_finger.mean(axis=1)
Xdel_finger_nocis = Xdel_finger - Xdel_finger_cis[:, None]

# delay window end
t_end_del_samp = int(800/TS) if participant == 'T5' else int(delay_min*1000/TS)
# data source and window for each column of the grid
epoch_slices_2d = [
    (Xdel_wrist_nocis,  GO_CUE + int(0/TS), GO_CUE + t_end_del_samp),
    (Xdel_finger_nocis, GO_CUE + int(0/TS), GO_CUE + t_end_del_samp),
]

# define the 2D projections to plot
epoch_labels_2d = ['Wrist data', 'Finger data']
pc_spaces_2d    = [D_wrist, D_finger]
pc_labels_2d    = ['Wrist subspace', 'Finger subspace']
pc_colors_2d    = [(0.2, 0.2, 0.9), (0.75, 0, 0)]

# trajectory start marker per column
start_marker_2d = ['o', '^']

# project both effectors' data onto both PC spaces
Z_proj_2d = [
    [Xdata[:, :, sta:end].transpose(1, 2, 0) @ D
     for D in pc_spaces_2d]
    for Xdata, sta, end in epoch_slices_2d
]

# generate 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True,
                        gridspec_kw={'hspace': 0.55, 'wspace': -0.1})
fig.subplots_adjust(top=0.84)

# common axis limits across all four panels
plot_lim = 1.10 *max(np.abs(Z[:, :, :2]).max() for epoch in Z_proj_2d for Z in epoch)
ticks_pos = max(1, round(plot_lim * 0.75 / 1.1))  # extent of the partial spines and ticks
arrow_head  = 0.10 * plot_lim  # arrowhead size

# loop over PC spaces (rows) and data sources (cols)
for row in range(2):  # PC space
    for col in range(2):  # data source

        ax = axs[row, col]

        # plot one delay-period trajectory per direction, marking its start and end
        for ia, angle in enumerate(angle_data_unique):
            c = colors.hsv_to_rgb([angle / (2 * np.pi), 1, 1])
            traj = Z_proj_2d[col][row][ia]
            ax.plot(traj[:, 0], traj[:, 1], color=c, lw=1.5)
            ax.scatter(traj[0, 0], traj[0, 1], color=c,
                       marker=start_marker_2d[col], s=25, zorder=5)
            ax.arrow(traj[-2, 0], traj[-2, 1],
                     traj[-1, 0] - traj[-2, 0], traj[-1, 1] - traj[-2, 1],
                     head_width=arrow_head, head_length=arrow_head, overhang=0.3,
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
        ax.text(0.5, 1, epoch_labels_2d[col], transform=ax.transAxes,
                ha='center', va='top', fontsize=12)

# apply single global limit to all panels
axs[0, 0].set_xlim(-plot_lim, plot_lim)
axs[0, 0].set_ylim(-plot_lim, plot_lim)

# draw figure
fig.canvas.draw()

# annotate each panel with the VAF in the first 2 PCs
vaf_grid = [
    [(vaf_1to1[0] + vaf_1to1[1]) * scale_wrist  * 100,
     (vaf_1to2[0] + vaf_1to2[1]) * scale_wrist  * 100],
    [(vaf_2to1[0] + vaf_2to1[1]) * scale_finger * 100,
     (vaf_2to2[0] + vaf_2to2[1]) * scale_finger * 100],
]
for row in range(2):
    for col in range(2):
        axs[row, col].text(0.02, 0.015, f'{vaf_grid[col][row]:.1f}% of\ntotal var.',
                           transform=axs[row, col].transAxes,
                           ha='left', va='bottom', fontsize=11, color='black')

# draw one row title per PC space, centered over its two panels
for row in range(2):
    bbox0 = axs[row, 0].get_position()
    bbox1 = axs[row, 1].get_position()
    fig.text((bbox0.x0 + bbox1.x1) / 2, bbox0.y1 + 0.02,
             pc_labels_2d[row], ha='center', va='bottom', fontsize=14,
             fontweight='bold', color=pc_colors_2d[row])

# save pdf
savepath = os.path.join(save_plot_filepath, f'fig2_{participant}_pca_proj2d.pdf')
plt.savefig(savepath, format='pdf')

plt.show()


# %%
## Build dPCA arrays

# build condition-averaged and per-trial arrays
X, trialsX, trialNum = build_marg_arrays(neural_data_norm_all,
                                        CONDITIONS, CONDITION_ARRAYS,)

# delay window: target cue to the end of the shortest delay
T_STA = int(000/TS)
T_END = int(delay_min*1000/TS)

# single-trial and trial-averaged firing rates handed to dPCA
firingRates = trialsX[:,:,:,GO_CUE+T_STA:GO_CUE+T_END,:]
firingRatesAverage = X[:,:,:,GO_CUE+T_STA:GO_CUE+T_END]
margNames = ['Effector independent', 'Effector dependent', 'Condition independent', 'Interactions']

# plotting order of the marginalizations, with titles and colors
marg_order = [2, 0, 1, 3]
marg_titles = [margNames[2], margNames[0], margNames[1] , margNames[3]]
color_list = [(0.5, 0.5, 0.5),
              (0.2, 0.2, 0.9),
              (0.75, 0, 0),
              (0.5, 0, 0.5)]


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

# run dpca script for 2 marginalizations
output = eng.dpca_analysis_2d(firingRates_matlab,
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
Xfull = firingRatesAverage - np.nanmean(firingRatesAverage, axis=(1,2,3), keepdims=True)

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
plt.figure(figsize=(4, 2))

# plot one bar per dPC type, pooling the effector-dependent and interaction components
for im, marg_list in enumerate([[0], [1], [2,3]]):
    marg_idx = np.concatenate([np.where(whichMarg == marg_order[marg]+1)[1] for marg in marg_list])
    # variance explained jointly by these marginalizations
    SSerr = np.sum((Xfull_flat - dpcaV[:,marg_idx] @ dpcaW[:,marg_idx].T @ Xfull_flat)**2)
    SStot = np.sum(Xfull_flat**2)
    R2 = (SStot - SSerr) / (SStot)
    var = R2
    print(var)

    # draw the bar and label it with the variance explained
    plt.barh(3-im, var*100, color=color_list[im], height=0.5)
    plt.text(var*100+0.25, 3-im, f'{var*100:.1f}%', va='center', ha='left', fontsize=16, color=color_list[im], fontweight='bold')

# misc. plot settings
plt.gca().spines[['top', 'right']].set_visible(False)
plt.gca().yaxis.set_ticks([])
if participant == 'T16':
    plt.gca().xaxis.set_ticks(np.arange(0, 30+1, 10))
else:
    plt.gca().xaxis.set_ticks(np.arange(0, 30+1, 5))
plt.gca().xaxis.set_tick_params(labelsize=14)
plt.gca().yaxis.set_tick_params(labelsize=16)
plt.ylim(0+0.3, 4-0.3)
plt.xlabel('% of total variance\nexplained by dPC type', fontsize=17)

# save pdf
savepath = os.path.join(save_plot_filepath, f'fig2_{participant}_dpca_var_bars.pdf')
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
Xdpca = trialsX[:,:,:,T_STA:T_END]
t_pca = t_data[T_STA:T_END]

# mean-center across conditions, then project onto the dPCA decoder axes
Xcen = Xdpca - np.nanmean(np.nanmean(Xdpca, axis=-1, keepdims=True), axis=(1,2,3), keepdims=True)
Z = Xcen.T @ dpcaW

# dPCs to plot per marginalization (rows)
PCS = [0,1]

# generate plot (rows = dPC, cols = marginalization)
_, axs = plt.subplots(nrows=len(PCS), ncols=3, figsize=(3*4.5,len(PCS)*3.5), sharex=True, sharey=False, facecolor='w', constrained_layout=True)

# participant-specific y limits and ticks
if participant == 'T16':
    ylims = [-1, 1]
    yticks = [-0.5, 0, 0.5, 1]
elif participant == 'T11':
    ylims = [-2, 2]
    yticks = [-2, -1, 0, 1, 2]

# loop over marginalizations (columns), skipping the interaction terms
for marg, _ in enumerate(marg_titles[:-1]):

    # get the components belonging to this marginalization
    marg_i = np.where(whichMarg == marg_order[marg]+1)[1]

    # loop over dPCs (rows)
    for pc, _ in enumerate(PCS):

        # only plot components this marginalization has
        if pc >= len(marg_i):
            continue

        ax = axs[pc,marg]

        # loop over conditions
        for cond_set in itertools.product(*CONDITIONS):
            masks = [cond_set[i] == CONDITION_ARRAYS[i] for i in range(len(CONDITIONS))]
            mask = np.all(masks, axis=0)

            # locate this condition in the direction x effector grid
            i0 = np.where(cond_set[0] == angle_data_unique)[0][0]
            i1 = np.where(cond_set[1] == modality_data_unique)[0][0]

            # compute mean and SEM projection across trials
            Zmarg = np.nanmean(Z[:,:,i1,i0,marg_i[pc]], axis=0)
            Zmarg_sem = np.nanstd(Z[:,:,i1,i0,marg_i[pc]], axis=0, ddof=1) / np.sqrt(np.sum(mask))

            # color by direction; wrist solid, finger dashed
            if cond_set[1] == 0:
                color = colors.hsv_to_rgb([cond_set[0]/(2*np.pi), 1, 1])
                linestyle = '-'
            else:
                color = colors.hsv_to_rgb([cond_set[0]/(2*np.pi), 1, 1])
                linestyle = linestyle = (0, (5, 1))
            linewidth=2

            # plot mean ± SEM
            ax.fill_between(t_pca, Zmarg - Zmarg_sem, Zmarg + Zmarg_sem,
                            color=color, alpha=0.15, linewidth=0)
            ax.plot(t_pca,Zmarg,color=color, linewidth=linewidth, linestyle=linestyle)

        # draw target cue (yellow)
        ax.axvline(0, color='y', linestyle='-', linewidth=3)

        # draw the projection scale bar
        ax.plot([-250, -250], [-0.05, 0.05], c='k', linewidth=6)

        # label the dPC on the leftmost column
        if marg == 0:
            ax.set_ylabel(f'Projection onto dPC {pc+1}' , fontsize=20)
            ax.set_yticks(yticks)

        # enforce a minimum y range
        ylims = ax.get_ylim()
        if ylims[1] - ylims[0] < 0.25:
            ax.set_ylim(-0.125, 0.125)
            ylims = ax.get_ylim()

        # annotate the variance explained by this component
        marg_variance_i = comp_variance[marg_i[pc]] * 100
        ax.text(t_pca[-1], ylims[0] + (ylims[1]-ylims[0])*0.01*0, f'{marg_variance_i:2.2f}%',
                    color='k', fontsize=20, ha='right', va='bottom', fontweight='normal')

        # draw the time scale bar on one panel only
        if marg == 0 and pc == 1:
            ax.plot([100, 300], [ylims[0], ylims[0]], c='k', linewidth=6)

        # misc. subplot settings
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlim(T0-100, TF)

    # title each column with its marginalization name
    axs[0,marg].set_title(f'{marg_titles[marg]}', fontsize=20, fontweight='bold')

# save pdf
savepath = os.path.join(save_plot_filepath, f'fig2_{participant}_dpca_traces.pdf')
plt.savefig(savepath, format='pdf')

plt.show()


# %%
## Run SVM decoding sweep — effector and direction

# only run the sweep when requested
if RUN_DECODING_SWEEP:

    np.random.seed(42)

    # delay window
    T_STA = int(400/TS)
    T_END = int(delay_min*1000/TS)

    # loop over decode targets (3rd is placeholder for the joint effector x direction decode)
    for yi, y_var in enumerate([angle_data_all, modality_data_all, angle_data_all]):

        # only run the joint effector x direction decode
        if yi == 0:
            var_title = 'Direction'
            continue
        elif yi == 1:
            var_title = 'Modality'
            continue
        else:
            var_title = 'Modality + Direction'

        # only use long-delay trials
        trial_mask = (delay_data_all >= 0.8)

        # get integer class labels
        if yi == 2:
            # combined effector x direction labels for the joint decode
            combined_conds_tuple = [(modality, angle) for modality, angle in zip(modality_data_all, angle_data_all)]
            y = np.expand_dims(np.unique(combined_conds_tuple, return_inverse=True, axis=0)[1],axis=1).flatten()
        else:
            y = np.expand_dims(np.unique(y_var, return_inverse=True, axis=0)[1],axis=1).flatten()

        # one-hot the labels, mask to the kept trials, then convert back to class indices
        Y = np.zeros((y.shape[0], np.unique(y).shape[0]))
        for i in range(np.unique(y).shape[0]):
            Y[y==i, i] = 1
        Y = Y[trial_mask]
        y = np.argmax(Y, axis=1)

        # build condition codes for each trial
        cond_codes = np.full(Y.shape[0], np.nan)
        for i, cond_set in enumerate(itertools.product(*CONDITIONS)):
            mask = np.all([cond_set[j] == CONDITION_ARRAYS[j] for j in range(len(CONDITIONS))], axis=0)
            mask = mask[trial_mask]
            cond_codes[mask] = i

        # shuffle trials and then sort by condition so every K-th index is a balanced fold
        idxs = np.arange(Y.shape[0])
        np.random.shuffle(idxs)
        sorted_idxs = np.argsort(cond_codes[idxs])
        sorted_idxs = idxs[sorted_idxs]

        # leave-one-trial-out cross-validation
        cv_splits = Y.shape[0]

        # containers for per-fold predictions and accuracies
        true_all = []
        pred_all = []
        true_all_split = []
        pred_all_split = []
        acc_chance_all = np.zeros(cv_splits)
        acc_test_all = np.zeros(cv_splits)

        # get the delay-window neural features
        X = neural_data_norm_all[:,GO_CUE+T_STA:GO_CUE+T_END,:]
        X = X[trial_mask]

        # train and test one SVM per held-out trial, in parallel
        with Parallel(n_jobs=32, require='sharedmem') as parallel:
            tasks = [
                delayed(decoder_sweep_parallel)(X, y, sorted_idxs, i, cv_splits)
                for i in range(cv_splits)
            ]
            decoding_results = parallel(tasks)
            # unpack the parallel execution results
            for i, item in enumerate(decoding_results):
                acc_chance, acc_test, Y_test, Y_pred = item
                acc_chance_all[i] = acc_chance
                acc_test_all[i] = acc_test
                true_all_split.append(Y_test)
                pred_all_split.append(Y_pred)

        # pool the held-out predictions across folds
        true_all = np.concatenate(true_all_split)
        pred_all = np.concatenate(pred_all_split)

        # compute mean per-fold and pooled decoding accuracy
        acc_folds = 100*np.mean([accuracy_score(true_all_split[i], pred_all_split[i]) for i in range(cv_splits)])
        acc_overall = 100*accuracy_score(true_all, pred_all)

        # compute the corresponding chance levels
        chance_folds = 100*np.mean(acc_chance_all)
        chance_overall = 100*np.bincount(y.ravel()).max()/np.bincount(y.ravel()).sum()

        # report the decoding results
        print(f'Fold accuracies: {acc_folds}')
        print(f'Overall accuracy: {acc_overall}')
        print('-----')
        print(f'Fold chance: {chance_folds}')
        print(f'Overall chance: {chance_overall}')

        # save the decoding results
        savepath = os.path.join(save_plot_filepath, f'fig2_{participant}_decoding_{yi}.pkl')
        with open(savepath, 'wb') as f:
            pickle.dump({'true_all_split': true_all_split,
                            'pred_all_split': pred_all_split,
                            'true_all': true_all,
                            'pred_all': pred_all,
                            'acc_chance_all': acc_chance_all,
                            'acc_test_all': acc_test_all,
                            'sorted_idxs': sorted_idxs,
                            }, f)


# %%
## Plot SVM confusion matrix

# load the cached joint effector x direction decode results
savepath = os.path.join(save_plot_filepath, f'fig2_{participant}_decoding_2.pkl')

# if results are not cached, warn the user and skip plotting
if not os.path.exists(savepath):
    print(f'WARNING: {savepath} not found. Set RUN_DECODING_SWEEP = True to (re)generate it.')
else:
    # load the cached results from the pickle file
    with open(savepath, 'rb') as f:
        decoding_data = pickle.load(f)
        true_all_split = decoding_data['true_all_split']
        pred_all_split = decoding_data['pred_all_split']
        true_all = decoding_data['true_all']
        pred_all = decoding_data['pred_all']
        acc_chance_all = decoding_data['acc_chance_all']
        acc_test_all = decoding_data['acc_test_all']

    # only long-delay trials were decoded
    trial_mask = (delay_data_all >= 0.8)
    cv_splits = len(true_all_split)  # number of folds actually run

    # compute mean per-fold and pooled decoding accuracy
    acc_folds = 100*np.mean([accuracy_score(true_all_split[i], pred_all_split[i]) for i in range(cv_splits)])
    acc_overall = 100*accuracy_score(true_all, pred_all)

    # report the decoding results
    print(f'Fold accuracies: {acc_folds}')
    print(f'Overall accuracy: {acc_overall}')

    # generate plot
    plt.figure(figsize=(8,6))

    # plot the confusion matrix over the pooled folds
    ax = sns.heatmap(sklearn.metrics.confusion_matrix(true_all,
                                                      pred_all,
                                                      normalize='true',
                                                        ),
                                                        cmap='viridis',
                                                      vmin=0,
                                                      vmax=1,cbar=True,)

    # misc. plot settings
    ax.set_xlabel('Predicted effector/direction', fontsize=14)
    ax.set_ylabel('True effector/direction', fontsize=14)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    plt.gca().invert_yaxis()
    plt.title(f'{participant}\nDecoding accuracy: {acc_folds:2.1f}%', fontsize=16)

    # save pdf
    savepath = os.path.join(save_plot_filepath, f'fig2_{participant}_decoding_svm.pdf')
    plt.savefig(savepath, format='pdf')

    plt.show()


# %%
## Plot EMG

# only plot if EMG data is available
if 'emg_data_all' in trial_data_delay.keys():

    # per-effector time axes, bin widths, go-cue bins, and row labels
    t_data_list      = [t_data, t_data2]
    TS_list          = [TS, TS2]
    GO_CUE_MOVE_list = [GO_CUE_MOVE, GO_CUE_MOVE2]
    label_list       = ['Wrist', 'Finger']

    # panel counts, and the plotting window offsets
    n_emg_ch  = emg_data_all.shape[2]
    n_mod     = len(modality_data_unique)
    prep_win  = 1000  # ms of delay context before the go cue
    t_lag     = 40  # ms lag to align recorded EMG to the neural signal

    # normalize EMG per block, baseline-centered, using both epochs
    _, emg_move_norm_all = normalize_data(
        emg_data_all, block_id_data_all,
        neural_data_move_all=emg_data_move_all,
        SOFT_NORM=1e-9, center_baseline=True, use_move_data=True,
        GO_CUE=GO_CUE, TS=TS, plot=False)

    # generate plot (rows = effector, cols = EMG channel)
    _, axs = plt.subplots(nrows=n_mod, ncols=n_emg_ch,
                          figsize=(2.6*n_emg_ch, 2*n_mod),
                          sharex=True, sharey='col', facecolor='w')

    # loop over effectors (rows)
    for modality in modality_data_unique:

        # get this effector's row, line style, trials, and timing
        row           = int(modality)
        linestyle     = (0, (5, 1)) if modality == 1 else '-'
        mod_mask      = modality_data_all == modality
        t_data_m      = t_data_list[row]
        TS_m          = TS_list[row]
        GO_CUE_MOVE_m = GO_CUE_MOVE_list[row]

        # select this effector's normalized movement-period EMG
        emg_move_norm_m = emg_move_norm_all[mod_mask]

        # only keep long-delay go trials of this effector
        angle_unique_m = np.unique(angle_data_all[mod_mask])
        go_mask = (no_go_bool_all[mod_mask] == 0) & (delay_data_all[mod_mask] >= 0.8)

        # window from prep_win ms before the go cue to 1.5 s after it
        T_STA = int(GO_CUE_MOVE_m - prep_win/TS_m)
        T_END = int(GO_CUE_MOVE_m + 1.5*(1000/TS_m))

        # loop over EMG channels (cols)
        for ich in range(n_emg_ch):

            ax = axs[row, ich]

            # loop over directions
            for angle in angle_unique_m:

                # select trials of this direction and color by direction
                mask = go_mask & (angle_data_all[mod_mask] == angle)
                color = colors.hsv_to_rgb([angle/(2*np.pi), 0.8, 0.8])

                # compute mean and SEM EMG across trials
                emg = emg_move_norm_m[mask][:,T_STA:T_END,ich]
                sem = emg.std(axis=0, ddof=1) / np.sqrt(emg.shape[0])

                # plot mean ± SEM (lag-corrected)
                ax.fill_between(t_data_m[T_STA:T_END]-t_lag,
                                emg.mean(axis=0)-sem, emg.mean(axis=0)+sem,
                                alpha=0.2, color=color, linewidth=0.0)
                ax.plot(t_data_m[T_STA:T_END]-t_lag,
                        emg.mean(axis=0), color=color, alpha=1, linewidth=2, linestyle=linestyle)

            # draw go cue (green)
            ax.axvline(0, color='g', linestyle='-', linewidth=2)

            # misc. subplot settings
            ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
            ax.set_yticks([])
            ax.set_xticks([])

            # label the channel on the top row and the effector on the first column
            if row == 0:
                ax.set_title(f'EMG ch. {ich+1}', fontsize=12)
            if ich == 0:
                ax.set_ylabel(label_list[row], fontsize=10)

    plt.tight_layout()

    # save pdf
    savepath = os.path.join(save_plot_filepath,
                            f'fig2_{participant}_emg.pdf')
    plt.savefig(savepath, format='pdf')

    plt.show()

# %%
