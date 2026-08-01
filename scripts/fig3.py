# %%
## Imports

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import sklearn
from sklearn.metrics import accuracy_score
import pickle
import itertools

from sklearn.decomposition import PCA

sys.path.insert(0, '../lib')
from utils import build_marg_arrays, normalize_and_concat, load_trial_data_h5
from analyses import run_decoding_sweep, alignment_index


# %%
## Load data

# select participant
participant = 'T16'
# participant = 'T11'

# whether to (re)compute the SVM decoding sweep and save .pkl, or just use the
# cached .pkl results to plot (warns if a .pkl is missing)
RUN_DECODING_SWEEP = True

# load trialized delay-period data
filename_delay = f'./data/fig3_{participant}_delay.h5'
trial_data_delay = load_trial_data_h5(filename_delay)

# load trialized movement-period data
filename_move = f'./data/fig3_{participant}_move.h5'
trial_data_move = load_trial_data_h5(filename_move)

# define save path for plots
save_plot_filepath = './plots/fig3/'
if not os.path.exists(save_plot_filepath):
    os.makedirs(save_plot_filepath)


# %%
## Get arrays from the trialized data dicts

# identify trials with NaNs in either the delay or the movement period
nan_mask1 = np.isnan(trial_data_delay['spike_data_all']).any(axis=(1,2))
nan_mask2 = np.isnan(trial_data_move['spike_data_all']).any(axis=(1,2))
nan_mask = nan_mask1 | nan_mask2

# get neural data for the delay and movement periods (firing rates & SBP)
neural_data_all = trial_data_delay['spike_data_all'][~nan_mask]
neural_data_move_all = trial_data_move['spike_data_all'][~nan_mask]
sbp_data_all = trial_data_delay['sbp_data_all'][~nan_mask]
sbp_data_move_all = trial_data_move['sbp_data_all'][~nan_mask]

# get trial info (target, start and opening positions, delay duration, block id)
target_data_all = trial_data_delay['target_data_all'][~nan_mask]
start_data_all = trial_data_delay['start_data_all'][~nan_mask]
opening_data_all = trial_data_delay['opening_data_all'][~nan_mask]
delay_data_all = trial_data_delay['delay_data_all'][~nan_mask]
block_id_data_all = trial_data_delay['block_id_data_all'][~nan_mask]

# get timing info (bin width in ms, and the bin index of the go cue)
t_data = trial_data_delay['t_data']
TS = t_data[1]-t_data[0]
T_START = int(t_data[0])
GO_CUE = -int(T_START/TS)

# get channel count and the kept (active) channel indices
N_CHANNELS = neural_data_all.shape[2]
keep_chans_all = trial_data_delay['keep_chans_all']

# convert firing rates to Hz
neural_data_all *= 1000/TS
neural_data_move_all *= 1000/TS


# %%
## Get trial condition variables

# compute initial movement direction per trial (start to opening) and the unique directions
initial_angle_data_all = np.round(np.arctan2(opening_data_all[:,1]-start_data_all[:,1], opening_data_all[:,0]-start_data_all[:,0]),4)
initial_angle_data_all[initial_angle_data_all<0] += 2*np.pi
initial_angle_data_all = np.round(initial_angle_data_all,4)
initial_angle_data_unique = np.unique(initial_angle_data_all)

# compute final target direction per trial (start to target) and the unique directions
target_angle_data_all = np.arctan2(target_data_all[:,1]-start_data_all[:,1], target_data_all[:,0]-start_data_all[:,0])
target_angle_data_all[target_angle_data_all<0] += 2*np.pi
target_angle_data_all = np.round(target_angle_data_all,4)
target_angle_data_unique = np.unique(target_angle_data_all)

# compute curvature per trial as target minus initial direction
angle_diff = target_angle_data_all - initial_angle_data_all
# wrap to [-π, π]
angle_diff[np.abs(angle_diff) > np.pi] -= 2*np.pi * np.sign(angle_diff[np.abs(angle_diff) > np.pi])
angle_diff_all = np.round(angle_diff,4)
angle_diff_unique = np.unique(angle_diff_all)

# get unique delay durations and the shortest usable delay (>= 0.8 s)
delay_data_unique = np.unique(delay_data_all)
delay_min = delay_data_unique[delay_data_unique >= 0.8].min()

# build condition lists (initial direction × curvature)
CONDITIONS = [initial_angle_data_unique, angle_diff_unique]
CONDITION_ARRAYS = [initial_angle_data_all, angle_diff_all]

# one color per curvature condition
curvature_colors = [
    colors.hsv_to_rgb([163/360, 1.00, 0.85]),   # -90°
    colors.hsv_to_rgb([201/360, 0.93, 0.92]),   # -45°
    colors.hsv_to_rgb([228/360, 0.83, 0.72]),   #   0°
    colors.hsv_to_rgb([286/360, 0.84, 0.78]),   #  45°
    colors.hsv_to_rgb([320/360, 1.00, 0.45]),   #  90°
]

# %%
## Plot PSTHs for example channels

# example channels to plot, per participant
if participant == 'T16':
    CHS = [8, 3, 32]
if participant == 'T11':
    CHS = [28, 4, 16]

# delay period plotting window: -200 ms to the end of the shortest delay
T_STA = int(-200/TS)
T_END = int(delay_min*1000/TS)
t_plot = np.arange(-200, delay_min*1000, 20)

# generate plot
_, axs = plt.subplots(3, len(CHS), figsize=(len(CHS)*2.75, 7.5), sharex=True, sharey='col', constrained_layout=True)

# loop over example channels, one column of PSTHs each
for ich, ch in enumerate(CHS):

    # 1. plot PSTHs grouped by initial direction
    ax = axs[0, ich]

    # loop over initial directions
    for ia, angle in enumerate(initial_angle_data_unique):

        # select trials with this initial direction
        mask = (initial_angle_data_all == angle)

        # compute mean and SEM firing rate across trials
        data_plot = neural_data_all[mask][:,GO_CUE+T_STA:GO_CUE+T_END,ch]
        data_plot_mean = data_plot.mean(axis=0)
        data_plot_sem = data_plot.std(axis=0, ddof=1) / np.sqrt(data_plot.shape[0])

        # color by initial direction
        color = colors.hsv_to_rgb([angle/(2*np.pi), 0.7, 1])

        # plot mean ± SEM
        ax.fill_between(t_plot, data_plot_mean - data_plot_sem, data_plot_mean + data_plot_sem,
                         alpha=0.2, color=color, linewidth=0.0)
        ax.plot(t_plot, data_plot_mean, color=color, linewidth=2)

    # 2. plot PSTHs grouped by target direction
    ax = axs[1, ich]

    # loop over target directions
    for ia, angle in enumerate(target_angle_data_unique):

        # select trials with this target direction
        mask = (target_angle_data_all == angle)

        # compute mean and SEM firing rate across trials
        data_plot = neural_data_all[mask][:,GO_CUE+T_STA:GO_CUE+T_END,ch]
        data_plot_mean = data_plot.mean(axis=0)
        data_plot_sem = data_plot.std(axis=0, ddof=1) / np.sqrt(data_plot.shape[0])

        # color by target direction
        color = colors.hsv_to_rgb([angle/(2*np.pi), 1, 0.8])

        # plot mean ± SEM
        ax.fill_between(t_plot, data_plot_mean - data_plot_sem, data_plot_mean + data_plot_sem,
                         alpha=0.2, color=color, linewidth=0.0)
        ax.plot(t_plot, data_plot_mean, color=color, linewidth=2)

    # 3. plot PSTHs grouped by curvature
    ax = axs[2, ich]

    # loop over curvatures
    for ia, angle in enumerate(angle_diff_unique):

        # select trials with this curvature
        mask = (angle_diff_all == angle)

        # compute mean and SEM firing rate across trials
        data_plot = neural_data_all[mask][:,GO_CUE+T_STA:GO_CUE+T_END,ch]
        data_plot_mean = data_plot.mean(axis=0)
        data_plot_sem = data_plot.std(axis=0, ddof=1) / np.sqrt(data_plot.shape[0])

        # color by curvature
        color = curvature_colors[ia]

        # plot mean ± SEM
        ax.fill_between(t_plot, data_plot_mean - data_plot_sem, data_plot_mean + data_plot_sem,
                         alpha=0.2, color=color, linewidth=0.0)
        ax.plot(t_plot, data_plot_mean, color=color, linewidth=2)

    ylims = axs[2,ich].get_ylim()

    for i, ax in enumerate(axs[:,ich]):

        # misc. subplot settings
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])

        # scale bar offset
        scale_offset = [-20, -2]
        ax.set_xlim(-300+scale_offset[0], 1100)
        ax.set_ylim(ylims[0]+scale_offset[1], ylims[1])

        # draw target cue (yellow)
        ax.plot([0,0],[ylims[0], ylims[1]], color='y', linestyle='-', linewidth=3)

        # draw firing rate and time scale bars on the bottom row only
        if i == 2:
            ax.plot([-300+scale_offset[0]/3,-300+scale_offset[0]/3],[ylims[0]+scale_offset[1]/2, ylims[0]+scale_offset[1]/2+5],
                    color='k', linestyle='-', linewidth=5)
            ax.plot([-300+scale_offset[0]/2,-300+200],[ylims[0]+scale_offset[1]/2, ylims[0]+scale_offset[1]/2],
                    color='k', linestyle='-', linewidth=5)

    # report the plotted channel and the trial count
    print(f'ich: {ich}, actual channel: {keep_chans_all[ch]}')
    print(f'trials: {target_angle_data_all.shape[0]}')

# save pdf
savepath = os.path.join(save_plot_filepath, f'fig3_{participant}_psths.pdf')
plt.savefig(savepath, format='pdf')

plt.show()


# %%
## ANOVA for single-channel tuning

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

# containers for the per-channel p-values
anova_results_initial = np.zeros((N_CHANNELS))
anova_results_target = np.zeros((N_CHANNELS))
anova_results_curvature = np.zeros((N_CHANNELS))

# loop over channels, testing tuning of the mean delay-period firing rate
for ch in range(N_CHANNELS):

    # delay window: 400 ms after the target cue to the end of the shortest delay
    T_STA = int(GO_CUE+(400/TS))
    T_END = int(GO_CUE+(delay_min*1000/TS))

    # one row per trial: conditions and the window-averaged firing rates for this channel
    df = pd.DataFrame({
        'initial_direction': initial_angle_data_all,
        'target_direction': target_angle_data_all,
        'curvature': angle_diff_all,
        'neural_data_ch': neural_data_all[:,
                                          T_STA:T_END,
                                          ch].mean(axis=1),
    })

    # run ANOVA for initial and target direction
    model = ols("""neural_data_ch ~ C(initial_direction) + C(target_direction)""", data = df).fit()
    anova = sm.stats.anova_lm(model, typ = 2)

    # get FDR-corrected p-values
    pvals = anova['PR(>F)'].values[:-1]
    corrected_pvals = multipletests(pvals, method='fdr_bh')[1]

    # save the initial and target direction p-values
    p = corrected_pvals[0]
    anova_results_initial[ch] = p
    p = corrected_pvals[1]
    anova_results_target[ch] = p

    # test curvature while controlling for initial direction
    model_curv = ols("""neural_data_ch ~ C(initial_direction) + C(curvature)""", data = df).fit()
    anova_curv = sm.stats.anova_lm(model_curv, typ = 2)

    # get FDR-corrected p-values
    pvals_curv = anova_curv['PR(>F)'].values[:-1]
    corrected_pvals_curv = multipletests(pvals_curv, method='fdr_bh')[1]
    anova_results_curvature[ch] = corrected_pvals_curv[1]


# %%
## Plot single-channel tuning results

# significance threshold
sig = 0.01

plt.figure(figsize=(4.5, 4.25))

# 1. plot % of significant channels for initial direction
anova_sig = anova_results_initial < sig
sig_percent = anova_sig.sum(axis=0) / N_CHANNELS * 100
plt.bar(0, sig_percent, color=(0.2, 0.8, 0.2), width=0.75)

print(f'\nInitial direction ANOVA: {sig_percent:.1f}% of channels')
print(np.where(anova_sig)[0])

# 2. plot % of significant channels for target direction
anova_sig = anova_results_target < sig
sig_percent = anova_sig.sum(axis=0) / N_CHANNELS * 100
plt.bar(2, sig_percent, color=(0.8, 0.2, 0.2), width=0.75)

print(f'\nTarget direction ANOVA: {sig_percent:.1f}% of channels')
print(np.where(anova_sig)[0])

# 3. plot % of significant channels for curvature
anova_sig = anova_results_curvature < sig
sig_percent = anova_sig.sum(axis=0) / N_CHANNELS * 100
plt.bar(4, sig_percent, color=(0.2, 0.2, 0.8), width=0.75)

print(f'\nCurvature ANOVA: {sig_percent:.1f}% of channels')
print(np.where(anova_sig)[0])

# annotate participant and channel count
plt.text(x=2, y=100, s=f'{participant}',  fontsize=20,
            color='k', ha='center', va='top')
plt.text(x=2, y=90, s=f'{N_CHANNELS} channels',  fontsize=16,
            color='k', ha='center', va='top') #, fontweight='bold')

# misc. plot settings
plt.xticks([0, 2, 4], ['Initial\ndirection', 'Target\ndirection', 'Curvature'],
           rotation=0, fontsize=16)
plt.yticks(np.arange(0, 101, 20), fontsize=14)
plt.ylabel('% of channels tuned to feature', fontsize=16)
plt.gca().axes.spines[['top', 'right']].set_visible(False)
plt.xlim(-1., 5.)
plt.ylim(0, 100)

# save pdf
savepath = os.path.join(save_plot_filepath, f'fig3_{participant}_tuned_chans.pdf')
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
## Alignment index — curvature conditions (Elsayed et al. 2016)

# compute a delay-period subspace per curvature group, then compare them pairwise

# delay window
T_STA_AI = GO_CUE + int(400/TS)
T_END_AI = GO_CUE + int(delay_min*1000/TS)
N_PCS = 10  # PCs retained per subspace

# trial mask: trials with sufficient delay
ai_mask = delay_data_all >= 0.8

# integer direction condition id (shared labelling across curvature groups)
dir_to_int = {a: i for i, a in enumerate(initial_angle_data_unique)}
dir_cond_id = np.array([dir_to_int[a] for a in initial_angle_data_all])

# number of curvature groups and their degree labels
n_curv = len(angle_diff_unique)
curv_labels = [f'{int(round(np.degrees(c)))}°' for c in angle_diff_unique]

# one color per curvature group
curv_colors = list(curvature_colors)

# containers for the per-group subspaces, the AI matrix, and the per-PC VAF
D_curv = [None] * n_curv
ai_matrix = np.zeros((n_curv, n_curv))
vaf_matrix = np.zeros((n_curv, n_curv, N_PCS))
scale_curv = np.zeros(n_curv)  # per-group ev.sum()/total_var

# loop over curvature groups, computing each group's own subspace
for ci in range(n_curv):
    mask_ci = ai_mask & (angle_diff_all == angle_diff_unique[ci])
    ai_ii, _, _, vaf_ii, _, D_ci, _, scale_ci, _ = alignment_index(
        neural_data_norm_all[mask_ci],
        neural_data_norm_all[mask_ci],
        dir_cond_id[mask_ci],
        dir_cond_id[mask_ci],
        t_start=T_STA_AI, t_end=T_END_AI,
        n_pcs=N_PCS, subtract_ci=True,
    )
    D_curv[ci] = D_ci
    ai_matrix[ci, ci] = ai_ii
    vaf_matrix[ci, ci] = vaf_ii
    scale_curv[ci] = scale_ci  # depends only on this group's own PCA

# loop over curvature group pairs
for ci, cj in itertools.combinations(range(n_curv), 2):
    mask_ci = ai_mask & (angle_diff_all == angle_diff_unique[ci])
    mask_cj = ai_mask & (angle_diff_all == angle_diff_unique[cj])
    ai_ij, ai_ji, _, vaf_ij, vaf_ji, _, _, _, _ = alignment_index(
        neural_data_norm_all[mask_ci],
        neural_data_norm_all[mask_cj],
        dir_cond_id[mask_ci],
        dir_cond_id[mask_cj],
        t_start=T_STA_AI, t_end=T_END_AI,
        n_pcs=N_PCS, subtract_ci=True,
    )
    ai_matrix[ci, cj] = ai_ij
    ai_matrix[cj, ci] = ai_ji
    vaf_matrix[ci, cj] = vaf_ij
    vaf_matrix[cj, ci] = vaf_ji

# convert per-PC VAF from top-N-subspace fraction to fraction of total variance
for ci in range(n_curv):
    vaf_matrix[ci] *= scale_curv[ci]

print('Alignment index matrix (row=data, col=subspace):')
print(np.round(ai_matrix, 3))


# 1. plot alignment index heatmap (row = data, col = subspace)
_, ax_ai = plt.subplots(figsize=(n_curv + 1.0, n_curv * 0.9))
im = ax_ai.pcolormesh(ai_matrix, vmin=0, vmax=1, cmap='magma')

# misc. subplot settings
ax_ai.set_aspect('equal')
ax_ai.set_xticks(np.arange(n_curv) + 0.5)
ax_ai.set_yticks(np.arange(n_curv) + 0.5)
ax_ai.set_xticklabels(curv_labels, rotation=0, fontsize=12)
ax_ai.set_yticklabels(curv_labels, fontsize=12)
ax_ai.set_xlabel('Subspace (curvature)', fontsize=15)
ax_ai.set_ylabel('Data (curvature)', fontsize=15)
ax_ai.set_title(f'{participant}\nAlignment index', fontsize=18, pad=14)

# add colorbar
cbar = plt.colorbar(im, ax=ax_ai)
cbar.ax.tick_params(labelsize=12)

# annotate each cell with its alignment index
for ci in range(n_curv):
    for cj in range(n_curv):
        text_color = 'white' if ai_matrix[ci, cj] < 0.6 else 'black'
        ax_ai.text(cj + 0.5, ci + 0.5, f'{ai_matrix[ci, cj]:.2f}',
                   ha='center', va='center', fontsize=11, color=text_color)

plt.tight_layout()

# save pdf
savepath = os.path.join(save_plot_filepath,
    f'fig3_{participant}_ai_curvature_heatmap.pdf')
plt.savefig(savepath, format='pdf')

plt.show()


# 2. plot per-PC VAF bars, one panel per subspace
bar_width = 0.8 / n_curv
x_pc = np.arange(N_PCS)
pc_labels = [str(i + 1) for i in range(N_PCS)]
vaf_max = vaf_matrix.max() * 100 * 1.15

# custom 3-row layout for 5 curvature groups
if n_curv == 5:
    from matplotlib.gridspec import GridSpec as GridSpec
    fig_vaf = plt.figure(figsize=(12, 14))
    gs = GridSpec(3, 4, figure=fig_vaf, hspace=0.45, wspace=0.35)
    ax_specs = [gs[0, 0:2], gs[0, 2:4], gs[1, 1:3], gs[2, 0:2], gs[2, 2:4]]
    axes_vaf = [fig_vaf.add_subplot(ax_specs[panel_i]) for panel_i in range(n_curv)]
    for ax_vaf in axes_vaf[1:]:
        ax_vaf.sharey(axes_vaf[0])
    ylabel_panels = {0, 2, 3}
    hide_ytick_panels = {1, 4}
# single row of panels for any other number of curvature groups
else:
    fig_vaf, axes_vaf = plt.subplots(1, n_curv, figsize=(5 * n_curv, 4), sharey=True)
    if n_curv == 1:
        axes_vaf = [axes_vaf]
    ylabel_panels = {0}
    hide_ytick_panels = set(range(1, n_curv))

# loop over subspaces, one panel each
for cj in range(n_curv):

    ax_vaf = axes_vaf[cj]

    # loop over curvature groups, drawing one bar per PC
    for ci in range(n_curv):
        offset = (ci - (n_curv - 1) / 2) * bar_width
        ax_vaf.bar(x_pc + offset, vaf_matrix[ci, cj] * 100, bar_width,
                color=curv_colors[ci],
                label=f'{curv_labels[ci]}')

    # misc. subplot settings
    ax_vaf.set_xticks(x_pc)
    ax_vaf.set_xticklabels(pc_labels, fontsize=15)
    ax_vaf.set_xlabel('PC', fontsize=17)
    if cj in ylabel_panels:
        ax_vaf.set_ylabel('% of total variance\nexplained', fontsize=17)
    ax_vaf.set_title(f'{curv_labels[cj]} curvature subspace', fontsize=18,
                  fontweight='bold', color=curv_colors[cj], pad=-20)
    ax_vaf.tick_params(axis='y', labelsize=15)
    ax_vaf.set_ylim(0, vaf_max)
    if cj == n_curv // 2:
        ax_vaf.legend(title='Data (curvature)', title_fontsize=18, frameon=True, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=15)
    ax_vaf.spines[['top', 'right']].set_visible(False)

# hide the redundant y tick labels
for cj in hide_ytick_panels:
    plt.setp(axes_vaf[cj].get_yticklabels(), visible=False)

plt.tight_layout()

# save pdf
savepath = os.path.join(save_plot_filepath,
    f'fig3_{participant}_ai_curvature_vaf.pdf')
plt.savefig(savepath, format='pdf')

plt.show()


# %%
## Plot 3D PCA of initial vs. final direction tuning — schematic

# condition pairings for the three PCA plots
CONDITIONS_all = [[target_angle_data_unique, angle_diff_unique],
                  [initial_angle_data_unique, angle_diff_unique],
                  [angle_diff_unique, initial_angle_data_unique]]
CONDITION_ARRAYS_all = [[target_angle_data_all, angle_diff_all],
                        [initial_angle_data_all, angle_diff_all],
                        [angle_diff_all, initial_angle_data_all]]
ID_all = ['final', 'initial', 'curvature']

# loop over condition pairings, one 3D figure each
for CONDITIONS, CONDITION_ARRAYS, ID in zip(CONDITIONS_all, CONDITION_ARRAYS_all, ID_all):

    # build the condition-marginalized activity array
    X, _, _ = build_marg_arrays(neural_data_norm_all,
                                            CONDITIONS, CONDITION_ARRAYS,)

    # subtract the condition-independent signal (CIS)
    Xcis = X.mean(axis=(1,2))
    Xnocis = X - Xcis[:,None,None]

    # fit PCA on the delay-period, time-averaged condition means
    T_STA = int(400/TS)
    T_END = int(delay_min*1000/TS)
    Xpca = Xnocis[:,:,:,GO_CUE+T_STA:GO_CUE+T_END].mean(axis=-1)
    Xpca = Xpca.reshape((Xpca.shape[0], -1))
    pca = PCA(n_components=3)
    pca.fit(Xpca.T)
    pc_variance = pca.explained_variance_ratio_
    print(np.sum(pc_variance))
    print(pc_variance)

    # marker and line scaling
    PCS = [0,1,2]
    linewidth_mult = 0.75
    markersize_mult = 1

    # generate 3D plot
    fig = plt.figure(figsize=(10,8), facecolor='w')
    ax = fig.add_subplot(111, projection='3d')

    # average over the delay period, then project onto the top 3 PCs
    T_STA = int(GO_CUE+(400/TS))
    T_END = int(GO_CUE+delay_min*(1000/TS))
    Zmean_arr = (Xnocis[:,:,:,T_STA:T_END].mean(axis=-1).T @ pca.components_.T)

    # loop over the first condition (marker shape, with its own centroid)
    for j in range(Zmean_arr.shape[1]):

        # centroid of this condition, averaged over the second condition
        mean_cond = Zmean_arr.mean(axis=0)[j]

        if ID == 'curvature':

            # loop over initial directions within this curvature group
            for i in range(Zmean_arr.shape[0]):

                cond_set = (CONDITIONS[0][j], CONDITIONS[1][i])

                # face color encodes the initial direction, edge color the curvature
                dir_angle = CONDITIONS[1][i]
                dir_angle_col = dir_angle if dir_angle >= 0 else dir_angle + 2*np.pi
                face_color = colors.hsv_to_rgb([dir_angle_col/(2*np.pi), 1, 1])
                edge_color = curvature_colors[j]

                # marker shape encodes the curvature
                curv_val = CONDITIONS[0][j]
                if curv_val == angle_diff_unique[0]:
                    marker='D'
                    linewidth=2*linewidth_mult
                    markersize=35*markersize_mult
                elif curv_val == angle_diff_unique[1]:
                    marker='v'
                    linewidth=2*linewidth_mult
                    markersize=50*markersize_mult
                elif curv_val == angle_diff_unique[2]:
                    marker='o'
                    linewidth=2*linewidth_mult
                    markersize=40*markersize_mult
                elif curv_val == angle_diff_unique[3]:
                    marker='^'
                    linewidth=2*linewidth_mult
                    markersize=50*markersize_mult
                else:
                    marker='s'
                    linewidth=2*linewidth_mult
                    markersize=40*markersize_mult

                # highlight straight-curvature (0°) conditions, greying out the curved ones
                straight = angle_diff_unique[len(angle_diff_unique)//2]
                is_straight = (curv_val == straight)
                if is_straight:
                    alpha = 1
                    s_mult = 2
                else:
                    face_color = colors.hsv_to_rgb([dir_angle_col/(2*np.pi), 0, 0.65])
                    edge_color = colors.hsv_to_rgb([dir_angle_col/(2*np.pi), 0, 0.1])
                    alpha = 0.5
                    s_mult = 1

                # PC projection of this condition pair
                curr = Zmean_arr[i,j]

                # draw a line from the centroid to the condition
                if is_straight:
                    ax.plot([mean_cond[PCS[0]], curr[PCS[0]]],
                            [mean_cond[PCS[1]], curr[PCS[1]]],
                            [mean_cond[PCS[2]], curr[PCS[2]]],
                            color=edge_color, linewidth=linewidth, linestyle=':', alpha=0.5)

                # draw the condition marker
                ax.scatter(curr[PCS[0]], curr[PCS[1]], curr[PCS[2]],
                           s=markersize*0.8*s_mult, marker=marker, alpha=alpha,
                           linewidth=1.75*linewidth_mult*np.sqrt(s_mult),
                           facecolors=face_color, edgecolors=edge_color)

            # draw the centroid marker for the straight-curvature group only
            if CONDITIONS[0][j] == straight:
                ax.scatter(mean_cond[PCS[0]], mean_cond[PCS[1]], mean_cond[PCS[2]],
                           color=curvature_colors[j], s=100*markersize_mult, marker='o', alpha=1, linewidth=0)

        else:

            # loop over the second condition (curvature) within this direction
            for i in range(Zmean_arr.shape[0]):

                # marker shape and color encode the pair of conditions
                cond_set = (CONDITIONS[0][j], CONDITIONS[1][i])
                angle = cond_set[0]
                if j == 0:
                    angle += 2*np.pi
                angle_col = angle
                if cond_set[1] == CONDITIONS[1][0]:
                    color = colors.hsv_to_rgb([angle_col/(2*np.pi), 0.33, 1])
                    marker='D'
                    linewidth=2 * linewidth_mult
                    markersize=35* markersize_mult
                    empty = True
                elif cond_set[1] == CONDITIONS[1][1]:
                    color = colors.hsv_to_rgb([angle_col/(2*np.pi), 0.66, 1])
                    marker='v'
                    linewidth=2* linewidth_mult
                    markersize=50 * markersize_mult
                    empty = True
                elif cond_set[1] == CONDITIONS[1][2]:
                    color = colors.hsv_to_rgb([angle_col/(2*np.pi), 1, 1])
                    marker='o'
                    linewidth=2* linewidth_mult
                    markersize=40* markersize_mult
                    empty = True
                elif cond_set[1] == CONDITIONS[1][3]:
                    color = colors.hsv_to_rgb([angle_col/(2*np.pi), 1, 0.66])
                    marker='^'
                    linewidth=2* linewidth_mult
                    markersize=50* markersize_mult
                    empty = True
                elif cond_set[1] == CONDITIONS[1][4]:
                    color = colors.hsv_to_rgb([angle_col/(2*np.pi), 1, 0.33])
                    marker='s'
                    linewidth=2 * linewidth_mult
                    markersize=40* markersize_mult
                    empty = True
                # guard for any other curvature conditions; drawn as filled 'x' markers
                else:
                    color = colors.hsv_to_rgb([angle_col/(2*np.pi), 1.0, 0.5])
                    marker='x'
                    linewidth=2 * linewidth_mult
                    markersize=35* markersize_mult
                    empty = False

                # highlight one example direction and grey out the rest
                highlight = target_angle_data_unique[1]
                if cond_set[0] == highlight:
                    if ID == 'final':
                        color = colors.hsv_to_rgb([angle_col/(2*np.pi), 1, 0.8])
                    else:
                        color = colors.hsv_to_rgb([angle_col/(2*np.pi), 0.7, 1])
                    alpha = 1
                    s_mult = 2
                else:
                    color = colors.hsv_to_rgb([angle_col/(2*np.pi), 0, 0.1])
                    alpha = 0.5
                    s_mult = 1

                # PC projection of this condition pair
                curr = Zmean_arr[i,j]

                # draw a line from the centroid to the condition
                if cond_set[0] == highlight:
                    ax.plot([mean_cond[PCS[0]], curr[PCS[0]]],
                            [mean_cond[PCS[1]], curr[PCS[1]]],
                            [mean_cond[PCS[2]], curr[PCS[2]]],
                            color=color, linewidth=linewidth, linestyle=':', alpha=0.5)

                # draw the condition marker, open or filled
                if empty:
                    ax.scatter(curr[PCS[0]],
                                curr[PCS[1]],
                                curr[PCS[2]],
                                s=markersize*0.8*s_mult, marker=marker, alpha=alpha, linewidth=1.75*linewidth_mult*np.sqrt(s_mult),
                                facecolors='none', edgecolors=color)
                else:
                    ax.scatter(curr[PCS[0]],
                                curr[PCS[1]],
                                curr[PCS[2]],
                                color=color, s=markersize*s_mult, marker=marker, alpha=alpha, linewidth=0)

            # draw the centroid marker for the highlighted direction only
            highlight_j = target_angle_data_unique[1]
            if cond_set[0] == highlight_j:
                ax.scatter(mean_cond[PCS[0]],
                            mean_cond[PCS[1]],
                            mean_cond[PCS[2]],
                            color=color, s=100* markersize_mult, marker='o', alpha=1, linewidth=0)

    # per-participant axis limits, scale bar geometry, and camera angle
    if participant == 'T16':
        limits = [3, 3, 3]
        center = np.array([-0.35,0,0])
        axes_length = np.array([3,-3,2.5])
        elevation = -30
        azimuth = 110
        roll = 0
    else:
        limits = [3, 3, 3]
        center = np.array([-10,10,-22.8])
        axes_length = np.array([-1.5,1.4,-2.2])
        elevation = -130
        azimuth = -45
        roll = -40

    # draw axes scale
    ax.plot([center[0],center[0]+axes_length[0]],
            [center[1],center[1]],
            [center[2],center[2]], color=(0.66,0.66,0.66), linestyle='-', linewidth=2.5, alpha=1)
    ax.plot([center[0],center[0]],
            [center[1],center[1]+axes_length[1]],
            [center[2],center[2]], color=(0.43,0.43,0.43), linestyle='-', linewidth=2.5, alpha=1)
    ax.plot([center[0],center[0]],
            [center[1],center[1]],
            [center[2],center[2]+axes_length[2]], color=(0.20,0.20,0.20), linestyle='-', linewidth=2.5, alpha=1)

    # other misc. plot settings
    ax.set_xlim([-limits[0], limits[0]])
    ax.set_ylim([-limits[1], limits[1]])
    ax.set_zlim([-limits[2], limits[2]])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.set_axis_off()
    ax.view_init(elev=elevation, azim=azimuth, roll=roll)
    ax.set_xlabel(f'PC{PCS[0]+1}')
    ax.set_ylabel(f'PC{PCS[1]+1}')
    ax.set_zlabel(f'PC{PCS[2]+1}')

    # save pdf
    savepath = os.path.join(save_plot_filepath, f'fig3_{participant}_pca3D_direction_{ID}_schem.pdf')
    plt.savefig(savepath, format='pdf')

    plt.show()


# %%
## Plot 3D PCA of initial vs. final direction tuning

# condition pairings for the three PCA plots
CONDITIONS_all = [[target_angle_data_unique, angle_diff_unique],
                  [initial_angle_data_unique, angle_diff_unique],
                  [angle_diff_unique, initial_angle_data_unique]]
CONDITION_ARRAYS_all = [[target_angle_data_all, angle_diff_all],
                        [initial_angle_data_all, angle_diff_all],
                        [angle_diff_all, initial_angle_data_all]]
ID_all = ['final', 'initial', 'curvature']

# loop over condition pairings, one 3D figure each
for CONDITIONS, CONDITION_ARRAYS, ID in zip(CONDITIONS_all, CONDITION_ARRAYS_all, ID_all):

    # build the condition-marginalized activity array
    X, _, _ = build_marg_arrays(neural_data_norm_all,
                                            CONDITIONS, CONDITION_ARRAYS,)

    # subtract the condition-independent signal (CIS)
    Xcis = X.mean(axis=(1,2))
    Xnocis = X - Xcis[:,None,None]

    # fit PCA on the delay-period, time-averaged condition means
    T_STA = int(400/TS)
    T_END = int(delay_min*1000/TS)
    Xpca = Xnocis[:,:,:,GO_CUE+T_STA:GO_CUE+T_END].mean(axis=-1)
    Xpca = Xpca.reshape((Xpca.shape[0], -1))
    pca = PCA(n_components=3)
    pca.fit(Xpca.T)
    pc_variance = pca.explained_variance_ratio_
    print(np.sum(pc_variance))
    print(pc_variance)

    # marker and line scaling
    PCS = [0,1,2]
    linewidth_mult = 0.75
    markersize_mult = 1

    # generate 3D plot
    fig = plt.figure(figsize=(10,8), facecolor='w')
    ax = fig.add_subplot(111, projection='3d')

    # project onto the top 3 PCs, then average over the delay period
    T_STA = int(GO_CUE+(400/TS))
    T_END = int(GO_CUE+delay_min*(1000/TS))
    Zmean_arr = (Xnocis[:,:,:,T_STA:T_END].T @ pca.components_.T).mean(axis=0)

    # loop over the first condition (marker shape, with its own centroid)
    for j in range(Zmean_arr.shape[1]):

        # centroid of this condition, averaged over the second condition
        mean_cond = Zmean_arr.mean(axis=0)[j]

        if ID == 'curvature':

            # loop over initial directions within this curvature group
            for i in range(Zmean_arr.shape[0]):

                cond_set = (CONDITIONS[0][j], CONDITIONS[1][i])

                # face color encodes the initial direction, edge color the curvature
                dir_angle = CONDITIONS[1][i]
                dir_angle_col = dir_angle if dir_angle >= 0 else dir_angle + 2*np.pi
                face_color = colors.hsv_to_rgb([dir_angle_col/(2*np.pi), 1, 1])
                edge_color = curvature_colors[j]

                # marker shape encode the curvature
                curv_val = CONDITIONS[0][j]
                if curv_val == angle_diff_unique[0]:
                    marker='D'
                    linewidth=2*linewidth_mult
                    markersize=35*markersize_mult
                elif curv_val == angle_diff_unique[1]:
                    marker='v'
                    linewidth=2*linewidth_mult
                    markersize=50*markersize_mult
                elif curv_val == angle_diff_unique[2]:
                    marker='o'; linewidth=2*linewidth_mult
                    markersize=40*markersize_mult
                elif curv_val == angle_diff_unique[3]:
                    marker='^'
                    linewidth=2*linewidth_mult
                    markersize=50*markersize_mult
                else:
                    marker='s'
                    linewidth=2*linewidth_mult
                    markersize=40*markersize_mult

                # PC projection of this condition pair
                curr = Zmean_arr[i,j]

                # draw a line from the centroid to the condition
                ax.plot([mean_cond[PCS[0]], curr[PCS[0]]],
                        [mean_cond[PCS[1]], curr[PCS[1]]],
                        [mean_cond[PCS[2]], curr[PCS[2]]],
                        color=edge_color, linewidth=linewidth, linestyle=':', alpha=0.5)

                # draw the condition marker
                ax.scatter(curr[PCS[0]], curr[PCS[1]], curr[PCS[2]],
                           s=markersize*0.8, marker=marker, alpha=1,
                           linewidth=1.75*linewidth_mult,
                           facecolors=face_color, edgecolors=edge_color)

            # draw the centroid marker for this curvature group
            ax.scatter(mean_cond[PCS[0]], mean_cond[PCS[1]], mean_cond[PCS[2]],
                       color=curvature_colors[j], s=100*markersize_mult, marker='o', alpha=1, linewidth=0)

        else:

            # loop over the second condition (curvature) within this direction
            for i in range(Zmean_arr.shape[0]):

                # marker shape and color encode the pair of conditions
                cond_set = (CONDITIONS[0][j], CONDITIONS[1][i])
                angle = cond_set[0]
                if j == 0:
                    angle += 2*np.pi
                angle_col = angle
                if cond_set[1] == CONDITIONS[1][0]:
                    color = colors.hsv_to_rgb([angle_col/(2*np.pi), 0.33, 1])
                    marker='D'
                    linewidth=2 * linewidth_mult
                    markersize=35* markersize_mult
                    empty = True
                elif cond_set[1] == CONDITIONS[1][1]:
                    color = colors.hsv_to_rgb([angle_col/(2*np.pi), 0.66, 1])
                    marker='v'
                    linewidth=2* linewidth_mult
                    markersize=50 * markersize_mult
                    empty = True
                elif cond_set[1] == CONDITIONS[1][2]:
                    color = colors.hsv_to_rgb([angle_col/(2*np.pi), 1, 1])
                    marker='o'
                    linewidth=2* linewidth_mult
                    markersize=40* markersize_mult
                    empty = True
                elif cond_set[1] == CONDITIONS[1][3]:
                    color = colors.hsv_to_rgb([angle_col/(2*np.pi), 1, 0.66])
                    marker='^'
                    linewidth=2* linewidth_mult
                    markersize=50* markersize_mult
                    empty = True
                elif cond_set[1] == CONDITIONS[1][4]:
                    color = colors.hsv_to_rgb([angle_col/(2*np.pi), 1, 0.33])
                    marker='s'
                    linewidth=2 * linewidth_mult
                    markersize=40* markersize_mult
                    empty = True
                # guard for any other curvature conditions; drawn as filled 'x' markers
                else:
                    color = colors.hsv_to_rgb([angle_col/(2*np.pi), 1.0, 0.5])
                    marker='x'
                    linewidth=2 * linewidth_mult
                    markersize=35* markersize_mult
                    empty = False

                # darker colors for final direction, lighter for initial direction
                if ID == 'final':
                    color = colors.hsv_to_rgb([angle_col/(2*np.pi), 1, 0.8])
                else:
                    color = colors.hsv_to_rgb([angle_col/(2*np.pi), 0.7, 1])

                # PC projection of this condition pair
                curr = Zmean_arr[i,j]

                # draw a line from the centroid to the condition
                ax.plot([mean_cond[PCS[0]], curr[PCS[0]]],
                        [mean_cond[PCS[1]], curr[PCS[1]]],
                        [mean_cond[PCS[2]], curr[PCS[2]]],
                        color=color, linewidth=linewidth, linestyle=':', alpha=0.5)

                # draw the condition marker, open or filled
                if empty:
                    ax.scatter(curr[PCS[0]],
                                curr[PCS[1]],
                                curr[PCS[2]],
                                s=markersize*0.8, marker=marker, alpha=1, linewidth=1.75*linewidth_mult,
                                facecolors='none', edgecolors=color)
                else:
                    ax.scatter(curr[PCS[0]],
                                curr[PCS[1]],
                                curr[PCS[2]],
                                color=color, s=markersize, marker=marker, alpha=1, linewidth=0)

            # draw the centroid marker for this direction
            ax.scatter(mean_cond[PCS[0]],
                        mean_cond[PCS[1]],
                        mean_cond[PCS[2]],
                        color=color, s=100* markersize_mult, marker='o', alpha=1, linewidth=0)

    # per-participant axis limits, scale bar geometry, and camera angle
    if participant == 'T16':
        limits = [3, 3, 3]
        center = np.array([-0.35,0,0])
        axes_length = np.array([3,-3,2.5])
        elevation = -30
        azimuth = 110
        roll = 0
    else:
        limits = [3, 3, 3]
        center = np.array([-10,10,-22.8])
        axes_length = np.array([-1.5,1.4,-2.2])
        elevation = -130
        azimuth = -45
        roll = -40

    # draw axes scale
    ax.plot([center[0],center[0]+axes_length[0]],
            [center[1],center[1]],
            [center[2],center[2]], color=(0.66,0.66,0.66), linestyle='-', linewidth=2.5, alpha=1)
    ax.plot([center[0],center[0]],
            [center[1],center[1]+axes_length[1]],
            [center[2],center[2]], color=(0.43,0.43,0.43), linestyle='-', linewidth=2.5, alpha=1)
    ax.plot([center[0],center[0]],
            [center[1],center[1]],
            [center[2],center[2]+axes_length[2]], color=(0.20,0.20,0.20), linestyle='-', linewidth=2.5, alpha=1)

    # misc. plot settings
    ax.set_xlim([-limits[0], limits[0]])
    ax.set_ylim([-limits[1], limits[1]])
    ax.set_zlim([-limits[2], limits[2]])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.set_axis_off()
    ax.view_init(elev=elevation, azim=azimuth, roll=roll)
    ax.set_xlabel(f'PC{PCS[0]+1}')
    ax.set_ylabel(f'PC{PCS[1]+1}')
    ax.set_zlabel(f'PC{PCS[2]+1}')

    # save pdf
    savepath = os.path.join(save_plot_filepath, f'fig3_{participant}_pca3D_direction_{ID}.pdf')
    plt.savefig(savepath, format='pdf')

    plt.show()


# %%
## Cross-validated decoding sweep

# only run the sweep when requested
if RUN_DECODING_SWEEP:

    np.random.seed(42)

    # delay window
    T_STA = int(GO_CUE+400/TS)
    T_END = int(GO_CUE+delay_min*1000/TS)

    # restore the condition arrays for fold stratification (overwritten by the PCA loop)
    CONDITIONS = [initial_angle_data_unique, angle_diff_unique]
    CONDITION_ARRAYS = [initial_angle_data_all, angle_diff_all]

    # loop over the decoded variables
    for yi, y_var in enumerate([initial_angle_data_all, target_angle_data_all, angle_diff_all]):

        # label for plot titles and filenames
        if yi == 0:
            var_title = 'Initial direction'
        elif yi == 1:
            var_title = 'Target direction'
        else:
            var_title = 'Curvature'

        # only keep trials with a long enough delay
        trial_mask = (delay_data_all >= 0.8)

        # run the cross-validated SVM sweep
        res = run_decoding_sweep(
            neural_data_norm_all, y_var, CONDITIONS, CONDITION_ARRAYS,
            trial_mask, T_STA, T_END,
        )

        # unpack the results
        y = res['y']
        true_all, pred_all = res['true_all'], res['pred_all']
        true_all_split, pred_all_split = res['true_all_split'], res['pred_all_split']
        acc_chance_all, acc_test_all = res['acc_chance_all'], res['acc_test_all']
        cv_splits = len(acc_chance_all)

        # compute fold-mean and pooled accuracy
        acc_folds = 100*np.mean([accuracy_score(true_all_split[i], pred_all_split[i]) for i in range(cv_splits)])
        acc_overall = 100*accuracy_score(true_all, pred_all)

        # compute fold-mean and pooled chance accuracy
        chance_folds = 100*np.mean(acc_chance_all)
        chance_overall = 100*np.bincount(y.ravel()).max()/np.bincount(y.ravel()).sum()

        # report the decoding results
        print(f'Fold accuracies: {acc_folds}')
        print(f'Overall accuracy: {acc_overall}')
        print('-----')
        print(f'Fold chance: {chance_folds}')
        print(f'Overall chance: {chance_overall}')

        # save decoding results
        savepath = os.path.join(save_plot_filepath, f'fig3_{participant}_decoding_{yi}.pkl')
        with open(savepath, 'wb') as f:
            pickle.dump({'true_all_split': true_all_split,
                            'pred_all_split': pred_all_split,
                            'true_all': true_all,
                            'pred_all': pred_all,
                            'acc_chance_all': acc_chance_all,
                            'acc_test_all': acc_test_all,
                            }, f)


# %%
## Plot decoding accuracy — both participants

from scipy.stats import norm

# bar colors, two shades per decoded feature
colors2 = [(0.32, 0.8, 0.32), (0.15, 0.55, 0.15),
           (0.9, 0.35, 0.35), (0.6, 0.18, 0.18),
           (0.35, 0.35, 0.9), (0.18, 0.18, 0.6),
           ]

# loop over decoded features, one plot each
for i, _ in enumerate(['Initial direction', 'Target direction', 'Curvature']):

    plt.figure(figsize=(2,10))

    # loop over participants, drawing one bar each
    for j, part in enumerate(['T16', 'T11']):

        pos = j*2.5

        # load the cached sweep results
        savepath = os.path.join(save_plot_filepath, f'fig3_{part}_decoding_{i}.pkl')
        if not os.path.exists(savepath):
            print(f'WARNING: {savepath} not found. Set RUN_DECODING_SWEEP = True to (re)generate it.')
            continue
        with open(savepath, 'rb') as f:
            decoding_data = pickle.load(f)
            true_all_split = decoding_data['true_all_split']
            pred_all_split = decoding_data['pred_all_split']
            true_all = decoding_data['true_all']
            pred_all = decoding_data['pred_all']
            acc_chance_all = decoding_data['acc_chance_all']
            acc_test_all = decoding_data['acc_test_all']

        # compute the 95% CI on the fold accuracies
        sem = np.sqrt(np.mean(acc_test_all) * (1 - np.mean(acc_test_all))) / np.sqrt(acc_test_all.shape[0] - 1)
        z = norm.ppf(0.975)
        acc_test_CI = z * sem

        # compute the mean chance level across folds
        chance_folds = np.mean(acc_chance_all)

        # plot accuracy with CI error bars, and the chance level
        plt.bar([pos], acc_test_all.mean(), yerr=acc_test_CI, label='decoding', width=1, color=colors2[2*i+j*0], )
        plt.plot([pos-0.8, pos+0.8], [acc_chance_all.mean(), acc_chance_all.mean()], 'k--')

    # misc. plot settings
    plt.xticks([0,2.5], ['T16', 'T11'], rotation=0, va='top', fontsize=20)
    plt.yticks(fontsize=16)
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.xlim([-1.5,4])
    plt.ylim([0,1])
    plt.ylabel('Accuracy', fontsize=20)

    # save pdf
    savepath = os.path.join(save_plot_filepath, f'fig3_decoding_bars_{i}.pdf')
    plt.savefig(savepath, format='pdf')

    plt.show()


# %%
## Statistical tests comparing decoding accuracies

from statsmodels.stats.proportion import proportions_ztest

# loop over participants
for j, part in enumerate(['T16', 'T11']):

    print(f'\nParticipant {part}:')

    # containers for per-feature fold accuracies and fold counts
    x = [None, None, None]
    n = [None, None, None]

    # loop over decoded features
    for i, _ in enumerate(['Initial direction', 'Target direction', 'Curvature']):

        # load the cached sweep results
        savepath = os.path.join(save_plot_filepath, f'fig3_{part}_decoding_{i}.pkl')
        if not os.path.exists(savepath):
            print(f'WARNING: {savepath} not found. Set RUN_DECODING_SWEEP = True to (re)generate it.')
            continue
        with open(savepath, 'rb') as f:

            decoding_data = pickle.load(f)
            true_all_split = decoding_data['true_all_split']
            pred_all_split = decoding_data['pred_all_split']
            true_all = decoding_data['true_all']
            pred_all = decoding_data['pred_all']
            acc_chance_all = decoding_data['acc_chance_all']
            acc_test_all = decoding_data['acc_test_all']

        # compute the per-fold accuracies
        cv_splits = len(true_all_split)
        acc_folds = [accuracy_score(true_all_split[i], pred_all_split[i]) for i in range(cv_splits)]

        # store the fold accuracies and the fold count
        x[i] = np.array(acc_folds)
        n[i] = cv_splits

    # compare each pair of features with a two-proportion z-test
    feature_names = ['Initial direction', 'Target direction', 'Curvature']
    for (i1, i2) in [(0, 1), (0, 2), (1, 2)]:

        # only compare features with cached results
        if x[i1] is None or x[i2] is None:
            print(f'\n{feature_names[i1]} vs {feature_names[i2]}: skipped (missing data)')
            continue

        # pool the two fold-accuracy means
        p1 = np.mean(x[i1])
        p2 = np.mean(x[i2])
        n1 = n[i1]
        n2 = n[i2]
        pooled_p = (p1 * n1 + p2 * n2) / (n1 + n2)

        # compute the z statistic and one-sided p-value
        stat = (p1 - p2) / np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
        pval = 1 - norm.cdf(stat)
        print(f"\n{feature_names[i1]} vs {feature_names[i2]}:")
        print("z =", stat, "p =", pval, "(computed manually)")

        # cross-check against the statsmodels implementation
        stat, pval = proportions_ztest([np.sum(x[i1]), np.sum(x[i2])], [n[i1], n[i2]], alternative='larger')
        print("z =", stat, "p =", pval)


# %%
