# %%
## Imports

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pickle
import itertools
import sklearn
import seaborn as sns
from sklearn.metrics import accuracy_score

sys.path.insert(0, '../lib')
from utils import normalize_and_concat, load_trial_data_h5
from analyses import run_decoding_sweep


# %%
## Load data

# select participant
participant = 'T16'

# whether to (re)compute the SVM decoding sweep and save .pkl, or just use the
# cached .pkl results to plot (warns if a .pkl is missing)
RUN_DECODING_SWEEP = True

# load delay-period data (aligned to the target cue)
filename_delay = f'./data/fig1c_{participant}_delay.h5'
trial_data_delay = load_trial_data_h5(filename_delay)

# load movement-period data (aligned to the go cue)
filename_move = f'./data/fig1c_{participant}_move.h5'
trial_data_move = load_trial_data_h5(filename_move)

# define save path for plots
save_plot_filepath = './plots/fig1c/'
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

# get gaze data if available
if 'gaze_data_all' in trial_data_delay:
    gaze_data_all = trial_data_delay['gaze_data_all'][~nan_mask]

# get trial info (target and start positions, delay duration, block id)
target_data_all = trial_data_delay['target_data_all'][~nan_mask]
start_data_all = trial_data_delay['start_data_all'][~nan_mask]
delay_data_all = trial_data_delay['delay_data_all'][~nan_mask]
block_id_data_all = trial_data_delay['block_id_data_all'][~nan_mask]

# get timing info (bin width in ms, and the bin index of the cues)
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

# blocks for each visual condition
blocks_cond_1 = [6, 9, 14]  # normal
blocks_cond_2 = [7, 12, 15]  # alt. visuals

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
## Plot gaze data

# only plot if gaze data is available
if 'gaze_data_all' in trial_data_delay.keys():

    _, axs = plt.subplots(2, 1, figsize=(4,8), facecolor='w')

    # gaze averaging window
    T_STA = int(GO_CUE+int(600/TS))
    T_END = int(GO_CUE+int(1000/TS))

    # loop over visual conditions
    for ib, blocks_cond in enumerate([blocks_cond_1, blocks_cond_2]):

        block_mask = np.isin(block_id_data_all, blocks_cond)
        ax = axs[ib]

        # loop over target directions
        for angle in angle_data_unique:

            # draw target circle
            x = 400 * np.cos(angle) + 1920/2
            y = 400 * np.sin(angle) + 1080/2
            radius = 60
            color = colors.hsv_to_rgb([angle/(2*np.pi), 1, 1])
            circle = plt.Circle((x, y), radius, color=color, fill=False, linestyle='-', linewidth=2)
            ax.add_artist(circle)

            # average gaze position over the window
            mask = block_mask & (angle_data_all == angle)
            gaze_tray = (gaze_data_all[mask][:,T_STA:T_END,:])
            gaze = gaze_tray.mean(axis=1)

            # compute mean and covariance ellipse of the gaze points
            ell_mean = gaze.mean(axis=0)
            ell_cov = np.cov(gaze[:,0], gaze[:,1])
            ell_eigval, ell_eigvec = np.linalg.eig(ell_cov)
            ell_eigangle = np.arctan2(ell_eigvec[1,0], ell_eigvec[0,0])

            # draw covariance ellipse
            ell = plt.matplotlib.patches.Ellipse(ell_mean, 2*np.sqrt(ell_eigval[0]), 2*np.sqrt(ell_eigval[1]), angle=ell_eigangle*180/np.pi,
                                                    color=color, fill=True, alpha=0.2, linewidth=0,)
            ax.add_artist(ell)

            # draw gaze points
            ax.scatter(gaze[:,0], gaze[:,1], s=25, c=color, alpha=0.5, linewidth=0.0)

        # misc. subplot settings
        ax.set_xlim([1920/2 - 600, 1920/2 + 600])
        ax.set_ylim([1080/2 - 600, 1080/2 + 600])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # save pdf
    savepath = os.path.join(save_plot_filepath, f'fig1c_{participant}_gaze.pdf')
    plt.savefig(savepath, format='pdf')

    plt.show()


# %%
## Plot PSTHs for example channels

# example channels to plot, per participant
if participant == 'T16':
    CHS = [49, 53, 57]
else:
    CHS = list(range(min(3, N_CHANNELS)))

# generate plot
_, axs = plt.subplots(2,len(CHS), figsize=(len(CHS)*3,4), sharex=True, sharey='col', constrained_layout=True)

# loop over example channels
for ch, ich in enumerate(CHS):

    # loop over visual conditions
    for ib, blocks_cond in enumerate([blocks_cond_1, blocks_cond_2]):

        block_mask = np.isin(block_id_data_all, blocks_cond)

        # solid lines in the top row for the normal visuals
        if ib == 0:
            ax = axs[0, ch]
            linestyle = '-'
        # dashed lines in the bottom row for the alt. visuals
        else:
            ax = axs[1, ch]
            linestyle = (0, (5, 1))

        # loop over conditions
        for cond_set in itertools.product(*CONDITIONS):

            # select trials of this direction from this condition's blocks
            masks = [cond_set[i] == CONDITION_ARRAYS[i] for i in range(len(CONDITIONS))]
            mask = np.all(masks, axis=0)
            mask = mask & block_mask

            # delay period plotting window: -200 ms to the end of the shortest delay
            T_STA = int(GO_CUE+(-200/TS))
            T_END = int(GO_CUE+(delay_min*1000/TS))
            t_plot = t_data[T_STA:T_END]

            # compute mean and SEM firing rate across trials
            data_plot = neural_data_all[mask][:,T_STA:T_END,ich]
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

    # get y limits from the top row of this channel
    ylims = axs[0,ch].get_ylim()

    # loop over both rows of this channel
    for i, ax in enumerate(axs[:,ch]):

        # misc. subplot settings
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])

        # scale bar offset
        scale_offset = [-20, -2]
        ax.set_xlim(-300+scale_offset[0], 1100)
        ax.set_ylim(ylims[0]+scale_offset[1], ylims[1])

        # draw target cue (yellow)
        ax.plot([0,0],[ylims[0], ylims[1]], color='y', linestyle='-', linewidth=2)

        # only draw the rate and time scale bars on the bottom row
        if i == 1:
            ax.plot([-300+scale_offset[0]/3, -300+scale_offset[0]/3],
                    [ylims[0]+scale_offset[1]/2, ylims[0]+scale_offset[1]/2+2],
                    color='k', linestyle='-', linewidth=5)
            ax.plot([-300+scale_offset[0]/3, -300+scale_offset[0]/3+200],
                    [ylims[0]+scale_offset[1]/2, ylims[0]+scale_offset[1]/2],
                    color='k', linestyle='-', linewidth=5)

    # report the plotted channel and the trial counts behind each subplot
    print(f'ich: {ich}, actual channel: {keep_chans_all[ich]}')
    print(f'modality 0 trials: {((np.isin(block_id_data_all, blocks_cond_1))).sum()}')
    print(f'modality 1 trials: {((np.isin(block_id_data_all, blocks_cond_2))).sum()}')

# save pdf
savepath = os.path.join(save_plot_filepath, f'fig1c_{participant}_psths.pdf')
plt.savefig(savepath, format='pdf')

plt.show()


# %%
## Normalize neural data

# soft-normalization constants for the spike and SBP features
SOFT_NORM_SPIKES = 1e-9
SOFT_NORM_SBP = 1e-9

# soft-normalize per block and optionally concatenate the spike and SBP features
neural_data_norm_all, _ = normalize_and_concat(
    neural_data_all, sbp_data_all, neural_data_move_all, sbp_data_move_all,
    block_id_data_all, GO_CUE, TS,
    USE_SBP=True, SOFT_NORM_SPIKES=SOFT_NORM_SPIKES, SOFT_NORM_SBP=SOFT_NORM_SBP,
)


# %%
## Cross-validated decoding sweep

np.random.seed(42)

# delay window: 400 ms after the target cue to the end of the shortest delay
T_STA = int(GO_CUE+400/TS)
T_END = int(GO_CUE+delay_min*1000/TS)

# loop over visual conditions
for ib, blocks_cond in enumerate([blocks_cond_1, blocks_cond_2]):

    # select trials from this condition's blocks with a long enough delay
    block_mask = np.isin(block_id_data_all, blocks_cond)
    trial_mask = (delay_data_all >= 0.8) & block_mask

    # pkl path for caching the decoding sweep results
    pkl_path = os.path.join(save_plot_filepath, f'fig1c_{participant}_decoding_{ib}.pkl')

    # run the sweep and cache it, or load the cached results
    if RUN_DECODING_SWEEP:
        res = run_decoding_sweep(
            neural_data_norm_all, angle_data_all, CONDITIONS, CONDITION_ARRAYS,
            trial_mask, T_STA, T_END,
        )
        with open(pkl_path, 'wb') as f:
            pickle.dump(res, f)
    else:
        # if results are not cached, warn the user and skip this visual condition
        if not os.path.exists(pkl_path):
            print(f'WARNING: {pkl_path} not found. Set RUN_DECODING_SWEEP = True to (re)generate it.')
            continue
        # load the cached results from the pickle file
        with open(pkl_path, 'rb') as f:
            res = pickle.load(f)

    # get the condition labels, the true and predicted labels, and the chance accuracies
    y = res['y']
    true_all, pred_all = res['true_all'], res['pred_all']
    true_all_split, pred_all_split = res['true_all_split'], res['pred_all_split']
    acc_chance_all = res['acc_chance_all']
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

    # generate plot
    plt.figure(figsize=(6,4.5))

    # plot the SVM confusion matrix
    ax = sns.heatmap(sklearn.metrics.confusion_matrix(true_all,
                                                    pred_all,
                                                    normalize='true',
                                                        ),
                                                        cmap='viridis',
                                                    annot=True,
                                                    annot_kws={"size": 16},
                                                    fmt= '.1g',
                                                    vmin=0,
                                                    vmax=1,cbar=True)

    # misc. plot settings
    ax.set_xlabel('Predicted', fontsize=18)
    ax.set_ylabel('True', fontsize=18)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    plt.gca().invert_yaxis()
    plt.title(f'Decoding acc.: {acc_folds:2.1f}%', fontsize=20)

    # save pdf
    savepath = os.path.join(save_plot_filepath, f'fig1c_{participant}_decoding_svm_{ib}.pdf')
    plt.savefig(savepath, format='pdf')

    plt.show()

# %%
